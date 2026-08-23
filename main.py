import platform
import sys
import time
import traceback

import setproctitle

from cfg import cfg
from cfg.options_loader import load_recognizer_options
from data_loader.data_loader import load_data
from web_ui.plot import plot
from web_ui.stats import save_results
from utils.mediapipe_extensions import (
    _wrap_result,
    create_recognizer,
    is_landmark_model,
)


def run_once(recognizer):
    """Run a single recognition session. Returns (images, results)."""
    images = []
    results = []

    try:
        for image, frame in load_data(cfg.data):
            # In LIVE_STREAM mode, check for stop request from frontend
            if cfg.running_mode == "LIVE_STREAM":
                from web_ui.streaming_web_server.backend import stop_requested

                if stop_requested.is_set():
                    print("\nStream stopped by frontend. Saving stats...")
                    break

            # In LIVE_STREAM mode, skip frames if MediaPipe is still processing
            if cfg.running_mode == "LIVE_STREAM":
                from engine.callback import recognizer_ready

                if not recognizer_ready.is_set():
                    continue  # Drop this frame, MediaPipe is still busy
                recognizer_ready.clear()  # Mark as busy before sending

            result = recognizer.run_auto(image, timestamp_ms=frame)

            if result:  # for IMAGE and VIDEO modes; LIVE_STREAM is handled in callback
                result = _wrap_result(result)
                if cfg.verbose:
                    print(", ".join(result.summary_str()))

                images.append(image)
                results.append(result)
    except KeyboardInterrupt:
        print("\nStream interrupted by user (Ctrl+C). Saving stats...")

    return images, results


def _has_dynamic_model() -> bool:
    """True when cfg.dynamic_model is configured with a checkpoint."""
    dm = getattr(cfg, "dynamic_model", None)
    # Accept either legacy 'checkpoint' key or the more common 'model' key
    return isinstance(dm, dict) and bool(dm.get("checkpoint") or dm.get("model"))


def _is_live_stream() -> bool:
    return cfg.running_mode == "LIVE_STREAM"


def _start_web_stack_if_needed() -> None:
    if not _is_live_stream():
        return
    from web_ui.streaming_web_server.backend import start_backend
    from web_ui.streaming_web_server.frontend.run import start_frontend

    start_backend()
    start_frontend()


def _wait_for_start_signal() -> bool:
    from web_ui.streaming_web_server.backend import start_requested

    print("Waiting for /start from web app (or Ctrl+C to exit)...")
    try:
        while not start_requested.is_set():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nExiting...")
        return False

    start_requested.clear()
    return True


def _reset_livestream_runtime_state() -> None:
    from engine.callback import recognizer_ready

    recognizer_ready.set()
    if not is_landmark_model():
        from engine.callback import gesture_filter_left, gesture_filter_right

        gesture_filter_left.reset()
        gesture_filter_right.reset()


def main():
    setproctitle.setproctitle("gesture")
    _start_web_stack_if_needed()

    # Stacked pipeline: MediaPipe landmarks → PyTorch dynamic gesture model
    if _has_dynamic_model():
        from engine.dynamic_recognizer import run_stacked_pipeline

        run_stacked_pipeline()
        return

    recognizer = create_recognizer(load_recognizer_options())
    print(f"Using model: {cfg.model}")

    web_only = bool(getattr(cfg, "livestream_web_only", False))
    if _is_live_stream() and web_only:
        if not _wait_for_start_signal():
            return

    while True:
        images, results = run_once(recognizer)

        save_results(results)

        if not is_landmark_model() and cfg.verbose and platform.system() == "Windows":
            plot(images, results)

        if not _is_live_stream():
            break  # Non-livestream modes run once

        if not _wait_for_start_signal():
            break

        _reset_livestream_runtime_state()
        print("\n--- Starting stream ---\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("Critical error in main thread:")
        traceback.print_exc()
        sys.exit(1)
