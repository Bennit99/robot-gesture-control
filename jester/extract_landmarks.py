"""Parallel Jester dataset landmark extraction using multiprocessing.

Architecture
------------
* **Main process** - gathers video folder paths, manages the process pool,
  and displays progress.  No inference happens here.
* **Worker processes** - each one owns an isolated ``HandLandmarker``
  (VIDEO mode, GPU delegate with CPU fallback).  Workers decode JPEGs with
  OpenCV and run inference frame-by-frame, resetting temporal tracking state
  between videos.

Concurrency
-----------
Uses ``concurrent.futures.ProcessPoolExecutor`` (not threads) to bypass the
GIL.  Worker count is auto-detected from available CPUs / RAM but can be
overridden via ``--workers``.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import sys
import threading
import time

# Suppress noisy TensorFlow / MediaPipe C++ warnings
# (inference_feedback_manager "Disabling support for feedback tensors")
os.environ["GLOG_minloglevel"] = "2"  # hide INFO & WARNING from glog
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide TF C++ warnings
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1" if os.name == "nt" else "0"
os.environ["ABSL_MIN_LOG_LEVEL"] = "2"  # absl: suppress WARNING and below
os.environ["GRPC_VERBOSITY"] = "ERROR"  # gRPC noise
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

# Optional: progress bar -----------------------------------------------------------
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Optional: system monitoring ------------------------------------------------------
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

log = logging.getLogger("jester")

# -- Constants -----------------------------------------------------------------
JESTER_FPS = 12
FRAME_DURATION_MS = 1000.0 / JESTER_FPS

DEFAULT_DATA_DIR = "./data/jester/images"
DEFAULT_OUTPUT_DIR = "./data/jester/landmarks"
DEFAULT_MODEL_PATH = "models/hand_landmarker.task"


# ==============================================================================
# Model profiling (estimation for MediaPipe HandLandmarker)
# ==============================================================================
def estimate_hand_landmarker_flops(
    input_height: int = 256, input_width: int = 256, num_hands: int = 2
) -> tuple[float, float]:
    """Estimate parameter count and FLOPs for a typical MediaPipe HandLandmarker.

    MediaPipe hand landmark models typically include:
    - Hand detection (lightweight CNN or EfficientDet-style backbone)
    - Per-hand landmark regression (MobileNet-style encoder)
    - Typical model size: 20-40 MB, ~4-7M parameters

    Parameters
    ----------
    input_height, input_width : int
        Input image dimensions (typically 256x256 or 192x192)
    num_hands : int
        Maximum number of hands tracked (affects landmark branch)

    Returns
    -------
    parameters : float - estimated parameter count
    flops : float - estimated FLOPs per inference
    """
    # Typical MediaPipe HandLandmarker (as of 2024) architecture estimate
    # Based on model size (32 MB) and common architectures

    # Hand detection branch (EfficientNet-B0 backbone)
    detection_params = 3.2e6  # ~3.2M params
    detection_flops = (
        input_height * input_width * 24 * 3  # Initial conv layers
        + input_height * input_width * 24  # Squeeze-excitation
        + (input_height // 4) * (input_width // 4) * 64 * 24  # Feature pyramid
    )

    # Hand presence classifier
    presence_params = 0.1e6  # ~0.1M params
    presence_flops = 1024  # Small classifier

    # Landmark regression branch per hand
    landmark_params = 1.5e6  # ~1.5M per hand
    landmark_flops = (
        num_hands * 192 * 192 * 16  # Encoder path
        + num_hands * 1024 * 21 * 3  # 21 landmarks with (x, y, z)
    )

    # Total estimate
    total_params = detection_params + presence_params + landmark_params
    total_flops = detection_flops + presence_flops + landmark_flops

    return total_params, total_flops


def get_model_info(model_path: str) -> dict:
    """Extract available metadata from the MediaPipe .task file.

    Returns dict with 'size_mb', 'parameters_est', 'flops_est' keys.
    """
    info = {}
    if os.path.exists(model_path):
        size_bytes = os.path.getsize(model_path)
        info["size_mb"] = size_bytes / (1024**2)

    params, flops = estimate_hand_landmarker_flops()
    info["parameters_est"] = params
    info["flops_est"] = flops
    return info


# ==============================================================================
# Worker-process state  (global within each *spawned* worker; never shared)
# ==============================================================================
_landmarker: vision.HandLandmarker | None = None
_model_path: str = ""
_use_gpu: bool = True
_delegate_name: str = "unknown"


def _effective_use_gpu(use_gpu: bool) -> bool:
    """Return whether GPU should be used on this platform."""
    # MediaPipe GPU delegate for this pipeline is unsupported on Windows.
    return bool(use_gpu) and os.name != "nt"


def _create_landmarker() -> vision.HandLandmarker:
    """Create a fresh ``HandLandmarker`` in the *current* process."""
    global _use_gpu, _delegate_name

    delegate = (
        mp_python.BaseOptions.Delegate.GPU
        if _use_gpu
        else mp_python.BaseOptions.Delegate.CPU
    )
    base_opts = mp_python.BaseOptions(
        model_asset_path=_model_path,
        delegate=delegate,
    )
    opts = vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=VisionTaskRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    lm = vision.HandLandmarker.create_from_options(opts)
    _delegate_name = "GPU" if _use_gpu else "CPU"
    return lm


def _create_landmarker_quiet() -> vision.HandLandmarker:
    """Create a HandLandmarker while suppressing C++ stderr spam.

    MediaPipe's inference_feedback_manager emits warnings directly to
    stderr from C++ code.  No env var reliably silences them on all
    platforms, so we temporarily redirect the file descriptor.
    """
    # Save the real stderr fd and redirect it to devnull
    _stderr_fd = sys.stderr.fileno()
    _saved_fd = os.dup(_stderr_fd)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, _stderr_fd)
        os.close(devnull)
        return _create_landmarker()
    finally:
        # Restore real stderr
        os.dup2(_saved_fd, _stderr_fd)
        os.close(_saved_fd)


def _suppress_mediapipe_warnings() -> None:
    """Ensure C++ log suppression env vars are set in this (worker) process."""
    os.environ["GLOG_minloglevel"] = "2"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["ABSL_MIN_LOG_LEVEL"] = "2"
    os.environ["GRPC_VERBOSITY"] = "ERROR"


def _worker_init(model_path: str, use_gpu: bool) -> None:
    """``ProcessPoolExecutor`` *initializer* - called once per worker process."""
    global _landmarker, _model_path, _use_gpu

    _suppress_mediapipe_warnings()

    _model_path = model_path
    _use_gpu = _effective_use_gpu(use_gpu)

    try:
        _landmarker = _create_landmarker_quiet()
    except Exception as exc:
        if use_gpu:
            log.warning(
                "PID %d: GPU delegate failed (%s) \u2013 falling back to CPU",
                os.getpid(),
                exc,
            )
            _use_gpu = False
            _landmarker = _create_landmarker_quiet()
        else:
            raise


def _reset_landmarker() -> None:
    """Close and re-create the landmarker to flush VIDEO-mode temporal state.

    This is **mandatory** between videos: MediaPipe's VIDEO mode tracks hand
    positions across frames, so the first frame of Video B would be corrupted
    by tracking history from Video A.
    """
    global _landmarker

    if _landmarker is not None:
        try:
            _landmarker.close()
        except Exception:
            pass
    _landmarker = _create_landmarker_quiet()


# ==============================================================================
# DataFrame builder
# ==============================================================================
_LANDMARK_COLS: list[str] = []


def _get_landmark_cols() -> list[str]:
    """Lazily build and cache the column name list."""
    global _LANDMARK_COLS
    if not _LANDMARK_COLS:
        cols = ["frame", "hand_index", "handedness"]
        for i in range(21):
            cols.extend([f"x{i}", f"y{i}", f"z{i}"])
        _LANDMARK_COLS = cols
    return _LANDMARK_COLS


def _build_landmark_df(results: list, use_world: bool = False) -> pd.DataFrame:
    """Convert a list of ``HandLandmarkerResult`` into a tidy DataFrame."""
    cols = _get_landmark_cols()
    rows: list[list] = []
    for fidx, r in enumerate(results):
        if r is None or not r.handedness:
            continue
        lms = r.hand_world_landmarks if use_world else r.hand_landmarks
        for hidx, (hand, lm) in enumerate(zip(r.handedness, lms)):
            hb = 0 if hand[0].category_name == "Left" else 1
            row: list = [fidx, hidx, hb]
            for pt in lm:
                row.extend([pt.x, pt.y, pt.z])
            rows.append(row)
    return pd.DataFrame(rows, columns=cols)


# ==============================================================================
# Single-video worker task
# ==============================================================================
def _process_video_task(args: tuple) -> dict:
    """Process one Jester video folder: read JPEGs  infer  save Parquet.

    Designed to run inside a worker process.  The worker-local
    ``_landmarker`` is reused across calls but **reset after every video**
    to prevent cross-video temporal bleed.

    Parameters (packed as a tuple for ``pool.submit`` compatibility)
    ----------------------------------------------------------------
    video_rel_path : str - relative path from *data_dir* to the video folder
    data_dir       : str - root image directory
    output_dir     : str - root output directory

    Returns
    -------
    dict  with keys: ``video``, ``status``, ``frames``, ``hands``, ``error``
    """
    global _landmarker
    video_rel_path, data_dir, output_dir = args

    folder_path = os.path.join(data_dir, video_rel_path)
    # Flatten output: use only the leaf folder name (video ID)
    # e.g. "4/107946"  output_dir/107946/
    video_id = os.path.basename(video_rel_path)
    out_path = os.path.join(output_dir, video_id)

    lm_file = os.path.join(out_path, "landmarks.parquet")
    wlm_file = os.path.join(out_path, "world_landmarks.parquet")

    # -- Resume: skip if both output files already exist ------------------
    if os.path.exists(lm_file) and os.path.exists(wlm_file):
        return {"video": video_rel_path, "status": "skipped"}

    try:
        # -- Load sorted JPEG frames --------------------------------------
        image_files = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg"))
        )
        if not image_files:
            return {"video": video_rel_path, "status": "empty", "frames": 0}

        # -- Inference ----------------------------------------------------
        results: list = []
        for idx, fname in enumerate(image_files):
            frame = cv2.imread(os.path.join(folder_path, fname))
            if frame is None:
                results.append(None)
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(idx * FRAME_DURATION_MS)
            results.append(_landmarker.detect_for_video(mp_img, ts_ms))

        # -- CRITICAL: reset temporal tracking after every video ----------
        _reset_landmarker()

        # -- Persist Parquet files ----------------------------------------
        os.makedirs(out_path, exist_ok=True)
        _build_landmark_df(results, use_world=False).to_parquet(lm_file, index=False)
        _build_landmark_df(results, use_world=True).to_parquet(wlm_file, index=False)

        n_frames = len(results)
        n_hands = sum(
            len(r.handedness) for r in results if r is not None and r.handedness
        )

        return {
            "video": video_rel_path,
            "status": "processed",
            "frames": n_frames,
            "hands": n_hands,
        }

    except Exception as exc:
        # Re-init landmarker so the next video starts clean
        try:
            _reset_landmarker()
        except Exception:
            pass
        return {
            "video": video_rel_path,
            "status": "error",
            "error": str(exc),
        }


# ==============================================================================
# Public API
# ==============================================================================
def process_video(
    video_id: str,
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    use_gpu: bool = True,
) -> dict:
    """Process a **single** Jester video (standalone, no pool required).

    Creates and tears down its own MediaPipe instance.  For batch
    processing prefer :func:`run_parallel`.

    Parameters
    ----------
    video_id   : video folder name or relative path (e.g. ``"107946"`` or ``"4/107946"``)
    data_dir   : root directory containing video sub-folders of JPEG frames
    output_dir : root directory to write per-video Parquet files
    model_path : path to ``hand_landmarker.task``
    use_gpu    : request GPU delegate (falls back to CPU on failure)

    Returns
    -------
    dict  - ``{"video": ..., "status": "processed"|"skipped"|"error", ...}``
    """
    global _landmarker

    _worker_init(model_path, _effective_use_gpu(use_gpu))
    try:
        return _process_video_task((video_id, data_dir, output_dir))
    finally:
        if _landmarker is not None:
            try:
                _landmarker.close()
            except Exception:
                pass
            _landmarker = None


def gather_video_folders(data_dir: str) -> list[str]:
    """Walk *data_dir* and return relative paths to every folder containing JPEGs.

    Handles arbitrary nesting (e.g. ``images/4/107946/``  ``"4/107946"``).
    """
    videos: list[str] = []
    for root, _dirs, files in os.walk(data_dir):
        if any(f.lower().endswith((".jpg", ".jpeg")) for f in files):
            videos.append(os.path.relpath(root, data_dir))
    return sorted(videos)


def get_optimal_workers() -> int:
    """Auto-detect a sensible worker count from CPUs and available RAM."""
    cpus = os.cpu_count() or 4

    if HAS_PSUTIL:
        avail_gb = psutil.virtual_memory().available / (1024**3)
        mem_limited = max(1, int(avail_gb / 0.5))
        return max(1, min(cpus, mem_limited))

    # Without psutil, be conservative
    return max(1, cpus - 1)


# -- System monitor (background thread) ---------------------------------------
def _system_monitor(stop: threading.Event, interval: float = 30.0) -> None:
    """Print CPU / RAM usage every *interval* seconds until *stop* is set."""
    if not HAS_PSUTIL:
        return
    while not stop.wait(interval):
        cpu = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        print(
            f"  [Monitor] CPU {cpu:4.0f}% | "
            f"RAM {mem.used / 1024**3:.1f} / {mem.total / 1024**3:.1f} GB "
            f"({mem.percent:.0f}%)"
        )


# ==============================================================================
# Parallel orchestrator
# ==============================================================================
def run_parallel(
    data_dir: str = DEFAULT_DATA_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    *,
    max_workers: int | None = None,
    use_gpu: bool = True,
    single_video: str | None = None,
) -> None:
    """Process Jester videos in parallel with ``ProcessPoolExecutor``.

    * Resumes interrupted runs: skips folders where both output Parquet
      files already exist.
    * Shows a ``tqdm`` progress bar if installed; falls back to periodic
      percentage prints otherwise.
    * Logs system resource usage every 30 s (requires ``psutil``).

    Parameters
    ----------
    data_dir     : root folder with video sub-folders of JPEG frames
    output_dir   : root folder to write per-video landmark Parquet files
    model_path   : path to ``hand_landmarker.task``
    max_workers  : override auto-detected worker count (``None`` = auto)
    use_gpu      : request GPU delegate (falls back to CPU on failure)
    single_video : process just this one relative path (``None`` = all)
    """
    # -- Discover work -----------------------------------------------------
    if single_video:
        folders = [single_video]
    else:
        folders = gather_video_folders(data_dir)

    total = len(folders)
    if total == 0:
        print(f"No video folders found in {data_dir}")
        return

    already_done = sum(
        1
        for v in folders
        if os.path.exists(
            os.path.join(output_dir, os.path.basename(v), "landmarks.parquet")
        )
        and os.path.exists(
            os.path.join(output_dir, os.path.basename(v), "world_landmarks.parquet")
        )
    )
    remaining = total - already_done
    workers = max_workers or get_optimal_workers()
    effective_gpu = _effective_use_gpu(use_gpu)

    # -- Banner ------------------------------------------------------------
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Jester Landmark Extraction Pipeline")
    print(sep)
    print(f"  Data dir  : {os.path.abspath(data_dir)}")
    print(f"  Output dir: {os.path.abspath(output_dir)}")
    print(f"  Model     : {model_path}")

    # Model profiling info
    model_info = get_model_info(model_path)
    if "size_mb" in model_info:
        print(f"  Model size: {model_info['size_mb']:.1f} MB")
    print(
        f"  Model params (est.): {model_info['parameters_est'] / 1e6:.2f}M | "
        f"FLOPs (est.): {model_info['flops_est'] / 1e9:.3f}G"
    )

    print(f"  Videos    : {total} total | {already_done} done | {remaining} to process")
    print(f"  Workers   : {workers}")
    print(f"  Delegate  : {'GPU' if effective_gpu else 'CPU'}")
    if HAS_PSUTIL:
        m = psutil.virtual_memory()
        print(
            f"  RAM       : {m.available / 1024**3:.1f} GB free / "
            f"{m.total / 1024**3:.1f} GB total"
        )
        print(f"  CPUs      : {os.cpu_count()}")
    print(sep)

    if remaining == 0:
        print("All videos already processed. Nothing to do.\n")
        return

    # -- Build task list ---------------------------------------------------
    tasks = [(v, data_dir, output_dir) for v in folders]

    # -- Start resource monitor --------------------------------------------
    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(
        target=_system_monitor, args=(monitor_stop,), daemon=True
    )
    monitor_thread.start()

    # -- Execute -----------------------------------------------------------
    t0 = time.time()
    processed = skipped = errors = total_frames = total_hands = 0
    cancelled = False

    # Use 'spawn' context explicitly for safe GPU / MediaPipe behaviour
    ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(model_path, effective_gpu),
        mp_context=ctx,
    ) as pool:
        futures = {pool.submit(_process_video_task, t): t for t in tasks}

        if HAS_TQDM:
            pbar = tqdm(
                total=total,
                desc="Extracting landmarks",
                unit="vid",
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                ),
            )
        else:
            pbar = None

        try:
            for future in as_completed(futures):
                res = future.result()
                st = res.get("status")

                if st == "processed":
                    processed += 1
                    total_frames += res.get("frames", 0)
                    total_hands += res.get("hands", 0)
                elif st == "skipped":
                    skipped += 1
                else:  # "error" or "empty"
                    errors += 1
                    if st == "error":
                        log.warning(
                            "Error in %s: %s",
                            res.get("video"),
                            res.get("error", "?"),
                        )

                if pbar:
                    pbar.set_postfix_str(
                        f"ok={processed} skip={skipped} err={errors}",
                        refresh=False,
                    )
                    pbar.update(1)
                else:
                    done = processed + skipped + errors
                    if done % 200 == 0 or done == total:
                        pct = 100 * done / total
                        elapsed = time.time() - t0
                        print(
                            f"  [{pct:5.1f}%]  processed={processed}  "
                            f"skipped={skipped}  errors={errors}  "
                            f"({elapsed:.0f}s)"
                        )

        except KeyboardInterrupt:
            cancelled = True
            print("\n\n  Ctrl+C received - shutting down gracefully...")
            print("  (already-saved videos are safe, cancelling pending work)")

            # Cancel all futures that haven't started yet
            n_cancelled = 0
            for f in futures:
                if f.cancel():
                    n_cancelled += 1
            print(f"  Cancelled {n_cancelled} pending tasks.")

            # Shut down the pool without waiting for running tasks
            pool.shutdown(wait=False, cancel_futures=True)

        finally:
            if pbar:
                pbar.close()

    monitor_stop.set()
    elapsed = time.time() - t0

    # -- Summary -----------------------------------------------------------
    label = "INTERRUPTED" if cancelled else "COMPLETE"
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  Wall time  : {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
    print(f"  Processed  : {processed}")
    print(f"  Skipped    : {skipped}")
    print(f"  Errors     : {errors}")
    print(f"  Frames     : {total_frames}")
    print(f"  Hands found: {total_hands}")
    if processed > 0:
        print(
            f"  Throughput : {processed / elapsed:.1f} vid/s | "
            f"{total_frames / elapsed:.0f} frame/s"
        )
    if cancelled:
        print(f"  Remaining  : ~{total - processed - skipped - errors} videos")
        print("  (Re-run to resume - completed videos will be skipped)")
    print(f"{sep}\n")


# ==============================================================================
# CLI
# ==============================================================================
def cli() -> None:
    """Command-line entry point (``python -m jester``)."""
    import argparse

    p = argparse.ArgumentParser(
        description="Extract hand landmarks from the Jester dataset in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default=DEFAULT_DATA_DIR,
        help="Root dir of JPEG video folders",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help="Root output dir for Parquet files",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="MediaPipe hand_landmarker.task path",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes (default: auto-detect from CPUs/RAM)",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU delegate (skip GPU attempt)",
    )
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="Process a single video (relative path or folder name)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help="Print model profiling info (parameters, FLOPs estimate) and exit",
    )

    args = p.parse_args()

    if args.profile:
        info = get_model_info(args.model)
        print("\nMediaPipe HandLandmarker Model Profiling")
        print("=" * 50)
        if "size_mb" in info:
            print(f"  Model file size: {info['size_mb']:.1f} MB")
        print(f"  Parameters (est.): {info['parameters_est'] / 1e6:.2f}M")
        print(f"  FLOPs per inference (est.): {info['flops_est'] / 1e9:.3f} GFLOPs")
        print("=" * 50)
        print()
        return

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    run_parallel(
        data_dir=args.data,
        output_dir=args.output,
        model_path=args.model,
        max_workers=args.workers,
        use_gpu=not args.cpu,
        single_video=args.video,
    )
