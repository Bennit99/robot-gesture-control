import os
import platform
import time
from collections.abc import Iterator

import cv2
import mediapipe as mp

from cfg import cfg
from utils.timeTracker import start_next_iteration, time_me


def load_data(data_dir: str = "./data") -> Iterator[tuple[mp.Image, int | None]]:
    """Generator that yields (MediaPipe Image, timestamp ms | None) for images or video or livestream frames based on running mode."""
    if cfg.running_mode == "IMAGE":
        for img in load_images(data_dir):
            yield img, None
    elif cfg.running_mode == "VIDEO":
        yield from load_video_frames(data_dir)
    elif cfg.running_mode == "LIVE_STREAM":
        yield from stream_webcam()


def load_video_frames(data_dir: str = "./data") -> Iterator[tuple[mp.Image, int]]:
    """Generator that yields (MediaPipe Image, frame timestamp ms) for each frame in all videos in a directory."""
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(video_extensions):
                video_path = os.path.join(root, filename)
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Warning: Cannot open video file: {video_path}")
                    continue
                fps = cap.get(cv2.CAP_PROP_FPS)
                if cfg.video_fps != fps:
                    cfg.video_fps = fps
                    print(
                        f"Detected video FPS: {fps}, updating cfg.video_fps to {cfg.video_fps}"
                    )
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=frame_rgb
                    )
                    # Calculate timestamp in ms
                    frame_timestamp_ms = (
                        int((frame_idx / fps) * 1000) if fps > 0 else frame_idx
                    )
                    yield mp_image, frame_timestamp_ms
                    frame_idx += 1
                cap.release()


def load_images(data_dir: str = "./data") -> Iterator[mp.Image]:
    """Generator that yields images from the specified directory one by one."""
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(root, filename)
                yield mp.Image.create_from_file(image_path)


def stream_webcam() -> Iterator[tuple[mp.Image, int]]:
    """Yields (MediaPipe Image, frame timestamp ms) from webcam or stream in real time, throttled to max_fps."""

    cap = None

    # 1) Linux: try /dev/video0 directly (Jetson / native Linux)
    if platform.system() == "Linux":
        print("Trying /dev/video0 ...")
        for attempt in range(5):
            cap = cv2.VideoCapture("/dev/video0")
            if cap.isOpened():
                print(f"  -> opened /dev/video0 on attempt {attempt + 1}")
                break
            time.sleep(2)  # Give camera time to initialize
            print(f"  -> failed, retrying ({attempt + 2}/5) in 2s...")

    # 2) Windows / fallback: default camera index 0 with DirectShow
    if cap is None or not cap.isOpened():
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        print("Trying default camera index 0 ...")
        for attempt in range(3):
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                print("  -> opened camera 0")
                break
            cap.release()
            if attempt < 2:
                print(f"  -> failed, retrying ({attempt + 2}/3)...")
                time.sleep(1)

    # 3) UDP stream (Docker on Windows -> ffmpeg bridge)
    if not cap.isOpened():
        print(
            "No local camera found. Trying UDP stream at host.docker.internal:1234 ..."
        )
        cap = cv2.VideoCapture("udp://host.docker.internal:1234")
        if cap.isOpened():
            print("  -> opened UDP stream")

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open any camera source.\n"
            "  • Linux:   check /dev/video0 permissions\n"
            "  • Windows: check that no other app is using the camera\n"
            "  • Docker:  ensure the ffmpeg UDP bridge is running"
        )

    # Reduce motion blur: disable auto-exposure first, then set exposure and gain
    if hasattr(cfg, "stream_exposure"):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure mode
        time.sleep(0.1)  # Give camera time to switch mode
        cap.set(cv2.CAP_PROP_EXPOSURE, cfg.stream_exposure)
        print(f"Setting camera exposure to {cfg.stream_exposure}")
    if hasattr(cfg, "stream_brightness"):
        cap.set(cv2.CAP_PROP_BRIGHTNESS, cfg.stream_brightness)
        print(f"Setting camera brightness to {cfg.stream_brightness}")
    print(
        f"Camera exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}, "
        f"brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}, "
        f"auto exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}"
    )

    min_frame_time = 1.0 / cfg.stream_fps if cfg.stream_fps > 0 else 0
    try:
        while True:
            start_next_iteration("idle / wait")
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from webcam or video stream.")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            frame_timestamp_ms = int(time.time() * 1000)
            time_me("frame_loading")
            yield mp_image, frame_timestamp_ms
            elapsed = time.time() - start_time
            sleep_time = min_frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        print("Releasing video capture.")
        cap.release()
