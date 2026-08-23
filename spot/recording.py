#!/usr/bin/env python3
"""
Record Spot body cameras to separate MP4 files (Python SDK only, no ROS).
Optionally produce a stitched front panorama from front-left/right.

Usage:
  python spot/recording.py --robot 192.168.10.3 --out-dir data/spot_captures --fps 10 --seconds 60 --stitch-front
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import bosdyn.client
import bosdyn.client.util
import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient
from dotenv import load_dotenv

# Body camera image source names
FRONT_LEFT = "frontleft_fisheye_image"
FRONT_RIGHT = "frontright_fisheye_image"
LEFT = "left_fisheye_image"
RIGHT = "right_fisheye_image"
BACK = "back_fisheye_image"

ALL_SOURCES = [FRONT_LEFT, FRONT_RIGHT]


def topic_to_filename(name: str) -> str:
    return name.replace("/", "_")


def decode_image_response(image_response):
    """
    Fallback decoder if bosdyn.client.image_helper(s) is missing.
    Converts bosdyn.api.ImageResponse -> numpy array (OpenCV style).
    Works for JPEG, PNG, RAW, and depth_uint16.
    """
    img_data = image_response.shot.image.data
    fmt = image_response.shot.image.format
    pixel_format = image_response.shot.image.pixel_format

    # Most Spot camera feeds are JPEGs or PNGs
    if (
        fmt == image_response.shot.image.FORMAT_JPEG
        or fmt == image_response.shot.image.FORMAT_PNG
    ):
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        return img

    # Handle raw uncompressed data (rare for body cameras)
    if pixel_format == image_response.shot.image.PIXEL_FORMAT_RGB_U8:
        img = np.frombuffer(img_data, dtype=np.uint8)
        img = img.reshape(
            image_response.shot.image.rows, image_response.shot.image.cols, 3
        )
        return img
    if pixel_format == image_response.shot.image.PIXEL_FORMAT_GREYSCALE_U8:
        img = np.frombuffer(img_data, dtype=np.uint8)
        img = img.reshape(
            image_response.shot.image.rows, image_response.shot.image.cols
        )
        return img
    if pixel_format == image_response.shot.image.PIXEL_FORMAT_DEPTH_U16:
        img = np.frombuffer(img_data, dtype=np.uint16)
        img = img.reshape(
            image_response.shot.image.rows, image_response.shot.image.cols
        )
        return img

    # Unknown format
    raise ValueError(f"Unknown Spot image format: {fmt}, pixel_format: {pixel_format}")


class VideoSink:
    """Lazy-initialized cv2.VideoWriter that fixes frame size after first frame."""

    def __init__(self, path: Path, fourcc: str, fps: float):
        self.path = path
        self.fourcc = fourcc
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.size: tuple[int, int] | None = None
        self.frames = 0

    def write(self, bgr: np.ndarray):
        h, w = bgr.shape[:2]
        if self.writer is None:
            self.size = (w, h)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            cc = cv2.VideoWriter_fourcc(*self.fourcc)
            self.writer = cv2.VideoWriter(str(self.path), cc, self.fps, self.size)
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open writer at {self.path}")
        elif (w, h) != self.size:
            bgr = cv2.resize(bgr, self.size, interpolation=cv2.INTER_AREA)

        self.writer.write(bgr)
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def stitch_front(img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
    """Simple panorama stitch with OpenCV; falls back to side-by-side if it fails."""
    # Light downscale improves robustness/speed on fisheyes
    scale = 0.75
    l_small = cv2.resize(
        img_left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    r_small = cv2.resize(
        img_right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    create = getattr(cv2, "Stitcher_create", None) or getattr(
        cv2.Stitcher, "create", None
    )
    stitcher = create(cv2.Stitcher_PANORAMA)
    status, pano_small = stitcher.stitch([l_small, r_small])
    if status == cv2.Stitcher_OK and pano_small is not None:
        inv = 1.0 / scale
        return cv2.resize(
            pano_small, None, fx=inv, fy=inv, interpolation=cv2.INTER_CUBIC
        )
    # Fallback: place side-by-side
    h = min(img_left.shape[0], img_right.shape[0])
    l_resized = cv2.resize(
        img_left, (img_left.shape[1], h), interpolation=cv2.INTER_AREA
    )
    r_resized = cv2.resize(
        img_right, (img_right.shape[1], h), interpolation=cv2.INTER_AREA
    )
    return np.hstack([l_resized, r_resized])


def to_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure BGR uint8 for writing."""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Spot images often arrive as RGB; decode helper usually returns proper channels
    # If you detect RGB, convert here (kept simple for brevity)
    return img.astype(np.uint8)


load_dotenv()


def build_rgb_request_compat(name: str):
    """Create an ImageRequest that works across Spot SDK versions."""
    req = image_pb2.ImageRequest()

    # image source field name changed across versions
    if hasattr(req, "image_source"):
        req.image_source = name
    else:
        # older SDKs
        req.image_source_name = name

    # Request JPEG (most body cams return JPEG anyway)
    if hasattr(req, "image_format"):
        req.image_format = image_pb2.Image.FORMAT_JPEG

    # Quality (if the field exists on your version)
    if hasattr(req, "quality_percent"):
        req.quality_percent = 75

    # Ask for RGB if available (mono sources will ignore this)
    if hasattr(req, "pixel_format"):
        req.pixel_format = image_pb2.Image.PIXEL_FORMAT_RGB_U8

    return req


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, help="Robot hostname or IP")
    ap.add_argument("--out-dir", type=Path, default=Path("recordings"))
    ap.add_argument("--fps", type=float, default=10.0, help="Target output FPS")
    ap.add_argument(
        "--seconds", type=int, default=30, help="Duration (0 = run until Ctrl+C)"
    )
    ap.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="Codec fourcc, e.g., mp4v, avc1, XVID",
    )
    ap.add_argument(
        "--stitch-front",
        action="store_true",
        help="Also write a stitched front panorama video",
    )
    ap.add_argument(
        "--display", action="store_true", help="Show a small live preview window"
    )
    args = ap.parse_args()

    import os

    ROBOT_USERNAME = os.getenv("SPOT_USERNAME")
    ROBOT_PASSWORD = os.getenv("SPOT_PASSWORD")

    # --- Connect to Spot
    sdk = bosdyn.client.create_standard_sdk("spot-bodycams-recorder")
    robot = sdk.create_robot(args.robot)
    robot.authenticate(ROBOT_USERNAME, ROBOT_PASSWORD)

    bosdyn.client.util.authenticate(robot)  # uses env or prompts
    robot.time_sync.wait_for_sync()
    image_client = robot.ensure_client(ImageClient.default_service_name)

    # Pre-build requests (ask for RGB if available)
    requests = [build_rgb_request_compat(s) for s in ALL_SOURCES]

    # Prepare writers
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sinks: dict[str, VideoSink] = {}
    for s in ALL_SOURCES:
        sinks[s] = VideoSink(args.out_dir / f"{s}_{ts}.mp4", args.fourcc, args.fps)
    if args.stitch_front:
        sinks["frontmiddle_virtual"] = VideoSink(
            args.out_dir / f"frontmiddle_virtual_{ts}.mp4", args.fourcc, args.fps
        )

    # Optional preview
    if args.display:
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("preview", 1280, 720)

    print(
        f"[INFO] Recording to {args.out_dir} at {args.fps} FPS "
        f"for {'∞' if args.seconds == 0 else str(args.seconds) + 's'}"
    )
    # end_time = float("inf") if args.seconds == 0 else (start + args.seconds)
    frame_period = 1.0 / max(1e-6, args.fps)

    try:
        while True:
            t0 = time.time()

            # Single batched request -> images are captured “together” as much as the service allows
            responses = image_client.get_image(requests)

            # Map images by source name
            frames: dict[str, np.ndarray] = {}
            for resp in responses:
                name = resp.source.name  # e.g., 'frontleft_fisheye_image'
                img = decode_image_response(resp)
                frames[name] = to_bgr(img)

            # Rotate and swap left/right before writing single view videos
            if FRONT_LEFT in frames:
                frames[FRONT_LEFT] = cv2.rotate(
                    frames[FRONT_LEFT], cv2.ROTATE_90_CLOCKWISE
                )
                sinks[FRONT_LEFT].write(frames[FRONT_LEFT])
            if FRONT_RIGHT in frames:
                frames[FRONT_RIGHT] = cv2.rotate(
                    frames[FRONT_RIGHT], cv2.ROTATE_90_CLOCKWISE
                )
                sinks[FRONT_RIGHT].write(frames[FRONT_RIGHT])

            # Write other cameras as usual
            for s in [LEFT, RIGHT, BACK]:
                if s in frames and frames[s] is not None:
                    sinks[s].write(frames[s])

            # Optional stitched front
            if args.stitch_front and FRONT_LEFT in frames and FRONT_RIGHT in frames:
                pano = stitch_front(frames[FRONT_RIGHT], frames[FRONT_LEFT])
                sinks["frontmiddle_virtual"].write(pano)

            # Optional preview: show stitched if available, else back
            if args.display:
                disp = pano if args.stitch_front else frames.get(BACK, None)
                if disp is not None:
                    cv2.imshow("preview", disp)
                    cv2.waitKey(1)

            # pace to FPS
            dt = time.time() - t0
            if dt < frame_period:
                time.sleep(frame_period - dt)

    except KeyboardInterrupt:
        pass
    finally:
        for name, sink in sinks.items():
            sink.close()
            print(f"[INFO] Saved {name}: {sink.frames} frames -> {sink.path}")
        if args.display:
            cv2.destroyAllWindows()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
