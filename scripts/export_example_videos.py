"""Export example gesture videos from the Jester dataset.

For each dynamic gesture class defined in cfg/gesture_config.py, this script:
1. Parses the Jester training labels CSV to find all video folder IDs for each class.
2. Filters to only folder IDs that exist locally in data/jester/images/.
3. Randomly selects 3 video folders per class (seed=42 for reproducibility).
4. Reads the JPEG frame sequences, concatenates them with gap frames, and writes
   a single MP4 video per class into web_ui/streaming_web_server/example_videos/.

Usage:
    uv run python scripts/export_example_videos.py
"""

from __future__ import annotations

import csv
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "data" / "jester" / "images"
TRAIN_CSV = ROOT / "data" / "jester" / "labels" / "jester-v1-train.csv"
OUTPUT_DIR = ROOT / "web_ui" / "streaming_web_server" / "example_videos"

# ── Config ────────────────────────────────────────────────────────────────
EXAMPLES_PER_CLASS = 3
SEED = 42
FPS = 12.0
TARGET_HEIGHT = 240
GAP_FRAMES = 6  # black frames between clips


def sanitize_class_name(name: str) -> str:
    """Convert a class name to a safe filename (lowercase, underscores)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_train_labels(csv_path: Path) -> dict[str, list[str]]:
    """Parse training CSV and return {class_name: [folder_id, ...]}."""
    class_to_folders: dict[str, list[str]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        for row in reader:
            if len(row) >= 2:
                folder_id, class_name = row[0].strip(), row[1].strip()
                class_to_folders.setdefault(class_name, []).append(folder_id)
    return class_to_folders


def load_frames(folder: Path) -> list[np.ndarray]:
    """Load all JPG frames from a video folder, sorted by name."""
    paths = sorted(folder.glob("*.jpg"))
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
    return frames


def resize_frame(
    frame: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """Resize a frame to exact dimensions."""
    return cv2.resize(
        frame, (target_width, target_height), interpolation=cv2.INTER_AREA
    )


def compose_class_video(
    class_name: str,
    folder_ids: list[str],
    output_path: Path,
) -> bool:
    """Compose a single MP4 from multiple video folders for one gesture class."""
    # Collect all clips
    all_clips: list[list[np.ndarray]] = []
    for fid in folder_ids:
        folder = IMAGES_DIR / fid
        if not folder.is_dir():
            print(f"  [warn] Folder {folder} not found, skipping")
            continue
        frames = load_frames(folder)
        if frames:
            all_clips.append(frames)

    if not all_clips:
        print(f"  [error] No frames found for class '{class_name}'")
        return False

    # Determine uniform output size from first frame of first clip
    sample = all_clips[0][0]
    h, w = sample.shape[:2]
    scale = TARGET_HEIGHT / h
    out_width = int(round(w * scale))
    out_height = TARGET_HEIGHT

    # Open video writer - try H.264 first for browser compatibility
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try codecs in order of browser compatibility
    codec_attempts = [
        ("avc1", ".mp4"),  # H.264 — best browser support
        ("VP80", ".webm"),  # VP8/WebM — good fallback
        ("mp4v", ".mp4"),  # MPEG-4 Part 2 — last resort
    ]

    writer = None
    for fourcc_str, ext in codec_attempts:
        # Update output path extension if needed
        test_path = output_path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        test_writer = cv2.VideoWriter(
            str(test_path), fourcc, FPS, (out_width, out_height)
        )
        if test_writer.isOpened():
            writer = test_writer
            output_path = test_path
            print(f"  [codec] Using {fourcc_str} ({ext})")
            break
        test_writer.release()

    if writer is None or not writer.isOpened():
        print(f"  [error] Failed to open writer for {output_path}")
        return False

    total_frames = 0
    try:
        for clip_idx, clip in enumerate(all_clips):
            for frame in clip:
                resized = resize_frame(frame, out_height, out_width)
                writer.write(resized)
                total_frames += 1

            # Add gap frames between clips (not after the last one)
            if clip_idx < len(all_clips) - 1:
                gap = np.zeros((out_height, out_width, 3), dtype=np.uint8)
                for _ in range(GAP_FRAMES):
                    writer.write(gap)
                    total_frames += 1
    finally:
        writer.release()

    print(
        f"  [done] {class_name}: {len(all_clips)} clips, {total_frames} frames -> {output_path.name}"
    )
    return True


def main() -> None:
    # Add project root to path so we can import cfg
    sys.path.insert(0, str(ROOT))
    from cfg.gesture_config import gesture_mappings

    dynamic_classes = list(gesture_mappings["dynamic"].keys())
    print(f"Dynamic gesture classes: {len(dynamic_classes)}")

    # Load training labels
    if not TRAIN_CSV.exists():
        print(f"[error] Training labels not found: {TRAIN_CSV}")
        sys.exit(1)

    class_to_folders = load_train_labels(TRAIN_CSV)
    print(f"Loaded {sum(len(v) for v in class_to_folders.values())} video labels")

    # Check which folders exist locally
    if not IMAGES_DIR.exists():
        print(f"[error] Images directory not found: {IMAGES_DIR}")
        sys.exit(1)

    local_folders = set(p.name for p in IMAGES_DIR.iterdir() if p.is_dir())
    print(f"Local image folders: {len(local_folders)}")

    # Select examples and compose videos
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    success_count = 0

    for class_name in dynamic_classes:
        all_folder_ids = class_to_folders.get(class_name, [])
        # Filter to locally available folders
        available = [fid for fid in all_folder_ids if fid in local_folders]

        if len(available) < EXAMPLES_PER_CLASS:
            print(
                f"[warn] Only {len(available)} local videos for '{class_name}', need {EXAMPLES_PER_CLASS}"
            )
            if not available:
                continue

        selected = rng.sample(available, min(EXAMPLES_PER_CLASS, len(available)))
        sanitized = sanitize_class_name(class_name)
        output_path = OUTPUT_DIR / f"{sanitized}.mp4"

        print(f"\nComposing '{class_name}' from videos: {selected}")
        if compose_class_video(class_name, selected, output_path):
            success_count += 1

    print(f"\n{'=' * 60}")
    print(
        f"Exported {success_count}/{len(dynamic_classes)} class videos to {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
