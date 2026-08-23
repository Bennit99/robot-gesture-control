"""Preview Jester dataset gesture videos in a CV2 window.

Usage
-----
    # Show random examples for a specific class:
    python -m jester.preview "Swiping Left"

    # Cycle through ALL classes (random example each):
    python -m jester.preview

    # Show N examples per class (default 1):
    python -m jester.preview --n 3

    # Show a specific video by ID:
    python -m jester.preview --id 34870

Controls
--------
    SPACE /    Next video
               Previous video (replay)
    R           Replay current video
    P           Pause / unpause
    Q / ESC     Quit
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2

# -- Paths -----------------------------------------------------------------
LABELS_CSV = Path("data/jester/lables/jester-v1-labels.csv")
TRAIN_CSV = Path("data/jester/lables/jester-v1-train.csv")
VAL_CSV = Path("data/jester/lables/jester-v1-validation.csv")
IMAGES_DIR = Path("data/jester/images")

FPS = 12
WINDOW = "Jester Preview"
MIN_WIDTH = 640  # upscale small frames so the window is usable


# -- Helpers ---------------------------------------------------------------
def load_classes(path: Path = LABELS_CSV) -> list[str]:
    """Return ordered list of gesture class names."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_video_labels(*csv_files: Path) -> dict[str, str]:
    """Return ``{video_id: class_name}`` from one or more label CSVs."""
    mapping: dict[str, str] = {}
    for csv_file in csv_files:
        if not csv_file.exists():
            continue
        with open(csv_file, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                if len(row) >= 2:
                    mapping[row[0].strip()] = row[1].strip()
    return mapping


def videos_for_class(class_name: str, mapping: dict[str, str]) -> list[str]:
    """Return list of video IDs belonging to *class_name*."""
    return [vid for vid, cls in mapping.items() if cls == class_name]


def load_frames(video_id: str) -> list:
    """Load all JPG frames for a video, sorted by filename."""
    folder = IMAGES_DIR / str(video_id)
    if not folder.is_dir():
        print(f"  Video folder not found: {folder}")
        return []
    paths = sorted(folder.glob("*.jpg"))
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
    return frames


def draw_overlay(frame, text: str, position="top"):
    """Draw a text overlay on the frame (mutates in-place)."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, w / 600)
    thickness = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    if position == "top":
        y = th + 10
    else:
        y = h - 10

    # background
    cv2.rectangle(frame, (5, y - th - 5), (10 + tw + 5, y + 5), (0, 0, 0), -1)
    cv2.putText(
        frame, text, (10, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def _upscale(frame):
    """Upscale a frame so its width is at least MIN_WIDTH (keeps aspect ratio)."""
    h, w = frame.shape[:2]
    if w >= MIN_WIDTH:
        return frame
    scale = MIN_WIDTH / w
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def play_video(
    video_id: str,
    class_name: str,
    idx: int,
    total: int,
) -> str:
    """Play one video. Returns action: 'next', 'prev', or 'quit'."""
    frames = load_frames(video_id)
    if not frames:
        return "next"

    delay = int(1000 / FPS)
    paused = False
    fi = 0  # frame index

    while True:
        frame = _upscale(frames[fi].copy())
        # Overlays
        header = f"[{idx + 1}/{total}]  {class_name}  (video {video_id})"
        footer = f"Frame {fi + 1}/{len(frames)}  |  SPACE/Right=next  Left=prev  R=replay  P=pause  Q=quit"
        draw_overlay(frame, header, "top")
        draw_overlay(frame, footer, "bottom")

        cv2.imshow(WINDOW, frame)

        key = cv2.waitKeyEx(0 if paused else delay)

        if key == ord("q") or key == 27:  # Q or ESC
            return "quit"
        elif key in (ord(" "), 0x270000, ord("d")):  # SPACE, Right arrow, D
            return "next"
        elif key in (0x250000, ord("a")):  # Left arrow, A
            return "prev"
        elif key == ord("r"):  # Replay
            fi = 0
            paused = False
            continue
        elif key == ord("p"):  # Pause toggle
            paused = not paused
            continue

        if not paused:
            fi += 1
            if fi >= len(frames):
                # Finished - auto-advance to next video
                return "next"


# -- Main ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Preview Jester dataset gesture videos."
    )
    parser.add_argument(
        "class_name",
        nargs="?",
        default=None,
        help="Gesture class to preview (e.g. 'Swiping Left'). "
        "If omitted, cycles through all classes.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of random examples per class (default: 1).",
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Show a specific video by its numeric ID.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the order of classes when cycling through all.",
    )
    args = parser.parse_args()

    # Load labels
    print("Loading labels...")
    classes = load_classes()
    mapping = load_video_labels(TRAIN_CSV, VAL_CSV)
    print(f"  {len(classes)} classes, {len(mapping)} labelled videos")

    # Build playlist
    playlist: list[tuple[str, str]] = []  # (video_id, class_name)

    if args.id:
        # Single video by ID
        cls = mapping.get(args.id, "Unknown")
        playlist.append((args.id, cls))
    elif args.class_name:
        # Specific class
        matches = [c for c in classes if c.lower() == args.class_name.lower()]
        if not matches:
            # Fuzzy: substring match
            matches = [c for c in classes if args.class_name.lower() in c.lower()]
        if not matches:
            print(f"No class matching '{args.class_name}'. Available classes:")
            for c in classes:
                print(f"  - {c}")
            sys.exit(1)
        for cls in matches:
            vids = videos_for_class(cls, mapping)
            if not vids:
                print(f"  No videos found for '{cls}' in downloaded data.")
                continue
            chosen = random.sample(vids, min(args.n, len(vids)))
            for vid in chosen:
                playlist.append((vid, cls))
    else:
        # All classes
        order = list(classes)
        if args.shuffle:
            random.shuffle(order)
        for cls in order:
            vids = videos_for_class(cls, mapping)
            if not vids:
                continue
            chosen = random.sample(vids, min(args.n, len(vids)))
            for vid in chosen:
                playlist.append((vid, cls))

    if not playlist:
        print("No videos to show.")
        sys.exit(0)

    print(f"Playlist: {len(playlist)} video(s)\n")

    # Playback loop with prev/next navigation
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, max(MIN_WIDTH, 640), max(int(MIN_WIDTH * 0.75), 480))
    i = 0
    while 0 <= i < len(playlist):
        vid, cls = playlist[i]
        action = play_video(vid, cls, i, len(playlist))
        if action == "quit":
            break
        elif action == "next":
            i += 1
        elif action == "prev":
            i = max(0, i - 1)

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
