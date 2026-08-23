"""Generic video composer for Jester frame folders.

The script accepts either a manifest file (.json, .csv, .txt) or a direct list
of video IDs. Each row can control label text, landmark overlay, and text
position, so the three old scripts can be replaced by one entry point.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.framework.formats import landmark_pb2

from utils.video_layout import (
    compose_with_info_panel,
    draw_text_block,
    resize_preserve_height,
)

_mp_drawing = mp.solutions.drawing_utils
_mp_hands = mp.solutions.hands
_mp_drawing_styles = mp.solutions.drawing_styles

TextPosition = Literal["bottom", "side", "corner", "none"]


@dataclass(frozen=True)
class VideoSpec:
    width: int
    height: int
    fps: float


@dataclass
class VideoItem:
    video_id: str
    label: str | None = None
    landmarks: bool | None = None
    text_position: TextPosition | None = None
    true_label: str | None = None
    pred_label: str | None = None
    detection_rate: float | None = None
    is_correct: bool | None = None


@dataclass(frozen=True)
class RenderItem:
    video_id: str
    label: str
    landmarks: bool
    text_position: TextPosition
    true_label: str
    pred_label: str
    detection_rate: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose Jester videos from a manifest file or a direct list of video IDs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One manifest file (.json/.csv/.txt) or a list of video IDs",
    )
    parser.add_argument(
        "--images-root",
        default="./data/jester/images",
        help="Root folder containing frame folders named by video ID",
    )
    parser.add_argument(
        "--landmarks-root",
        default="./data/jester/landmarks",
        help="Root folder containing landmark parquet files",
    )
    parser.add_argument(
        "--split-file",
        default="./data/jester/labels/jester-v1-validation.csv",
        help="Optional split file used to fill in labels when missing",
    )
    parser.add_argument(
        "--output",
        default="./output/video_composed.mp4",
        help="Output MP4 path",
    )
    parser.add_argument("--fps", type=float, default=12.0, help="Output FPS")
    parser.add_argument(
        "--target-height",
        type=int,
        default=240,
        help="Output frame height before labels are drawn",
    )
    parser.add_argument(
        "--output-upsample",
        type=int,
        default=1,
        help="Scale output resolution by this factor",
    )
    parser.add_argument(
        "--max-frames-per-id",
        type=int,
        default=0,
        help="Optional cap per video ID (0 means all frames)",
    )
    parser.add_argument(
        "--gap-frames",
        type=int,
        default=0,
        help="Black frames inserted between clips",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Overwrite existing output file",
    )
    parser.add_argument(
        "--landmarks",
        action="store_true",
        help="Overlay hand landmarks on all videos unless overridden in the manifest",
    )
    parser.add_argument(
        "--text-position",
        choices=["bottom", "side", "corner", "none"],
        default="bottom",
        help="Default text position when the manifest does not specify one",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        help="Labels for direct video ID input",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=0,
        help="Info panel width in pixels when using side text (0 = auto)",
    )
    parser.add_argument(
        "--wrong-only",
        action="store_true",
        help="Keep only rows marked as wrong predictions when is_correct exists",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle input order before writing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --shuffle",
    )
    return parser.parse_args()


def _parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return None


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_text_position(value: Any) -> TextPosition | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"bottom", "side", "corner", "none"}:
        return text
    return None


def _clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _manifest_item_from_mapping(mapping: dict[str, Any]) -> VideoItem | None:
    video_id = str(mapping.get("video_id") or mapping.get("id") or "").strip()
    if not video_id:
        return None

    detection_rate = _parse_float(
        mapping.get("detection_rate")
        if mapping.get("detection_rate") is not None
        else mapping.get("detection_rate_percent")
    )

    return VideoItem(
        video_id=video_id,
        label=_clean_text(mapping.get("label")),
        landmarks=_parse_bool(mapping.get("landmarks")),
        text_position=_parse_text_position(mapping.get("text_position")),
        true_label=_clean_text(mapping.get("true_label")),
        pred_label=_clean_text(mapping.get("pred_label")),
        detection_rate=detection_rate,
        is_correct=_parse_bool(mapping.get("is_correct")),
    )


def _load_manifest_json(path: Path) -> list[VideoItem]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        raw_entries = data
    elif isinstance(data, dict):
        for key in ("items", "videos", "rows", "data"):
            if isinstance(data.get(key), list):
                raw_entries = data[key]
                break
        else:
            raw_entries = list(data.values())
    else:
        raise ValueError(f"Unsupported JSON manifest format: {path}")

    items: list[VideoItem] = []
    for entry in raw_entries:
        if isinstance(entry, str):
            video_id = entry.strip()
            if video_id:
                items.append(VideoItem(video_id=video_id))
        elif isinstance(entry, dict):
            item = _manifest_item_from_mapping(entry)
            if item is not None:
                items.append(item)
    return items


def _load_manifest_csv(path: Path) -> list[VideoItem]:
    df = pd.read_csv(path, sep=None, engine="python")
    if df.empty:
        return []

    items: list[VideoItem] = []
    for _, row in df.iterrows():
        mapping = {column: row[column] for column in df.columns}
        item = _manifest_item_from_mapping(mapping)
        if item is not None:
            items.append(item)
    return items


def _load_manifest_txt(path: Path) -> list[VideoItem]:
    items: list[VideoItem] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            video_id = line.strip()
            if video_id:
                items.append(VideoItem(video_id=video_id))
    return items


def _load_manifest(path: Path) -> list[VideoItem]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_manifest_json(path)
    if suffix == ".csv":
        return _load_manifest_csv(path)
    if suffix == ".txt":
        return _load_manifest_txt(path)

    try:
        return _load_manifest_json(path)
    except Exception:
        try:
            return _load_manifest_csv(path)
        except Exception:
            return _load_manifest_txt(path)


def _load_split_labels(split_file: Path) -> dict[str, str]:
    if not split_file.exists():
        return {}

    labels: dict[str, str] = {}
    with split_file.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        for row in reader:
            if len(row) >= 2:
                labels[row[0].strip()] = row[1].strip()
    return labels


def _filter_wrong_predictions(items: list[VideoItem]) -> list[VideoItem]:
    filtered: list[VideoItem] = []
    for item in items:
        if item.is_correct is False:
            filtered.append(item)
            continue
        if item.is_correct is None and item.true_label and item.pred_label:
            if item.true_label != item.pred_label:
                filtered.append(item)
    return filtered


def _resolve_raw_items(args: argparse.Namespace) -> list[VideoItem]:
    inputs = [str(value).strip() for value in args.inputs if str(value).strip()]
    if not inputs:
        raise ValueError("No input video IDs or manifest file supplied.")

    if len(inputs) == 1:
        maybe_path = Path(inputs[0])
        if maybe_path.exists():
            items = _load_manifest(maybe_path)
            if args.wrong_only:
                items = _filter_wrong_predictions(items)
            if args.shuffle:
                random.Random(args.seed).shuffle(items)
            return items

        if maybe_path.suffix.lower() in {".json", ".csv", ".txt"}:
            raise FileNotFoundError(f"Input file not found: {maybe_path}")

    labels = args.labels or []
    items = []
    for index, video_id in enumerate(inputs):
        items.append(
            VideoItem(
                video_id=video_id,
                label=labels[index] if index < len(labels) else None,
                landmarks=args.landmarks,
                text_position=args.text_position,
            )
        )

    if args.shuffle:
        random.Random(args.seed).shuffle(items)
    return items


def _resolve_items(
    raw_items: list[VideoItem], args: argparse.Namespace, split_labels: dict[str, str]
) -> list[RenderItem]:
    items: list[RenderItem] = []
    for raw_item in raw_items:
        items.append(
            RenderItem(
                video_id=raw_item.video_id,
                label=raw_item.label or split_labels.get(raw_item.video_id, ""),
                landmarks=args.landmarks
                if raw_item.landmarks is None
                else raw_item.landmarks,
                text_position=raw_item.text_position or args.text_position,
                true_label=raw_item.true_label or "",
                pred_label=raw_item.pred_label or "",
                detection_rate=raw_item.detection_rate,
            )
        )
    return items


def _iter_frame_paths(folder: Path, max_frames: int) -> list[Path]:
    frames: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        frames.extend(folder.glob(ext))
    frames = sorted(frames)
    if max_frames > 0:
        return frames[:max_frames]
    return frames


def _infer_output_width(
    images_root: Path, items: list[RenderItem], target_height: int
) -> int:
    max_width = 0
    for item in items:
        folder = images_root / item.video_id
        if not folder.exists():
            continue

        frame_paths = _iter_frame_paths(folder, 1)
        if not frame_paths:
            continue

        frame = cv2.imread(str(frame_paths[0]))
        if frame is None:
            continue

        height, width = frame.shape[:2]
        if height <= 0:
            continue

        scaled_width = int(round((width / height) * target_height))
        max_width = max(max_width, scaled_width)

    if max_width <= 0:
        raise FileNotFoundError("Could not find a readable frame for any input item.")
    return max_width


def _load_landmarks_for_video(
    landmarks_root: Path, video_id: str
) -> dict[int, list[np.ndarray]]:
    pq_path = landmarks_root / video_id / "landmarks.parquet"
    if not pq_path.exists():
        return {}

    try:
        df = pd.read_parquet(pq_path)
    except Exception:
        return {}

    if df.empty or "frame" not in df.columns:
        return {}

    coord_cols = [
        column
        for column in df.columns
        if isinstance(column, str)
        and len(column) > 1
        and column[0] in {"x", "y", "z"}
        and column[1:].isdigit()
    ]
    if len(coord_cols) != 63:
        return {}

    by_frame: dict[int, list[np.ndarray]] = {}
    for _, row in df.iterrows():
        try:
            frame_idx = int(row["frame"])
        except Exception:
            continue

        vec = row[coord_cols].to_numpy(dtype=np.float32)
        if vec.size != 63:
            continue

        hand = vec.reshape(21, 3)
        by_frame.setdefault(frame_idx, []).append(hand)
    return by_frame


def _draw_landmarks(frame_bgr: np.ndarray, hands: list[np.ndarray]) -> np.ndarray:
    out = frame_bgr.copy()
    for hand in hands:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=float(point[0]), y=float(point[1]), z=float(point[2])
                )
                for point in hand
            ]
        )
        _mp_drawing.draw_landmarks(
            out,
            proto,
            _mp_hands.HAND_CONNECTIONS,
            _mp_drawing_styles.get_default_hand_landmarks_style(),
            _mp_drawing_styles.get_default_hand_connections_style(),
        )
    return out


def _bottom_label(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    canvas = frame_bgr.copy()

    bar_h = max(28, int(round(h * 0.16)))
    y0 = h - bar_h

    overlay = canvas.copy()
    overlay[y0:h, :, :] = (8, 8, 8)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0.0, canvas)

    bar = np.full((bar_h * 2, w * 2, 3), 8, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(1.0, h / 320.0)) * 2
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    pad_x = 16
    pad_y = 10
    box_w = min(bar.shape[1] - 20, tw + pad_x * 2)
    box_h = th + baseline + pad_y * 2
    box_x = max(10, (bar.shape[1] - box_w) // 2)
    box_y = max(8, (bar.shape[0] - box_h) // 2)

    cv2.rectangle(bar, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 20, 20), -1)

    tx = max(box_x + pad_x, (bar.shape[1] - tw) // 2)
    ty = box_y + pad_y + th
    cv2.putText(
        bar, text, (tx + 1, ty + 1), font, scale, (0, 0, 0), thickness, cv2.LINE_AA
    )
    cv2.putText(
        bar, text, (tx, ty), font, scale, (245, 245, 245), thickness, cv2.LINE_AA
    )

    bar_lr = cv2.resize(bar, (w, bar_h), interpolation=cv2.INTER_AREA)
    canvas[y0:h, :, :] = bar_lr
    return canvas


def _corner_label(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = frame_bgr.copy()
    draw_text_block(canvas, lines, top=10, left=10)
    return canvas


def _overlay_lines(item: RenderItem) -> list[str]:
    lines: list[str] = []
    if item.label:
        lines.append(item.label)
    if item.true_label or item.pred_label:
        parts: list[str] = []
        if item.true_label:
            parts.append(f"True: {item.true_label}")
        if item.pred_label:
            parts.append(f"Pred: {item.pred_label}")
        lines.append(" | ".join(parts))
    if item.detection_rate is not None:
        lines.append(f"Detection rate: {item.detection_rate:.1f}%")
    return lines


def _frame_index_from_path(frame_path: Path, fallback_index: int) -> int:
    stem = frame_path.stem
    if stem.isdigit():
        return int(stem)
    return fallback_index


def _open_writer(path: Path, spec: VideoSpec) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), spec.fps, (spec.width, spec.height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {path}")
    return writer


def build_video(args: argparse.Namespace) -> None:
    raw_items = _resolve_raw_items(args)
    split_labels = _load_split_labels(Path(args.split_file))
    items = _resolve_items(raw_items, args, split_labels)

    if not items:
        raise ValueError("No video items loaded from input.")

    images_root = Path(args.images_root)
    landmarks_root = Path(args.landmarks_root)
    output_path = Path(args.output)

    if output_path.exists() and not args.override:
        raise FileExistsError(f"Output already exists: {output_path} (use --override)")

    upsample = max(1, int(args.output_upsample))
    base_width = _infer_output_width(images_root, items, args.target_height)
    frame_width = base_width * upsample
    frame_height = args.target_height * upsample
    needs_panel = any(item.text_position == "side" for item in items)

    panel_width = 0
    if needs_panel:
        if args.panel_width > 0:
            panel_width = args.panel_width
        else:
            panel_width = int(round(frame_width * 0.36))
            panel_width = max(280, min(panel_width, 620))

    spec = VideoSpec(
        width=frame_width + panel_width,
        height=frame_height,
        fps=args.fps,
    )

    print(f"[info] Output spec: {spec.width}x{spec.height} @ {spec.fps} fps")
    writer = _open_writer(output_path, spec)

    written_frames = 0
    landmarks_cache: dict[str, dict[int, list[np.ndarray]]] = {}

    try:
        for index, item in enumerate(items, start=1):
            folder = images_root / item.video_id
            if not folder.exists():
                print(f"[warn] Missing folder: {folder}")
                continue

            frame_paths = _iter_frame_paths(folder, args.max_frames_per_id)
            if not frame_paths:
                print(f"[warn] No image frames in: {folder}")
                continue

            print(f"[info] Writing {item.video_id} ({index}/{len(items)})")
            display_lines = _overlay_lines(item)

            use_landmarks = item.landmarks
            landmarks_by_frame: dict[int, list[np.ndarray]] = {}
            if use_landmarks:
                if item.video_id not in landmarks_cache:
                    landmarks_cache[item.video_id] = _load_landmarks_for_video(
                        landmarks_root, item.video_id
                    )
                landmarks_by_frame = landmarks_cache[item.video_id]

            for fallback_index, frame_path in enumerate(frame_paths):
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue

                frame = resize_preserve_height(frame, frame_width, frame_height)

                if use_landmarks and landmarks_by_frame:
                    frame_idx = _frame_index_from_path(frame_path, fallback_index)
                    hands = landmarks_by_frame.get(frame_idx)
                    if hands is None and frame_idx > 0:
                        hands = landmarks_by_frame.get(frame_idx - 1, [])
                    if hands:
                        frame = _draw_landmarks(frame, hands)

                if item.text_position == "bottom":
                    text = " | ".join(display_lines)
                    if text:
                        frame = _bottom_label(frame, text)
                elif item.text_position == "corner":
                    if display_lines:
                        frame = _corner_label(frame, display_lines)

                if needs_panel:
                    panel_lines = display_lines if item.text_position == "side" else []
                    frame = compose_with_info_panel(
                        frame,
                        panel_lines,
                        panel_width=panel_width,
                        panel_supersample=2,
                    )

                writer.write(frame)
                written_frames += 1

            if args.gap_frames > 0 and index < len(items):
                gap = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
                for _ in range(args.gap_frames):
                    writer.write(gap)
                    written_frames += 1
    finally:
        writer.release()

    if written_frames == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("No frames were written. Check the video IDs and paths.")

    print(f"[done] Saved {written_frames} frames to {output_path}")


def main() -> None:
    args = parse_args()
    build_video(args)


if __name__ == "__main__":
    main()
