"""Generate a video-composer manifest from a per-video evaluation CSV.

The output can be JSON or CSV and is intended to feed directly into
scripts/video_composer.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a manifest from a per-video evaluation CSV."
    )
    parser.add_argument(
        "input",
        help="Path to per_video_predictions.csv or a similar CSV with video_id columns",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output manifest path (.json, .csv, or .txt)",
    )
    parser.add_argument(
        "--wrong-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only rows marked as wrong predictions when is_correct exists",
    )
    parser.add_argument(
        "--landmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mark all generated entries as requiring landmark overlay",
    )
    parser.add_argument(
        "--text-position",
        choices=["bottom", "side", "corner", "none"],
        default="bottom",
        help="Text position to store in the manifest",
    )
    parser.add_argument(
        "--label-mode",
        choices=["true", "pred", "both", "empty"],
        default="both",
        help="How to populate the label field in the manifest",
    )
    return parser.parse_args()


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _label_from_row(row: pd.Series, mode: str) -> str:
    true_label = _clean_text(row.get("true_label"))
    pred_label = _clean_text(row.get("pred_label"))
    if mode == "true":
        return true_label
    if mode == "pred":
        return pred_label
    if mode == "empty":
        return ""
    if true_label and pred_label:
        return f"True: {true_label} | Pred: {pred_label}"
    return true_label or pred_label


def _load_items(
    input_path: Path,
    wrong_only: bool,
    landmarks: bool,
    text_position: str,
    label_mode: str,
) -> list[dict[str, object]]:
    df = pd.read_csv(input_path)
    if df.empty:
        return []

    if "video_id" not in df.columns:
        raise ValueError("Input CSV must contain a video_id column.")

    if wrong_only and "is_correct" in df.columns:
        df = df[df["is_correct"] == False].copy()

    items: list[dict[str, object]] = []
    for _, row in df.iterrows():
        video_id = _clean_text(row["video_id"])
        if not video_id:
            continue

        item: dict[str, object] = {
            "video_id": video_id,
            "label": _label_from_row(row, label_mode),
            "landmarks": landmarks,
            "text_position": text_position,
        }

        if "true_label" in df.columns:
            true_label = _clean_text(row.get("true_label"))
            if true_label:
                item["true_label"] = true_label
        if "pred_label" in df.columns:
            pred_label = _clean_text(row.get("pred_label"))
            if pred_label:
                item["pred_label"] = pred_label
        if "detection_rate_percent" in df.columns:
            value = row.get("detection_rate_percent")
            if not pd.isna(value):
                item["detection_rate"] = float(value)
        if "is_correct" in df.columns and not pd.isna(row.get("is_correct")):
            item["is_correct"] = bool(row.get("is_correct"))

        items.append(item)

    return items


def _write_output(output_path: Path, items: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".json":
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(items, handle, indent=2, ensure_ascii=True)
            handle.write("\n")
        return

    if suffix == ".csv":
        fieldnames: list[str] = ["video_id", "label", "landmarks", "text_position"]
        for key in ("true_label", "pred_label", "detection_rate", "is_correct"):
            if any(key in item for item in items):
                fieldnames.append(key)

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            for item in items:
                writer.writerow({field: item.get(field, "") for field in fieldnames})
        return

    if suffix == ".txt":
        with output_path.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(f"{item['video_id']}\n")
        return

    raise ValueError("Output path must end with .json, .csv, or .txt")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    items = _load_items(
        input_path=input_path,
        wrong_only=args.wrong_only,
        landmarks=args.landmarks,
        text_position=args.text_position,
        label_mode=args.label_mode,
    )

    _write_output(output_path, items)
    print(f"Wrote {len(items)} manifest items to {output_path}")


if __name__ == "__main__":
    main()
