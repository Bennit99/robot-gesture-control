from __future__ import annotations

import cv2
import numpy as np


def resize_preserve_height(
    frame_bgr: np.ndarray, target_w: int, target_h: int
) -> np.ndarray:
    """Resize to target height while preserving aspect ratio.

    If the resulting width differs from target_w, center-crop or pad to match.
    """
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid frame with non-positive dimensions.")

    scaled_w = max(1, int(round((w / h) * target_h)))
    resized = cv2.resize(frame_bgr, (scaled_w, target_h), interpolation=cv2.INTER_AREA)

    if scaled_w == target_w:
        return resized

    if scaled_w > target_w:
        start = (scaled_w - target_w) // 2
        return resized[:, start : start + target_w]

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - scaled_w) // 2
    canvas[:, x0 : x0 + scaled_w] = resized
    return canvas


def upscale_min_width(frame_bgr: np.ndarray, min_width: int) -> np.ndarray:
    """Upscale frame to at least min_width while preserving aspect ratio."""
    if min_width <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if w >= min_width:
        return frame_bgr

    scale = float(min_width) / float(w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def draw_text_block(
    frame_bgr: np.ndarray,
    lines: list[str],
    *,
    top: int = 10,
    left: int = 10,
    base_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw a compact text block with dark background directly on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, frame_bgr.shape[1] / 1400)
    thickness = 2
    line_h = int(28 * scale) + 8

    max_w = 0
    for text in lines:
        (tw, _th), _ = cv2.getTextSize(text, font, scale, thickness)
        max_w = max(max_w, tw)

    block_h = line_h * len(lines) + 10
    cv2.rectangle(
        frame_bgr,
        (left - 6, top - 6),
        (left + max_w + 12, top + block_h),
        bg_color,
        -1,
    )

    y = top + line_h - 4
    for text in lines:
        cv2.putText(
            frame_bgr,
            text,
            (left, y),
            font,
            scale,
            base_color,
            thickness,
            cv2.LINE_AA,
        )
        y += line_h


def compose_with_info_panel(
    video_bgr: np.ndarray,
    lines: list[str],
    *,
    panel_width: int = 0,
    panel_fraction: float = 0.36,
    panel_min_width: int = 280,
    panel_max_width: int = 620,
    panel_supersample: int = 2,
) -> np.ndarray:
    """Compose video with an adjacent info panel.

    If panel_width <= 0, width is chosen from video width by panel_fraction,
    clamped into [panel_min_width, panel_max_width]. Text is rendered on a
    supersampled canvas and downsampled for better glyph sharpness.
    """
    h, w = video_bgr.shape[:2]
    if panel_width > 0:
        panel_w = panel_width
    else:
        panel_w = int(round(w * panel_fraction))
    panel_w = max(panel_min_width, min(panel_w, panel_max_width))

    ss = max(1, int(panel_supersample))
    panel_hr = np.full((h * ss, panel_w * ss, 3), 28, dtype=np.uint8)

    scaled_lines = lines
    draw_text_block(panel_hr, scaled_lines, top=10 * ss, left=10 * ss)
    panel = cv2.resize(panel_hr, (panel_w, h), interpolation=cv2.INTER_AREA)

    return np.hstack([video_bgr, panel])
