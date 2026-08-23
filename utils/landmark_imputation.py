from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ImputationStats:
    gap_segments_interpolated: int = 0
    gap_frames_interpolated: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "gap_segments_interpolated": int(self.gap_segments_interpolated),
            "gap_frames_interpolated": int(self.gap_frames_interpolated),
        }


def _frame_valid_mask(sequence: np.ndarray) -> np.ndarray:
    """Return a boolean mask: frame is valid if any feature is non-zero."""
    if sequence.ndim != 2:
        raise ValueError(f"Expected sequence shape (T, D), got {sequence.shape}")
    return np.any(np.abs(sequence) > 0, axis=1)


def interpolate_missing_landmark_gaps(
    sequence: np.ndarray,
    *,
    max_gap: int = 4,
    in_place: bool = False,
) -> tuple[np.ndarray, ImputationStats]:
    """Linearly interpolate internal missing gaps up to max_gap frames.

    A frame is considered missing when all features are zero.

    Only gaps with valid anchors on both sides are interpolated.
    Leading/trailing missing runs are left unchanged.
    """
    if max_gap < 1:
        out = sequence if in_place else sequence.copy()
        return out, ImputationStats()

    out = sequence if in_place else sequence.copy()
    valid = _frame_valid_mask(out)
    t = out.shape[0]
    stats = ImputationStats()

    idx = 0
    while idx < t:
        if valid[idx]:
            idx += 1
            continue

        gap_start = idx
        while idx < t and not valid[idx]:
            idx += 1
        gap_end = idx - 1
        gap_len = gap_end - gap_start + 1

        left_idx = gap_start - 1
        right_idx = idx if idx < t else -1

        has_left = left_idx >= 0 and valid[left_idx]
        has_right = right_idx >= 0 and right_idx < t and valid[right_idx]

        if not (has_left and has_right):
            continue
        if gap_len > max_gap:
            continue

        left_vec = out[left_idx]
        right_vec = out[right_idx]
        denom = gap_len + 1
        for k in range(1, gap_len + 1):
            alpha = k / denom
            out[left_idx + k] = (1.0 - alpha) * left_vec + alpha * right_vec

        valid[gap_start : gap_end + 1] = True
        stats.gap_segments_interpolated += 1
        stats.gap_frames_interpolated += gap_len

    return out, stats
