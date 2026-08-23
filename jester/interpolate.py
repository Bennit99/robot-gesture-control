"""Linear interpolation utilities for hand landmarks.

Resample a sequence of 21-point hand landmarks from one frame rate to
another using per-landmark linear interpolation.

Usage
-----
::

    import numpy as np
    from jester.interpolate import lerp_landmarks, resample_landmarks

    # Interpolate halfway between two frames (t=0.5)
    mid = lerp_landmarks(frame_a, frame_b, t=0.5)

    # Resample a full video from 12  30 FPS
    upsampled = resample_landmarks(landmarks_12fps, src_fps=12, dst_fps=30)

Landmark format
---------------
Each landmark frame is a NumPy array of shape ``(21, 3)`` - 21 hand
landmarks with ``(x, y, z)`` coordinates.  This matches MediaPipe's
hand landmark model output.
"""

from __future__ import annotations

import numpy as np


def lerp_landmarks(
    a: np.ndarray,
    b: np.ndarray,
    t: float,
) -> np.ndarray:
    """Linearly interpolate between two landmark frames.

    Parameters
    ----------
    a : np.ndarray, shape (21, 3)
        Landmark positions at the earlier frame.
    b : np.ndarray, shape (21, 3)
        Landmark positions at the later frame.
    t : float
        Interpolation factor in [0, 1].  ``t=0`` returns *a*, ``t=1``
        returns *b*, ``t=0.5`` returns the midpoint.

    Returns
    -------
    np.ndarray, shape (21, 3)
        Interpolated landmarks.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return (1.0 - t) * a + t * b


def resample_landmarks(
    landmarks: np.ndarray,
    src_fps: float,
    dst_fps: float,
) -> np.ndarray:
    """Resample a landmark sequence from *src_fps* to *dst_fps*.

    Uses per-landmark linear interpolation.  Works for both upsampling
    (e.g. 12  30 FPS) and downsampling (e.g. 30  12 FPS).

    Parameters
    ----------
    landmarks : np.ndarray, shape (N, 21, 3)
        Source landmark sequence - *N* frames, each with 21 landmarks
        of (x, y, z).
    src_fps : float
        Frame rate of the input sequence.
    dst_fps : float
        Desired output frame rate.

    Returns
    -------
    np.ndarray, shape (M, 21, 3)
        Resampled landmark sequence with *M* frames, where
        ``M = ceil(N * dst_fps / src_fps)``.

    Examples
    --------
    Upsample 37 frames from 12 FPS to 30 FPS::

        >>> seq = np.random.rand(37, 21, 3)
        >>> out = resample_landmarks(seq, src_fps=12, dst_fps=30)
        >>> out.shape[0]   # ~93 frames
        93
    """
    landmarks = np.asarray(landmarks, dtype=np.float64)
    n_src = landmarks.shape[0]

    if n_src == 0:
        return landmarks.copy()
    if n_src == 1:
        # Single frame  replicate to fill dst duration
        n_dst = max(1, int(np.ceil(1 * dst_fps / src_fps)))
        return np.tile(landmarks, (n_dst, 1, 1))

    # Total duration in seconds (from first to last frame)
    duration = (n_src - 1) / src_fps

    # Number of output frames (covering the same duration)
    n_dst = int(np.ceil(duration * dst_fps)) + 1

    # Time of each output frame in "source frame" units
    dst_times = np.linspace(0, n_src - 1, n_dst)

    out = np.empty((n_dst, *landmarks.shape[1:]), dtype=np.float64)

    for i, t in enumerate(dst_times):
        idx_a = int(np.floor(t))
        idx_b = min(idx_a + 1, n_src - 1)
        frac = t - idx_a
        out[i] = lerp_landmarks(landmarks[idx_a], landmarks[idx_b], frac)

    return out
