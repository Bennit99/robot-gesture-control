"""Shared hand-landmark constants used by both training and inference."""

N_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # x, y, z
FEATURE_DIM = N_LANDMARKS * COORDS_PER_LANDMARK  # 63
DEFAULT_SEQ_LEN = 40  # pad/truncate to this many frames
