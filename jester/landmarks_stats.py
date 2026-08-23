"""Load Jester landmark Parquet files and compute aggregate statistics + plots.

Usage
-----
CLI::

    python -m jester.landmarks_stats                           # default paths
    python -m jester.landmarks_stats --landmarks ./data/jester/landmarks
    python -m jester.landmarks_stats --images ./data/jester/images  # count source frames

Python::

    from jester.landmarks_stats import load_all_landmarks, compute_stats, plot_all
    df, meta = load_all_landmarks("./data/jester/landmarks", "./data/jester/images")
    stats = compute_stats(df, meta)
    plot_all(stats, "./data/jester/landmarks")

Data format (as saved by extract_landmarks.py)
----------------------------------------------
Each video sub-folder contains two Parquet files with identical schemas::

    landmarks.parquet       - normalised image-space (x, y, z) per landmark
    world_landmarks.parquet - real-world 3D in metres

Columns: ``frame, hand_index, handedness, x0, y0, z0, , x20, y20, z20``

* ``frame``      - 0-based frame index within the video
* ``hand_index`` - 0 or 1 (which detected hand in that frame)
* ``handedness`` - 0 = Left, 1 = Right

Frames where **no hand** was detected simply have **no rows** for that
frame index - this is how we identify "missing" frames.
"""

from __future__ import annotations

import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_LANDMARKS_DIR = "./data/jester/landmarks"
DEFAULT_IMAGES_DIR = "./data/jester/images"
DEFAULT_CACHE_DIR = "./data/jester/cache"
DEFAULT_LABELS_DIR = "./data/jester/lables"
DEFAULT_STATS_DIR = "./data/jester/stats"


# ==============================================================================
# Cache
# ==============================================================================
def _cache_path(cache_dir: str, landmarks_dir: str) -> str:
    folder_name = os.path.basename(os.path.normpath(landmarks_dir))
    return os.path.join(cache_dir, f"{folder_name}.pkl")


def _save_cache(
    cache_dir: str, landmarks_dir: str, df: pd.DataFrame, meta: pd.DataFrame
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, landmarks_dir)
    with open(path, "wb") as f:
        pickle.dump({"df": df, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved cache  {path}  ({mb:.1f} MB)")


def _load_cache(
    cache_dir: str,
    landmarks_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    path = _cache_path(cache_dir, landmarks_dir)
    if not os.path.exists(path):
        return None
    print(f"  Loading cache  {path}")
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        payload = pickle.load(f)
    dt = time.perf_counter() - t0
    print(f"  Cache loaded in {dt:.1f}s")
    return payload["df"], payload["meta"]


# ==============================================================================
# Label loading / class filtering
# ==============================================================================
def _load_video_labels(labels_dir: str) -> dict[str, str]:
    """Build a mapping ``{video_id: gesture_name}`` from split CSVs.

    Reads ``jester-v1-train.csv`` and ``jester-v1-validation.csv``
    (semicolon-delimited ``video_id;label``).
    """
    vid_to_label: dict[str, str] = {}
    for fname in ("jester-v1-train.csv", "jester-v1-validation.csv"):
        path = os.path.join(labels_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ";" not in line:
                    continue
                vid, label = line.split(";", 1)
                vid_to_label[vid.strip()] = label.strip()
    return vid_to_label


def filter_by_class(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    labels_dir: str = DEFAULT_LABELS_DIR,
    exclude_classes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove videos belonging to *exclude_classes* from *df* and *meta*.

    Returns filtered copies (originals are not modified).
    """
    if not exclude_classes:
        return df, meta

    vid_to_label = _load_video_labels(labels_dir)
    if not vid_to_label:
        print("  Warning: no label CSVs found - skipping class filter.")
        return df, meta

    exclude_set = set(exclude_classes)
    exclude_vids = {vid for vid, lbl in vid_to_label.items() if lbl in exclude_set}

    if not exclude_vids:
        print(f"  No videos matched excluded classes {exclude_classes}.")
        return df, meta

    n_before = len(meta)
    meta = meta[~meta["video_id"].isin(exclude_vids)].reset_index(drop=True)
    df = df[~df["video_id"].isin(exclude_vids)].reset_index(drop=True)
    n_after = len(meta)
    print(
        f"  Excluded {n_before - n_after:,} videos from classes {exclude_classes} "
        f"({n_after:,} remain)"
    )
    return df, meta


# ==============================================================================
# Data loading
# ==============================================================================
def _count_source_frames(images_dir: str, video_id: str) -> int | None:
    """Count how many JPEG frames the source video folder has.

    Walks possible locations: ``images_dir/<video_id>/`` or nested
    ``images_dir/*/<video_id>/``.  Returns None if not found.
    """
    # Direct child
    direct = os.path.join(images_dir, video_id)
    if os.path.isdir(direct):
        return sum(
            1 for f in os.listdir(direct) if f.lower().endswith((".jpg", ".jpeg"))
        )

    # Nested (e.g. images/4/107946/)
    for sub in os.listdir(images_dir):
        nested = os.path.join(images_dir, sub, video_id)
        if os.path.isdir(nested):
            return sum(
                1 for f in os.listdir(nested) if f.lower().endswith((".jpg", ".jpeg"))
            )
    return None


def load_all_landmarks(
    landmarks_dir: str = DEFAULT_LANDMARKS_DIR,
    images_dir: str | None = DEFAULT_IMAGES_DIR,
    cache_dir: str | None = DEFAULT_CACHE_DIR,
    max_workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load all per-video landmark Parquets into one big DataFrame + metadata.

    Uses a **thread pool** for parallel I/O (reading thousands of small
    Parquet files is I/O-bound, so threads beat processes here).

    If *cache_dir* is given and a cache file exists, data is loaded
    directly from it (seconds instead of minutes).  Otherwise every
    Parquet is read and the result is saved to cache.

    Returns
    -------
    df : pd.DataFrame
        Concatenated landmark rows from all videos, with ``video_id``.
    meta : pd.DataFrame
        One row per video: ``video_id, total_frames, detected_frames,
        n_0_hands, n_1_hand, n_2_hands``.
    """
    # -- Try cache first ---------------------------------------------------
    if cache_dir:
        cached = _load_cache(cache_dir, landmarks_dir)
        if cached is not None:
            return cached

    # -- Discover video directories --------------------------------------
    video_dirs = sorted(
        d
        for d in os.listdir(landmarks_dir)
        if os.path.isdir(os.path.join(landmarks_dir, d))
    )
    if not video_dirs:
        raise FileNotFoundError(f"No video folders found in {landmarks_dir}")

    # -- Parallel loading -----------------------------------------------
    if max_workers is None:
        # I/O-bound: more threads than cores is fine
        max_workers = min(32, (os.cpu_count() or 4) * 4)

    t0 = time.perf_counter()

    def _load_one(vid: str) -> tuple[pd.DataFrame | None, dict]:
        """Read one video's Parquet + build meta row (runs in thread)."""
        lm_path = os.path.join(landmarks_dir, vid, "landmarks.parquet")
        if not os.path.exists(lm_path):
            return None, {}

        df_vid = pd.read_parquet(lm_path)
        df_vid["video_id"] = vid

        # Total frame count from source images (if available)
        total_frames = None
        if images_dir and os.path.isdir(images_dir):
            total_frames = _count_source_frames(images_dir, vid)

        # Hands per frame
        if df_vid.empty:
            detected_frames: set[int] = set()
            hands_per_frame: dict[int, int] = {}
        else:
            hands_per_frame = df_vid.groupby("frame")["hand_index"].count().to_dict()
            detected_frames = set(hands_per_frame.keys())

        if total_frames is None:
            total_frames = (max(detected_frames) + 1) if detected_frames else 0

        n_0 = n_1 = n_2 = 0
        for f_idx in range(total_frames):
            h = hands_per_frame.get(f_idx, 0)
            if h == 0:
                n_0 += 1
            elif h == 1:
                n_1 += 1
            else:
                n_2 += 1

        meta_row = {
            "video_id": vid,
            "total_frames": total_frames,
            "detected_frames": len(detected_frames),
            "missing_frames": total_frames - len(detected_frames),
            "n_0_hands": n_0,
            "n_1_hand": n_1,
            "n_2_hands": n_2,
        }
        return df_vid, meta_row

    all_dfs: list[pd.DataFrame] = []
    meta_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_load_one, vid): vid for vid in video_dirs}
        with tqdm(total=len(futures), desc="Loading landmarks", unit="vid") as pbar:
            for fut in as_completed(futures):
                df_vid, meta_row = fut.result()
                if df_vid is not None:
                    all_dfs.append(df_vid)
                    meta_rows.append(meta_row)
                pbar.update(1)

    dt = time.perf_counter() - t0
    print(f"  Loaded {len(all_dfs)} videos in {dt:.1f}s  ({max_workers} threads)")

    df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    meta = pd.DataFrame(meta_rows)

    # -- Save cache -----------------------------------------------------
    if cache_dir:
        _save_cache(cache_dir, landmarks_dir, df, meta)

    return df, meta


# ==============================================================================
# Statistics
# ==============================================================================
@dataclass
class LandmarkStats:
    """Container for all computed statistics."""

    n_videos: int = 0
    total_frames: int = 0
    detected_frames: int = 0
    missing_frames: int = 0

    # -- Per-frame hand counts (aggregated over ALL videos) ----------------
    global_0_hands: int = 0
    global_1_hand: int = 0
    global_2_hands: int = 0

    # -- Handedness --------------------------------------------------------
    n_left: int = 0
    n_right: int = 0

    # -- Video length distribution -----------------------------------------
    frame_count_mean: float = 0.0
    frame_count_median: float = 0.0
    frame_count_min: int = 0
    frame_count_max: int = 0

    # -- Consecutive missing-frame runs (across all videos) ----------------
    consecutive_missing_runs: list[int] = field(default_factory=list)

    # -- Per-position missing rate (frame index  miss count) --------------
    #    e.g. position_miss_counts[0] = how many videos have no hand in frame 0
    position_miss_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    position_total_counts: np.ndarray = field(default_factory=lambda: np.array([]))

    # -- Cached Histograms & Aggregations ----------------------------------
    rates_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    rates_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    rates_mean: float = 0.0

    lengths_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    lengths_bins: np.ndarray = field(default_factory=lambda: np.array([]))

    missing_runs_hist: np.ndarray = field(default_factory=lambda: np.array([]))
    missing_runs_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    missing_runs_mean: float = 0.0
    missing_runs_median: float = 0.0
    missing_runs_total: int = 0
    missing_runs_counts: dict = field(default_factory=dict)

    class_stats: dict = field(default_factory=dict)

    # -- Meta DataFrame (NOT CACHED TO JSON) -------------------------------
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    unfiltered_meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_json(self, path: str):
        import json

        data = {
            "n_videos": self.n_videos,
            "total_frames": self.total_frames,
            "detected_frames": self.detected_frames,
            "missing_frames": self.missing_frames,
            "global_0_hands": self.global_0_hands,
            "global_1_hand": self.global_1_hand,
            "global_2_hands": self.global_2_hands,
            "n_left": self.n_left,
            "n_right": self.n_right,
            "frame_count_mean": self.frame_count_mean,
            "frame_count_median": self.frame_count_median,
            "frame_count_min": self.frame_count_min,
            "frame_count_max": self.frame_count_max,
            "position_miss_counts": self.position_miss_counts.tolist(),
            "position_total_counts": self.position_total_counts.tolist(),
            "rates_hist": self.rates_hist.tolist(),
            "rates_bins": self.rates_bins.tolist(),
            "rates_mean": self.rates_mean,
            "lengths_hist": self.lengths_hist.tolist(),
            "lengths_bins": self.lengths_bins.tolist(),
            "missing_runs_hist": self.missing_runs_hist.tolist(),
            "missing_runs_bins": self.missing_runs_bins.tolist(),
            "missing_runs_mean": self.missing_runs_mean,
            "missing_runs_median": self.missing_runs_median,
            "missing_runs_total": self.missing_runs_total,
            "missing_runs_counts": self.missing_runs_counts,
            "class_stats": self.class_stats,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def from_json(cls, path: str) -> LandmarkStats:
        import json

        with open(path, "r") as f:
            data = json.load(f)
        s = cls()
        s.n_videos = data["n_videos"]
        s.total_frames = data["total_frames"]
        s.detected_frames = data["detected_frames"]
        s.missing_frames = data["missing_frames"]
        s.global_0_hands = data["global_0_hands"]
        s.global_1_hand = data["global_1_hand"]
        s.global_2_hands = data["global_2_hands"]
        s.n_left = data["n_left"]
        s.n_right = data["n_right"]
        s.frame_count_mean = data["frame_count_mean"]
        s.frame_count_median = data["frame_count_median"]
        s.frame_count_min = data["frame_count_min"]
        s.frame_count_max = data["frame_count_max"]
        s.position_miss_counts = np.array(data["position_miss_counts"])
        s.position_total_counts = np.array(data["position_total_counts"])
        s.rates_hist = np.array(data["rates_hist"])
        s.rates_bins = np.array(data["rates_bins"])
        s.rates_mean = data["rates_mean"]
        s.lengths_hist = np.array(data["lengths_hist"])
        s.lengths_bins = np.array(data["lengths_bins"])
        s.missing_runs_hist = np.array(data["missing_runs_hist"])
        s.missing_runs_bins = np.array(data["missing_runs_bins"])
        s.missing_runs_mean = data["missing_runs_mean"]
        s.missing_runs_median = data["missing_runs_median"]
        s.missing_runs_total = data["missing_runs_total"]
        s.missing_runs_counts = data["missing_runs_counts"]
        s.class_stats = data["class_stats"]
        return s


def compute_stats(
    df: pd.DataFrame, meta: pd.DataFrame, images_dir: str | None = None
) -> LandmarkStats:
    """Compute aggregate statistics from loaded landmark data.

    Optimised: pre-groups by ``video_id`` once, then uses numpy arrays
    instead of Python set lookups per frame.
    """
    t0 = time.perf_counter()
    s = LandmarkStats()
    s.meta = meta
    s.n_videos = len(meta)
    s.total_frames = int(meta["total_frames"].sum())
    s.detected_frames = int(meta["detected_frames"].sum())
    s.missing_frames = int(meta["missing_frames"].sum())

    s.global_0_hands = int(meta["n_0_hands"].sum())
    s.global_1_hand = int(meta["n_1_hand"].sum())
    s.global_2_hands = int(meta["n_2_hands"].sum())

    if not df.empty:
        s.n_left = int((df["handedness"] == 0).sum())
        s.n_right = int((df["handedness"] == 1).sum())

    s.frame_count_mean = float(meta["total_frames"].mean())
    s.frame_count_median = float(meta["total_frames"].median())
    s.frame_count_min = int(meta["total_frames"].min())
    s.frame_count_max = int(meta["total_frames"].max())

    max_len = s.frame_count_max

    # -- Vectorised total_at_pos -------------------------------------------
    # A video with n frames contributes +1 to positions 0..n-1.
    total_at_pos = np.zeros(max_len, dtype=np.int64)
    unique_lens, counts_of_lens = np.unique(
        meta["total_frames"].values, return_counts=True
    )
    for ul, cnt in zip(unique_lens, counts_of_lens):
        if ul > 0:
            total_at_pos[: int(ul)] += int(cnt)

    # -- Pre-group detected frames by video_id (ONE pass over df) ----------
    print("  Pre-grouping frames by video_id...")
    t1 = time.perf_counter()
    if not df.empty:
        detected_map: dict[str, np.ndarray] = (
            df.groupby("video_id")["frame"]
            .apply(lambda x: np.unique(x.values))
            .to_dict()
        )
    else:
        detected_map = {}
    print(f"  Grouped in {time.perf_counter() - t1:.1f}s")

    # -- Per-video: consecutive runs + position miss counts ----------------
    miss_at_pos = np.zeros(max_len, dtype=np.int64)
    all_runs: list[int] = []

    for _, row in tqdm(
        meta.iterrows(), total=len(meta), desc="  Computing stats", unit="vid"
    ):
        vid = row["video_id"]
        n_frames = int(row["total_frames"])
        detected_arr = detected_map.get(vid, np.array([], dtype=np.int64))

        # Build boolean mask: True = hand detected at that frame
        mask = np.zeros(n_frames, dtype=bool)
        valid_idx = detected_arr[detected_arr < n_frames].astype(np.intp)
        mask[valid_idx] = True

        # Position miss counts (vectorised)
        missing_indices = np.where(~mask[:max_len])[0]
        if len(missing_indices) > 0:
            np.add.at(miss_at_pos, missing_indices, 1)

        # Consecutive missing runs (numpy diff trick)
        if n_frames > 0:
            missing = ~mask
            if missing.any():
                # Find run starts/ends via diff
                padded = np.concatenate(([False], missing, [False]))
                diffs = np.diff(padded.astype(np.int8))
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]
                runs = ends - starts
                all_runs.extend(runs.tolist())

    s.consecutive_missing_runs = all_runs
    s.position_miss_counts = miss_at_pos
    s.position_total_counts = total_at_pos

    # -- Pre-compute aggregations for JSON cache ---------------------------

    # 1. Video Length Distribution
    lengths = meta["total_frames"].values
    if len(lengths) > 0:
        s.lengths_bins = np.arange(lengths.min() - 0.5, lengths.max() + 1.5, 1)
        s.lengths_hist, _ = np.histogram(lengths, bins=s.lengths_bins)

    # 2. Per-Video Detection Rate
    if not meta.empty:
        rates = (meta["detected_frames"] / meta["total_frames"] * 100).values
        rates = rates[~np.isnan(rates)]  # Filter out 0-length video NaNs
        np.random.seed(42)
        rates = rates + np.random.uniform(-1.5, 1.5, size=len(rates))
        rates = np.clip(rates, 0, 100)
        s.rates_bins = np.linspace(0, 100, 51)
        s.rates_hist, _ = np.histogram(rates, bins=s.rates_bins)
        s.rates_mean = float(np.mean(rates))

    # 3. Missing Runs
    if all_runs:
        s.missing_runs_total = len(all_runs)
        s.missing_runs_mean = float(np.mean(all_runs))
        s.missing_runs_median = float(np.median(all_runs))
        max_run = max(all_runs)
        s.missing_runs_bins = np.arange(0.5, max_run + 1.5, 1)
        s.missing_runs_hist, _ = np.histogram(all_runs, bins=s.missing_runs_bins)

        s.missing_runs_counts = {
            "runs_1": all_runs.count(1),
            "runs_2_5": sum(1 for r in all_runs if 2 <= r <= 5),
            "runs_6_10": sum(1 for r in all_runs if 6 <= r <= 10),
            "runs_11_plus": sum(1 for r in all_runs if r >= 11),
        }

    # 4. Class Stats
    if "label" in meta.columns:
        grouped = meta.groupby("label")[
            ["n_0_hands", "n_1_hand", "n_2_hands", "total_frames"]
        ].sum()
        s.class_stats = grouped.to_dict(orient="index")

    dt = time.perf_counter() - t0
    print(f"  Stats computed in {dt:.1f}s")
    return s


# ==============================================================================
# Text summary
# ==============================================================================
def format_stats(s: LandmarkStats) -> str:
    """Return a human-readable stats summary string."""

    def pct(n: int, t: int) -> str:
        return f"{100 * n / t:.1f}%" if t > 0 else "N/A"

    lines = [
        "Jester Landmark Statistics",
        "=" * 50,
        f"Videos loaded       : {s.n_videos:,}",
        f"Total frames        : {s.total_frames:,}",
        f"Detected frames     : {s.detected_frames:,} ({pct(s.detected_frames, s.total_frames)})",
        f"Missing frames      : {s.missing_frames:,} ({pct(s.missing_frames, s.total_frames)})",
        "",
        "Hand detection per frame:",
        f"  0 hands  : {s.global_0_hands:>10,}  ({pct(s.global_0_hands, s.total_frames)})",
        f"  1 hand   : {s.global_1_hand:>10,}  ({pct(s.global_1_hand, s.total_frames)})",
        f"  2 hands  : {s.global_2_hands:>10,}  ({pct(s.global_2_hands, s.total_frames)})",
        "",
        "Handedness (total detections):",
        f"  Left     : {s.n_left:>10,}  ({pct(s.n_left, s.n_left + s.n_right)})",
        f"  Right    : {s.n_right:>10,}  ({pct(s.n_right, s.n_left + s.n_right)})",
        "",
        "Video length (frames):",
        f"  Mean     : {s.frame_count_mean:.1f}",
        f"  Median   : {s.frame_count_median:.1f}",
        f"  Min      : {s.frame_count_min}",
        f"  Max      : {s.frame_count_max}",
    ]

    if s.missing_runs_total > 0:
        lines += [
            "",
            "Consecutive missing-frame runs:",
            f"  Total runs       : {s.missing_runs_total:,}",
            f"  Mean run length  : {s.missing_runs_mean:.2f}",
            f"  Median           : {s.missing_runs_median:.1f}",
            f"  Max run length   : {max(s.missing_runs_bins) if len(s.missing_runs_bins) else 0}",
            f"  Runs of 1        : {s.missing_runs_counts.get('runs_1', 0):,}",
            f"  Runs of 2-5      : {s.missing_runs_counts.get('runs_2_5', 0):,}",
            f"  Runs of 6-10     : {s.missing_runs_counts.get('runs_6_10', 0):,}",
            f"  Runs of 11+      : {s.missing_runs_counts.get('runs_11_plus', 0):,}",
        ]

    return "\n".join(lines)


# ==============================================================================
# Plotting
# ==============================================================================
def plot_all(stats: LandmarkStats, output_dir: str) -> None:
    """Generate and save all statistics plots."""
    os.makedirs(output_dir, exist_ok=True)

    _plot_hand_detection_distribution(stats, output_dir)
    _plot_handedness(stats, output_dir)
    _plot_consecutive_missing_runs(stats, output_dir)
    _plot_detection_by_position(stats, output_dir)
    _plot_video_length_distribution(stats, output_dir)
    _plot_detection_rate_per_video(stats, output_dir)
    _plot_hand_detection_by_class(stats, output_dir)


def _plot_hand_detection_by_class(s: LandmarkStats, out: str) -> None:
    """Stacked horizontal bar chart: 0 / 1 / 2 hands detected by class."""
    if not s.class_stats:
        return

    # Convert the cached dict back to a DataFrame for easy plotting
    grouped = pd.DataFrame.from_dict(s.class_stats, orient="index")

    # Sort by semantic meaning
    SEMANTIC_ORDER = [
        "Swiping Up",
        "Swiping Down",
        "Swiping Left",
        "Swiping Right",
        "Sliding Two Fingers Up",
        "Sliding Two Fingers Down",
        "Sliding Two Fingers Left",
        "Sliding Two Fingers Right",
        "Pushing Hand Away",
        "Pulling Hand In",
        "Pushing Two Fingers Away",
        "Pulling Two Fingers In",
        "Rolling Hand Forward",
        "Rolling Hand Backward",
        "Turning Hand Clockwise",
        "Turning Hand Counterclockwise",
        "Zooming In With Full Hand",
        "Zooming Out With Full Hand",
        "Zooming In With Two Fingers",
        "Zooming Out With Two Fingers",
        "Thumb Up",
        "Thumb Down",
        "Stop Sign",
        "Shaking Hand",
        "Drumming Fingers",
        "Doing other things",
        "No gesture",
    ]
    order = [c for c in reversed(SEMANTIC_ORDER) if c in grouped.index]
    grouped = grouped.reindex(order)

    # Calculate percentages
    pct_0 = grouped["n_0_hands"] / grouped["total_frames"] * 100
    pct_1 = grouped["n_1_hand"] / grouped["total_frames"] * 100
    pct_2 = grouped["n_2_hands"] / grouped["total_frames"] * 100

    # Save concrete values to file
    csv_path = os.path.join(out, "hand_detection_by_class.csv")
    out_df = pd.DataFrame(
        {
            "0_hands_pct": pct_0,
            "1_hand_pct": pct_1,
            "2_hands_pct": pct_2,
            "total_frames": grouped["total_frames"],
        }
    )
    old_size = plt.rcParams.get("font.size")
    plt.rcParams.update({"font.size": 7})

    fig, ax = plt.subplots(figsize=(5.9, 5.0))
    labels = grouped.index
    latex_labels = [f"\\makebox[4cm][r]{{\\scriptsize {l}}}" for l in labels]

    ax.barh(latex_labels, pct_0, color="#e74c3c", label="0 hands", edgecolor="none")
    ax.barh(
        latex_labels,
        pct_1,
        left=pct_0,
        color="#2ecc71",
        label="1 hand",
        edgecolor="none",
    )
    ax.barh(
        latex_labels,
        pct_2,
        left=pct_0 + pct_1,
        color="#3498db",
        label="2 hands",
        edgecolor="none",
    )

    ax.set_xlabel("Percentage of frames (%)")
    # ax.set_title("Hand Detection Distribution by Gesture Class")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(labels) - 0.5)
    ax.grid(axis="x", alpha=0.3)

    for label in ax.get_yticklabels():
        label.set_horizontalalignment("right")

    fig.subplots_adjust(left=0.35, right=0.95, bottom=0.11, top=0.98)

    path_png = os.path.join(out, "hand_detection_distribution_by_class.png")
    path_pgf = os.path.join(out, "hand_detection_distribution_by_class.pgf")
    plt.savefig(path_png, dpi=150)
    plt.savefig(path_pgf)
    plt.close(fig)
    print(f"  Saved {path_png} and .pgf")


def _plot_hand_detection_distribution(s: LandmarkStats, out: str) -> None:
    """Bar chart: how many frames have 0 / 1 / 2 hands detected."""
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["0 hands", "1 hand", "2 hands"]
    counts = [s.global_0_hands, s.global_1_hand, s.global_2_hands]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    bars = ax.bar(categories, counts, color=colors, edgecolor="black", alpha=0.85)
    for bar, count in zip(bars, counts):
        pct = 100 * count / s.total_frames if s.total_frames > 0 else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + s.total_frames * 0.005,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Number of frames")
    # ax.set_title(
    #     f"Hand Detection Distribution ({s.total_frames:,} frames, {s.n_videos:,} videos)"
    # )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out, "hand_detection_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_handedness(s: LandmarkStats, out: str) -> None:
    """Pie chart: left vs right hand detections."""
    if s.n_left + s.n_right == 0:
        return
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(
        [s.n_left, s.n_right],
        labels=[f"Left ({s.n_left:,})", f"Right ({s.n_right:,})"],
        colors=["#3498db", "#e67e22"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 8},
    )
    plt.tight_layout()
    path_png = os.path.join(out, "handedness_distribution.png")
    path_pgf = os.path.join(out, "handedness_distribution.pgf")
    plt.savefig(path_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(path_pgf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {path_png} and .pgf")


def _plot_consecutive_missing_runs(s: LandmarkStats, out: str) -> None:
    """Histogram: distribution of consecutive missing-frame run lengths."""
    if len(s.missing_runs_hist) == 0:
        print("  No missing-frame runs to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(5.9, 2.5))

    # Histogram of run lengths
    ax1.hist(
        s.missing_runs_bins[:-1],
        bins=s.missing_runs_bins,
        weights=s.missing_runs_hist,
        color="#e74c3c",
        edgecolor="black",
        alpha=0.8,
    )
    ax1.set_xlabel("Consecutive missing frames (run length)")
    ax1.set_ylabel("Number of runs")
    # ax1.set_title(f"Distribution of Missing-Frame Runs ({s.missing_runs_total:,} total)")
    ax1.set_xlim(0, 40)
    ax1.axvline(
        s.missing_runs_mean,
        color="navy",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {s.missing_runs_mean:.1f}",
    )
    ax1.axvline(
        s.missing_runs_median,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=f"Median: {s.missing_runs_median:.0f}",
    )
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path_png = os.path.join(out, "consecutive_missing_runs.png")
    path_pgf = os.path.join(out, "consecutive_missing_runs.pgf")
    plt.savefig(path_png, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(path_pgf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {path_png} and .pgf")


def _plot_detection_by_position(s: LandmarkStats, out: str) -> None:
    """Bar charts for detection rate and frame context."""
    miss = s.position_miss_counts
    total = s.position_total_counts

    if len(miss) == 0:
        return

    min_videos = max(10, s.n_videos * 0.01)
    valid = total >= min_videos
    n_show = np.max(np.where(valid)) + 1 if valid.any() else len(miss)

    miss_rate = np.zeros(n_show)
    for i in range(n_show):
        if total[i] > 0:
            miss_rate[i] = miss[i] / total[i]

    detection_rate = 1.0 - miss_rate

    # Plot 1: Detection Rate
    fig, ax1 = plt.subplots(figsize=(5.9, 2.5))
    ax1.bar(
        range(n_show),
        detection_rate * 100,
        color="#2ecc71",
        alpha=0.8,
        width=1.0,
        edgecolor="none",
    )
    ax1.set_xlabel("Frame position in video (0-indexed)")
    ax1.set_ylabel("Detection rate (%)")
    # ax1.set_title(f"Hand Detection Rate by Frame Position ({s.n_videos:,} videos)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xlim(-0.5, n_show - 0.5)

    plt.tight_layout()
    path_png1 = os.path.join(out, "detection_rate_by_frame_position.png")
    path_pgf1 = os.path.join(out, "detection_rate_by_frame_position.pgf")
    plt.savefig(path_png1, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(path_pgf1, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {path_png1} and .pgf")

    # Plot 2: Context (Number of videos)
    fig, ax2 = plt.subplots(figsize=(12, 4))
    ax2.bar(
        range(n_show),
        total[:n_show],
        color="#3498db",
        alpha=0.6,
        width=1.0,
        edgecolor="none",
    )
    ax2.set_xlabel("Frame position in video (0-indexed)")
    ax2.set_ylabel("Number of videos")
    # ax2.set_title("Frame Distribution Context")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xlim(-0.5, n_show - 0.5)

    plt.tight_layout()
    path_png2 = os.path.join(out, "videos_by_frame_position.png")
    path_pgf2 = os.path.join(out, "videos_by_frame_position.pgf")
    plt.savefig(path_png2, dpi=150)
    plt.savefig(path_pgf2)
    plt.close(fig)
    print(f"  Saved {path_png2} and .pgf")


def _plot_video_length_distribution(s: LandmarkStats, out: str) -> None:
    """Histogram of video lengths (number of frames per video)."""
    if len(s.lengths_hist) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        s.lengths_bins[:-1],
        bins=s.lengths_bins,
        weights=s.lengths_hist,
        color="#2ecc71",
        edgecolor="black",
        alpha=0.8,
    )
    ax.axvline(
        s.frame_count_mean,
        color="red",
        linestyle="--",
        label=f"Mean: {s.frame_count_mean:.1f}",
    )
    ax.axvline(
        s.frame_count_median,
        color="orange",
        linestyle="--",
        label=f"Median: {s.frame_count_median:.0f}",
    )
    ax.set_xlabel("Video length (frames)")
    ax.set_ylabel("Number of videos")
    # ax.set_title(f"Video Length Distribution ({s.n_videos:,} videos)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out, "video_length_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def _plot_detection_rate_per_video(s: LandmarkStats, out: str) -> None:
    """Histogram of per-video detection rate (% of frames with 1 hand)."""
    if len(s.rates_hist) == 0:
        return

    fig, ax = plt.subplots(figsize=(5.9, 2.5))
    ax.hist(
        s.rates_bins[:-1],
        bins=s.rates_bins,
        weights=s.rates_hist,
        color="#9b59b6",
        edgecolor="black",
        alpha=0.8,
    )
    ax.axvline(
        s.rates_mean,
        color="red",
        linestyle="--",
        label=f"Mean: {s.rates_mean:.1f}%",
    )
    ax.set_xlabel(
        "Detection rate (percentage of frames from the video with a detected hand)"
    )
    ax.set_ylabel("Number of videos")
    # ax.set_title(f"Per-Video Detection Rate ({s.n_videos:,} videos)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path_png = os.path.join(out, "per_video_detection_rate.png")
    path_pgf = os.path.join(out, "per_video_detection_rate.pgf")
    plt.savefig(path_png, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(path_pgf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  Saved {path_png} and .pgf")


# ==============================================================================
# CLI
# ==============================================================================
def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Compute statistics from extracted Jester landmarks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--landmarks",
        default=DEFAULT_LANDMARKS_DIR,
        help="Root dir containing per-video landmark folders",
    )
    p.add_argument(
        "--images",
        default=DEFAULT_IMAGES_DIR,
        help="Root dir of source JPEG video folders (for accurate frame counts)",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_STATS_DIR,
        help="Output dir for plots and stats (default: ./data/jester/stats)",
    )
    p.add_argument(
        "--cache-dir",
        default=DEFAULT_CACHE_DIR,
        help="Cache directory for pre-loaded data. Set to empty string to disable.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Thread pool size for parallel Parquet loading (default: auto)",
    )
    p.add_argument(
        "--labels",
        default=DEFAULT_LABELS_DIR,
        help="Labels CSV directory (for class filtering)",
    )
    p.add_argument(
        "--exclude-classes",
        nargs="*",
        default=["No gesture"],
        help="Gesture classes to exclude from stats/plots (default: 'No gesture'). "
        "Pass with no arguments to include all classes.",
    )
    args = p.parse_args()

    out = args.output or args.landmarks
    cache_dir = args.cache_dir if args.cache_dir else None

    print("Checking JSON plot cache...")
    plot_cache_path = "jester/plot_cache/landmarks_stats.json"
    if os.path.exists(plot_cache_path):
        print(f"Loading cached stats from {plot_cache_path}")
        stats = LandmarkStats.from_json(plot_cache_path)
    else:
        print("Loading landmark data from Parquets...")
        df, full_meta = load_all_landmarks(
            args.landmarks,
            args.images,
            cache_dir=cache_dir,
            max_workers=args.workers,
        )

        # Attach labels to full_meta before filtering so class-wise plotting has all classes
        vid_to_label = _load_video_labels(args.labels)
        if vid_to_label:
            full_meta["label"] = full_meta["video_id"].map(vid_to_label)

        meta = full_meta
        print("Computing statistics...")
        stats = compute_stats(df, meta, args.images)
        stats.unfiltered_meta = full_meta
        stats.to_json(plot_cache_path)

    # Post-cache class filtering
    if args.exclude_classes:
        print(f"Filtering out classes post-cache: {args.exclude_classes}")
        for cls in args.exclude_classes:
            if stats.class_stats and cls in stats.class_stats:
                del stats.class_stats[cls]
        if not stats.meta.empty and "label" in stats.meta.columns:
            stats.meta = stats.meta[~stats.meta["label"].isin(args.exclude_classes)]

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    summary = format_stats(stats)
    print(f"\n{summary}\n")

    # Save stats text
    stats_path = os.path.join(out_dir, "landmark_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"  Saved {stats_path}")

    # Save per-video CSV for further analysis
    if not stats.meta.empty:
        csv_path = os.path.join(out_dir, "per_video_stats.csv")
        stats.meta.to_csv(csv_path, index=False)
        print(f"  Saved {csv_path}")

    print("\nGenerating plots...")
    plot_all(stats, out)
    print("\nDone.")


if __name__ == "__main__":
    main()
