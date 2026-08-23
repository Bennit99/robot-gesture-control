"""PyTorch Dataset and Lightning DataModule for Jester landmarks.

Loads pre-extracted hand-landmark Parquet files, pads/truncates to a
fixed sequence length, and exposes train / val / test splits via a
single ``JesterLandmarkDataModule``.
"""

from __future__ import annotations

import json
import os

import numpy as np

try:
    import pyarrow  # Eager import to prevent Windows DLL SegFault with PyTorch
except ImportError:
    pass
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L  # type: ignore[no-redef]

# -- Constants (canonical source: cfg.constants) ------------------------------
from cfg.constants import (
    COORDS_PER_LANDMARK,
    DEFAULT_SEQ_LEN,
    FEATURE_DIM,
    N_LANDMARKS,
)
from utils.landmark_imputation import interpolate_missing_landmark_gaps

__all__ = ["COORDS_PER_LANDMARK", "DEFAULT_SEQ_LEN", "FEATURE_DIM", "N_LANDMARKS"]


# ==============================================================================
# Label loading
# ==============================================================================
def load_label_map(labels_csv: str) -> dict[str, int]:
    """Load ``jester-v1-labels.csv`` -> {gesture_name: class_index}."""
    with open(labels_csv, encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return {name: idx for idx, name in enumerate(names)}


def load_split(
    split_csv: str, label_map: dict[str, int] | None = None
) -> list[tuple[str, int | None]]:
    """Load a split CSV.  Returns list of ``(video_id, label_idx | None)``.

    Train/val CSVs have ``video_id;label``, test CSV has just ``video_id``.
    """
    entries: list[tuple[str, int | None]] = []
    with open(split_csv, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ";" in line:
                vid, label_name = line.split(";", 1)
                label_idx = label_map[label_name] if label_map else None
            else:
                vid = line
                label_idx = None
            entries.append((vid.strip(), label_idx))
    return entries


# ==============================================================================
# Dataset
# ==============================================================================
class JesterLandmarkDataset(Dataset):
    """PyTorch dataset that loads one landmark sequence per sample.

    Each sample is a tensor of shape ``(seq_len, feature_dim)`` where
    ``feature_dim = 63`` (21 landmarks x 3 coords).  Sequences shorter
    than *seq_len* are zero-padded; longer ones are truncated.

    When two hands are detected in a frame, their landmarks are
    concatenated (126-dim) or only the first hand is kept depending on
    ``max_hands``.

    Parameters
    ----------
    entries     : list of (video_id, label_index | None)
    landmarks_dir : root directory with per-video landmark sub-folders
    seq_len     : fixed sequence length (frames)
    max_hands   : 1 or 2 - how many hands to include per frame
    world       : use world_landmarks.parquet instead of landmarks.parquet
    min_landmarks: minimum number of non-zero landmarks per frame
    augment     : apply random scaling and shifting
    scale_range : (min, max) for random scaling
    shift_range : (min, max) for random shifting
    """

    def __init__(
        self,
        entries: list[tuple[str, int | None]],
        landmarks_dir: str,
        seq_len: int = DEFAULT_SEQ_LEN,
        max_hands: int = 1,
        world: bool = False,
        interpolate_missing: bool = False,
        interpolate_max_gap: int = 4,
        stats_name: str = "dataset",
        auto_extract_missing: bool = True,
        images_dir: str | None = None,
        model_path: str = "models/hand_landmarker.task",
        use_gpu_for_extraction: bool = True,
        cache_path: str | None = None,
        min_landmarks: int = 10,
        no_gesture_label_idx: int | None = None,
        augment: bool = True,
        scale_range: tuple[float, float] = (1.0, 4.0),
        shift_range: tuple[float, float] = (-0.5, 0.5),
        sabbarese_mode: bool = False,
    ):
        self.entries = entries
        self.landmarks_dir = landmarks_dir
        self.seq_len = seq_len
        self.max_hands = max_hands
        self.feature_dim = FEATURE_DIM * max_hands
        self.parquet_name = "world_landmarks.parquet" if world else "landmarks.parquet"
        self.interpolate_missing = interpolate_missing
        self.interpolate_max_gap = interpolate_max_gap
        self.stats_name = stats_name
        self.auto_extract_missing = auto_extract_missing
        self.images_dir = images_dir or self._infer_images_dir(landmarks_dir)
        self.model_path = model_path
        self.use_gpu_for_extraction = use_gpu_for_extraction
        self.cache_path = cache_path
        self.min_landmarks = min_landmarks
        self.no_gesture_label_idx = no_gesture_label_idx
        self.augment = augment
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.sabbarese_mode = sabbarese_mode

        # In sabbarese mode, we load the full video (up to 100 frames) into cache
        self.cache_seq_len = 100 if sabbarese_mode else seq_len

        # If a matching preload cache exists, trust it and skip filesystem checks.
        self._valid: list[tuple[str, int | None]] = list(entries)
        cache_exists = bool(self.cache_path and os.path.exists(self.cache_path))

        if cache_exists:
            print(f"  [Dataset] Initializing '{self.stats_name}' split from CACHED data: {self.cache_path}")
        else:
            # Keep entries with existing landmarks; optionally extract missing ones.
            self._valid = []
            missing: list[tuple[str, int | None]] = []
            for vid, label in entries:
                pq = os.path.join(landmarks_dir, vid, self.parquet_name)
                if os.path.exists(pq):
                    self._valid.append((vid, label))
                else:
                    missing.append((vid, label))

            if missing and self.auto_extract_missing:
                self._extract_missing_landmarks(missing)
                for vid, label in missing:
                    pq = os.path.join(landmarks_dir, vid, self.parquet_name)
                    if os.path.exists(pq):
                        self._valid.append((vid, label))

            if len(self._valid) < len(entries):
                n_missing = len(entries) - len(self._valid)
                print(
                    f"  [Dataset] {n_missing} / {len(entries)} videos have no "
                    f"landmark files - skipped."
                )
                if len(self._valid) == 0:
                    print("  [!] WARNING: NO DATA FOUND! Please download the dataset from:")
                    print("      https://www.qualcomm.com/developer/software/jester-dataset/downloads")
            
            print(f"  [Dataset] Initializing '{self.stats_name}' split from RAW DATASET ({len(self._valid)} samples).")

        # Preloaded tensors (populated by preload())
        self._x: torch.Tensor | None = None  # (N, seq_len, feature_dim)
        self._y: torch.Tensor | None = None  # (N,) or None
        self._vids: list[str] | None = None
        self._preloaded = False

        self._imputation_stats = {
            "videos_processed": 0,
            "gap_segments_interpolated": 0,
            "gap_frames_interpolated": 0,
            "interpolate_missing": bool(self.interpolate_missing),
            "interpolate_max_gap": int(self.interpolate_max_gap),
        }
        self._stats_seen_videos: set[str] = set()
        self._imputation_stats_emitted = False

    def _infer_images_dir(self, landmarks_dir: str) -> str | None:
        """Infer the Jester images root from a landmarks root."""
        root = os.path.normpath(landmarks_dir)
        parent = os.path.dirname(root)
        candidate = os.path.join(parent, "images")
        return candidate if os.path.isdir(candidate) else None

    def _extract_missing_landmarks(self, missing: list[tuple[str, int | None]]) -> None:
        """Extract landmarks for missing videos so they can be loaded immediately."""
        if not self.images_dir:
            print(
                "  [Dataset] Cannot auto-extract missing landmarks: images_dir not found."
            )
            return

        if not os.path.isdir(self.images_dir):
            print(
                f"  [Dataset] Cannot auto-extract missing landmarks: "
                f"images_dir does not exist: {self.images_dir}"
            )
            return

        print(
            f"  [Dataset] Missing landmarks for {len(missing)} / {len(self.entries)} videos. "
            "Attempting on-demand extraction in parallel..."
        )

        try:
            from jester.extract_landmarks import run_parallel
        except Exception as e:
            print(f"  [Dataset] Failed to import extractor: {e}")
            return

        run_parallel(
            data_dir=self.images_dir,
            output_dir=self.landmarks_dir,
            model_path=self.model_path,
            use_gpu=self.use_gpu_for_extraction,
        )

        print(
            f"  [Dataset] On-demand extraction finished for {len(missing)} missing videos."
        )

    # -- Preloading / caching ----------------------------------------------
    def preload(
        self,
        device: torch.device | str = "cpu",
        cache_path: str | None = None,
    ) -> None:
        """Load ALL samples into a single tensor and move to *device*.

        If *cache_path* is given and the ``.pt`` file exists, tensors are
        loaded directly from disk (very fast).  Otherwise every Parquet is
        read, converted, and the result is saved to *cache_path* for next
        time.

        After calling this, ``__getitem__`` becomes a simple index into
        pre-allocated GPU/CPU tensors - no per-sample I/O or parsing.
        """
        import time

        t0 = time.perf_counter()

        if cache_path and os.path.exists(cache_path):
            self._load_from_cache(cache_path, device)
            self._log_cached_imputation_stats(cache_path)
        else:
            self._build_from_parquets(device)
            if cache_path:
                self._save_cache(cache_path)
            self._emit_imputation_stats(cache_path)

        elapsed = time.perf_counter() - t0
        mb = self._x.nbytes / 1024 / 1024 if self._x is not None else 0
        extra = self._y.nbytes / 1024 / 1024 if self._y is not None else 0
        print(
            f"  Preloaded {len(self._valid)} samples -> {device}  "
            f"({mb + extra:.1f} MB, {elapsed:.1f}s)"
        )

    def _build_from_parquets(self, device: torch.device | str) -> None:
        """Read every Parquet, build tensors, move to *device*."""

        n = len(self._valid)
        x_buf = torch.zeros(
            n, self.cache_seq_len, self.feature_dim, dtype=torch.float32
        )
        has_labels = self._valid[0][1] is not None if n > 0 else False
        y_buf = torch.zeros(n, dtype=torch.long) if has_labels else None
        vids: list[str] = []
        valid_indices: list[int] = []
        kept = 0

        def process_vid(i, vid, label):
            pq_path = os.path.join(self.landmarks_dir, vid, self.parquet_name)
            try:
                df = pd.read_parquet(pq_path)
            except Exception:
                return None
            if self.min_landmarks > 0 and label != self.no_gesture_label_idx:
                try:
                    n_detected = int((df.groupby("frame").size()).shape[0])
                except Exception:
                    n_detected = 0
                if n_detected < self.min_landmarks:
                    return None
            return (i, vid, label, df)

        print(f"  [Dataset] Building cache for {n} videos using ThreadPool...")

        try:
            for i, (vid, label) in enumerate(self._valid):
                try:
                    res = process_vid(i, vid, label)
                    if res is not None:
                        _, vid, label, df = res
                        tensor = self._df_to_tensor(df, video_id=vid)
                        x_buf[kept] = tensor
                        if has_labels and y_buf is not None and label is not None:
                            y_buf[kept] = label
                        vids.append(vid)
                        valid_indices.append(i)
                        kept += 1
                except Exception as e:
                    print(f"  [Dataset] Error on video {vid}: {e}")

                if (i + 1) % 1000 == 0:
                    print(f"  [Dataset] Progress: {i + 1}/{n}", flush=True)
                    import gc

                    gc.collect()

            self._valid = [self._valid[idx] for idx in valid_indices]

            # Slice buffers to actual kept items
            x_buf = x_buf[:kept]
            if y_buf is not None:
                y_buf = y_buf[:kept]

            self._x = x_buf.to(device, non_blocking=True)
            self._y = y_buf.to(device, non_blocking=True) if y_buf is not None else None
            self._vids = vids
            self._preloaded = True
        except Exception:
            with open("DEBUG_CRASH.txt", "w", encoding="utf-8") as f:
                import traceback

                f.write(traceback.format_exc())
            raise

        if self.min_landmarks > 0 and kept < n:
            print(
                f"  [Dataset] Filter dropped {n - kept} / {n} videos with <{self.min_landmarks} landmark frames."
            )

    def _save_cache(self, cache_path: str) -> None:
        """Save preloaded tensors to a ``.pt`` file on disk."""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # Always save as CPU tensors so the cache is device-agnostic
        payload = {
            "x": self._x.cpu() if self._x is not None else None,
            "y": self._y.cpu() if self._y is not None else None,
            "vids": self._vids,
        }
        torch.save(payload, cache_path)
        mb = os.path.getsize(cache_path) / 1024 / 1024
        print(f"  Saved cache -> {cache_path}  ({mb:.1f} MB)")

    def _load_from_cache(self, cache_path: str, device: torch.device | str) -> None:
        """Load pre-built tensors from a ``.pt`` cache file."""
        print(f"  Loading cache <- {cache_path}")
        payload = torch.load(cache_path, map_location=device, weights_only=True)
        self._x = payload["x"]
        self._y = payload["y"]
        self._vids = payload["vids"]
        self._preloaded = True

        # Reconcile cache with current split entries by video_id.
        # This prevents stale caches (from a different split/version) from
        # silently producing mismatched video_id/label rows.
        if self._vids is None:
            self._valid = []
            return

        cache_index: dict[str, int] = {}
        for i, vid in enumerate(self._vids):
            if vid not in cache_index:
                cache_index[vid] = i

        selected_indices: list[int] = []
        aligned_valid: list[tuple[str, int | None]] = []
        for vid, label in self._valid:
            idx = cache_index.get(vid)
            if idx is not None:
                selected_indices.append(idx)
                aligned_valid.append((vid, label))

        if len(selected_indices) != len(self._vids):
            print(
                "  Cache/split mismatch detected: "
                f"using {len(selected_indices)} of {len(self._vids)} cached samples."
            )

        if self._x is not None:
            self._x = self._x[selected_indices]
        if self._y is not None:
            self._y = self._y[selected_indices]
        self._vids = [self._vids[i] for i in selected_indices]
        self._valid = aligned_valid

    def _imputation_stats_output_path(self, cache_path: str | None = None) -> str:
        if cache_path:
            root, _ = os.path.splitext(cache_path)
            return f"{root}_imputation_stats.json"
        base = (
            f"{self.stats_name}_imputation_stats_"
            f"s{self.seq_len}_h{self.max_hands}_"
            f"i{int(self.interpolate_missing)}g{self.interpolate_max_gap}.json"
        )
        return os.path.join(self.landmarks_dir, base)

    def _log_cached_imputation_stats(self, cache_path: str) -> None:
        stats_path = self._imputation_stats_output_path(cache_path)
        if not os.path.exists(stats_path):
            print("  Landmark-imputation stats not found next to cache (skip).")
            return
        try:
            with open(stats_path, encoding="utf-8") as f:
                stats = json.load(f)
            print(
                "  Landmark-imputation (from cache): "
                f"segments={stats.get('gap_segments_interpolated', 0)}, "
                f"frames={stats.get('gap_frames_interpolated', 0)}"
            )
        except Exception as e:
            print(f"  Failed to read cached landmark-imputation stats: {e}")

    def _emit_imputation_stats(self, cache_path: str | None = None) -> None:
        if self._imputation_stats_emitted:
            return
        s = self._imputation_stats
        if s["videos_processed"] == 0:
            return

        print(
            "  Landmark-imputation impact: "
            f"segments={s['gap_segments_interpolated']}, "
            f"frames={s['gap_frames_interpolated']} "
            f"(videos={s['videos_processed']}, enabled={s['interpolate_missing']}, "
            f"max_gap={s['interpolate_max_gap']})"
        )

        out_path = self._imputation_stats_output_path(cache_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        print(f"  Saved landmark-imputation stats -> {out_path}")
        self._imputation_stats_emitted = True

    def __len__(self) -> int:
        return len(self._valid)

    def __getitem__(self, idx: int) -> dict:
        # Fast path: preloaded tensors
        if self._preloaded:
            assert self._x is not None and self._vids is not None
            x = self._x[idx]
            label = self._y[idx] if self._y is not None else None
            vid = self._vids[idx]
        else:
            # Slow path: read from disk
            vid, label = self._valid[idx]
            pq_path = os.path.join(self.landmarks_dir, vid, self.parquet_name)
            df = pd.read_parquet(pq_path)
            x = self._df_to_tensor(df, video_id=vid)

            if (not self._preloaded) and (not self._imputation_stats_emitted):
                if len(self._stats_seen_videos) >= len(self._valid):
                    self._emit_imputation_stats(cache_path=None)

        if self.sabbarese_mode:
            # Determine actual valid length (where it's not all zeros)
            non_zero_frames = (x.abs().sum(dim=-1) > 0).nonzero(as_tuple=True)[0]
            if len(non_zero_frames) > 0:
                actual_len = non_zero_frames.max().item() + 1
            else:
                actual_len = 0

            actual_len = max(actual_len, 1)  # At least 1

            if self.stats_name == "train":
                # Random window of size self.seq_len
                start = 0
                if actual_len > self.seq_len:
                    start = torch.randint(0, actual_len - self.seq_len + 1, (1,)).item()
                x_crop = x[start : start + self.seq_len]
                # If shorter than seq_len, pad with zeros
                if len(x_crop) < self.seq_len:
                    pad = torch.zeros(self.seq_len - len(x_crop), self.feature_dim)
                    x_crop = torch.cat([x_crop, pad], dim=0)

                if self.augment:
                    x_crop = self._apply_augmentation(x_crop)

                item = {"x": x_crop, "video_id": vid}
                if label is not None:
                    item["y"] = label
                return item
            else:
                # Validation/Test: Extract all overlapping windows
                windows = []
                # Ensure we get at least one window if the video is short
                num_windows = max(1, actual_len - self.seq_len + 1)

                for start in range(num_windows):
                    x_crop = x[start : start + self.seq_len].clone()
                    if len(x_crop) < self.seq_len:
                        pad = torch.zeros(self.seq_len - len(x_crop), self.feature_dim)
                        x_crop = torch.cat([x_crop, pad], dim=0)
                    windows.append(x_crop)

                x_stacked = torch.stack(
                    windows, dim=0
                )  # (num_windows, seq_len, feature_dim)

                # Cannot apply spatial augmentations safely to multiple independent sliding windows easily
                # but validation shouldn't use augmentations anyway.
                item = {"x": x_stacked, "video_id": vid}
                if label is not None:
                    item["y"] = torch.tensor([label] * num_windows, dtype=torch.long)
                return item
        else:
            if self.augment:
                x = self._apply_augmentation(x)
            item = {"x": x, "video_id": vid}
            if label is not None:
                item["y"] = label
            return item

    def _df_to_tensor(
        self, df: pd.DataFrame, video_id: str | None = None
    ) -> torch.Tensor:
        """Convert a landmark DataFrame to a fixed-size tensor.

        Returns shape ``(cache_seq_len, feature_dim)``.
        """
        if df.empty:
            return torch.zeros(
                self.cache_seq_len, self.feature_dim, dtype=torch.float32
            )

        # Coordinate columns: x0,y0,z0 ... x20,y20,z20
        coord_cols = [
            c for c in df.columns if c[0] in ("x", "y", "z") and c[1:].isdigit()
        ]

        max_frame = int(df["frame"].max())
        n_frames = max_frame + 1

        # Allocate (n_frames, max_hands, 63)
        arr = np.zeros((n_frames, self.max_hands, len(coord_cols)), dtype=np.float32)

        for _, row in df.iterrows():
            f = int(row["frame"])
            h = int(row["hand_index"])
            if h < self.max_hands:
                arr[f, h] = row[coord_cols].values.astype(np.float32)

        # Flatten hands dim -> (n_frames, feature_dim)
        arr = arr.reshape(n_frames, -1)

        # Pad or truncate to cache_seq_len
        if n_frames >= self.cache_seq_len:
            arr = arr[: self.cache_seq_len]
        else:
            pad = np.zeros(
                (self.cache_seq_len - n_frames, arr.shape[1]), dtype=np.float32
            )
            arr = np.concatenate([arr, pad], axis=0)

        if self.interpolate_missing:
            arr, interp_stats = interpolate_missing_landmark_gaps(
                arr,
                max_gap=self.interpolate_max_gap,
                in_place=False,
            )
            if video_id is not None and video_id not in self._stats_seen_videos:
                self._stats_seen_videos.add(video_id)
                self._imputation_stats["videos_processed"] += 1
                self._imputation_stats["gap_segments_interpolated"] += int(
                    interp_stats.gap_segments_interpolated
                )
                self._imputation_stats["gap_frames_interpolated"] += int(
                    interp_stats.gap_frames_interpolated
                )
        elif video_id is not None and video_id not in self._stats_seen_videos:
            self._stats_seen_videos.add(video_id)
            self._imputation_stats["videos_processed"] += 1

        return torch.from_numpy(arr)

    def _apply_augmentation(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random isotropic scaling and spatial shifting to landmarks.

        Operates on a tensor of shape ``(seq_len, feature_dim)``.
        Landmarks are stored as (x0,y0,z0, x1,y1,z1, ), so we reshape
        to ``(seq_len, n_landmarks, 3)`` for coordinate-aware transforms,
        then flatten back.

        * **Scaling**  a single random factor drawn per sample (uniform in
          ``scale_range``) multiplies every coordinate.  This simulates
          being closer / farther from the camera.
        * **Shifting**  a single random (dx, dy) offset (uniform in
          ``shift_range``) is added to x and y coords of every landmark
          across all frames.  This simulates the hand appearing at a
          different position in the frame.  z is left unchanged.
        """
        # Create a non-zero mask so padding stays zero
        nonzero_mask = (tensor.abs().sum(dim=-1, keepdim=True) > 0).float()

        # Reshape -> (seq_len, n_landmarks_total, 3)
        seq_len, feat = tensor.shape
        n_lm = feat // 3
        t = tensor.reshape(seq_len, n_lm, 3).clone()

        # Random physical distance scale (1.0 to 2.0 simulates moving 1x to 2x further away)
        # Because perceived 2D scale is inversely proportional to physical distance,
        # we sample the distance factor uniformly and invert it for the 2D scale.
        lo, hi = self.scale_range
        dist_factor = torch.empty(1).uniform_(lo, hi).item()
        scale = 1.0 / dist_factor

        # Scale spatial coordinates relative to their geometric centroid to simulate camera zoom
        # without unintentionally translating the hand across the frame.
        cx = t[:, :, 0].mean(dim=1, keepdim=True)
        cy = t[:, :, 1].mean(dim=1, keepdim=True)

        t[:, :, 0] = (t[:, :, 0] - cx) * scale + cx
        t[:, :, 1] = (t[:, :, 1] - cy) * scale + cy
        t[:, :, 2] = t[:, :, 2] * scale  # Z is already centered at the wrist

        # Compute valid bounds (ignoring zero-padded frames)
        mask_bool = nonzero_mask.squeeze(-1) > 0

        if mask_bool.any():
            valid_x = t[:, :, 0][mask_bool]
            valid_y = t[:, :, 1][mask_bool]
            min_x, max_x = valid_x.min().item(), valid_x.max().item()
            min_y, max_y = valid_y.min().item(), valid_y.max().item()
        else:
            min_x, max_x, min_y, max_y = 0.0, 1.0, 0.0, 1.0

        slo, shi = self.shift_range

        # Calculate allowable shifts to keep landmarks strictly within [0, 1]
        allowed_min_dx = max(slo, -min_x)
        allowed_max_dx = min(shi, 1.0 - max_x)

        allowed_min_dy = max(slo, -min_y)
        allowed_max_dy = min(shi, 1.0 - max_y)

        # Sample dx
        if allowed_min_dx < allowed_max_dx:
            dx = torch.empty(1).uniform_(allowed_min_dx, allowed_max_dx).item()
        else:
            dx = 0.0

        # Sample dy
        if allowed_min_dy < allowed_max_dy:
            dy = torch.empty(1).uniform_(allowed_min_dy, allowed_max_dy).item()
        else:
            dy = 0.0

        t[:, :, 0] += dx
        t[:, :, 1] += dy

        # Flatten back and re-apply zero-padding mask
        out = t.reshape(seq_len, feat) * nonzero_mask
        return out


# ==============================================================================
# DataModule
# ==============================================================================
class JesterLandmarkDataModule(L.LightningDataModule):
    """Lightning DataModule for Jester landmarks.

    Parameters
    ----------
    landmarks_dir : path to extracted landmark folders
    labels_dir    : path to jester-v1-*.csv files
    seq_len       : fixed sequence length (frames)
    max_hands     : 1 or 2
    world         : use world landmarks
    batch_size    : training batch size
    num_workers   : DataLoader workers
    """

    def __init__(
        self,
        landmarks_dir: str = "./data/jester/landmarks",
        labels_dir: str = "./data/jester/lables",
        seq_len: int = DEFAULT_SEQ_LEN,
        max_hands: int = 1,
        world: bool = False,
        interpolate_missing: bool = False,
        interpolate_max_gap: int = 4,
        batch_size: int = 64,
        num_workers: int = 4,
        preload_device: str | None = None,
        min_landmarks: int = 0,
        augment: bool = False,
        scale_range: tuple[float, float] = (1.0, 4.0),
        shift_range: tuple[float, float] = (-0.5, 0.5),
        sabbarese_mode: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.landmarks_dir = landmarks_dir
        self.labels_dir = labels_dir
        self.seq_len = seq_len
        self.max_hands = max_hands
        self.world = world
        self.interpolate_missing = interpolate_missing
        self.interpolate_max_gap = interpolate_max_gap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preload_device = preload_device  # None = don't preload
        self.min_landmarks = min_landmarks
        self.augment = augment
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.sabbarese_mode = sabbarese_mode
        self.cache_dir: str | None = None  # set externally or via train CLI

        self.label_map: dict[str, int] = {}
        self.num_classes: int = 0
        self.feature_dim: int = FEATURE_DIM * max_hands

        self._train_ds: JesterLandmarkDataset | None = None
        self._val_ds: JesterLandmarkDataset | None = None
        self._test_ds: JesterLandmarkDataset | None = None

    def prepare_data(self) -> None:
        """Verify that label files exist (called on rank 0 only)."""
        labels_csv = os.path.join(self.labels_dir, "jester-v1-labels.csv")
        if not os.path.exists(labels_csv):
            raise FileNotFoundError(
                f"Labels file not found: {labels_csv}\n"
                "Download the Jester dataset from: https://www.qualcomm.com/developer/software/jester-dataset/downloads"
            )

    def setup(self, stage: str | None = None) -> None:
        """Load label map and create datasets for the requested stage."""
        labels_csv = os.path.join(self.labels_dir, "jester-v1-labels.csv")
        self.label_map = load_label_map(labels_csv)
        self.num_classes = len(self.label_map)

        no_gesture_idx = self.label_map.get("No gesture")

        if stage in ("fit", None):
            train_csv = os.path.join(self.labels_dir, "jester-v1-train.csv")
            train_entries = load_split(train_csv, self.label_map)
            train_cache_path = (
                self._cache_path("train") if self.preload_device is not None else None
            )
            self._train_ds = JesterLandmarkDataset(
                train_entries,
                self.landmarks_dir,
                self.seq_len,
                self.max_hands,
                self.world,
                interpolate_missing=self.interpolate_missing,
                interpolate_max_gap=self.interpolate_max_gap,
                stats_name="train",
                cache_path=train_cache_path,
                min_landmarks=self.min_landmarks,
                no_gesture_label_idx=no_gesture_idx,
                augment=self.augment,
                scale_range=self.scale_range,
                shift_range=self.shift_range,
                sabbarese_mode=self.sabbarese_mode,
            )
            print(f"  Train set: {len(self._train_ds)} videos", flush=True)

            val_csv = os.path.join(self.labels_dir, "jester-v1-validation.csv")
            val_entries = load_split(val_csv, self.label_map)
            val_cache_path = (
                self._cache_path("val") if self.preload_device is not None else None
            )
            self._val_ds = JesterLandmarkDataset(
                val_entries,
                self.landmarks_dir,
                self.seq_len,
                self.max_hands,
                self.world,
                interpolate_missing=self.interpolate_missing,
                interpolate_max_gap=self.interpolate_max_gap,
                stats_name="val",
                cache_path=val_cache_path,
                min_landmarks=self.min_landmarks,
                no_gesture_label_idx=no_gesture_idx,
                augment=self.augment,
                scale_range=self.scale_range,
                shift_range=self.shift_range,
                sabbarese_mode=self.sabbarese_mode,
            )
            print(f"  Val set:   {len(self._val_ds)} videos", flush=True)

            # Preload into GPU/CPU if requested
            if self.preload_device is not None:
                self._train_ds.preload(
                    self.preload_device,
                    cache_path=self._cache_path("train"),
                )
                self._val_ds.preload(
                    self.preload_device,
                    cache_path=self._cache_path("val"),
                )

        if stage in ("test", None):
            test_csv = os.path.join(self.labels_dir, "jester-v1-test.csv")
            if os.path.exists(test_csv):
                test_entries = load_split(test_csv, None)
                test_cache_path = (
                    self._cache_path("test")
                    if self.preload_device is not None
                    else None
                )
                self._test_ds = JesterLandmarkDataset(
                    test_entries,
                    self.landmarks_dir,
                    self.seq_len,
                    self.max_hands,
                    self.world,
                    interpolate_missing=self.interpolate_missing,
                    interpolate_max_gap=self.interpolate_max_gap,
                    stats_name="test",
                    cache_path=test_cache_path,
                    min_landmarks=self.min_landmarks,
                    no_gesture_label_idx=no_gesture_idx,
                    sabbarese_mode=self.sabbarese_mode,
                )
                print(f"  Test set:  {len(self._test_ds)} videos")

    def _cache_path(self, split: str) -> str | None:
        """Return the .pt cache file path for a split, or None."""
        if self.cache_dir is None:
            return None
        tag = "world" if self.world else "lm"
        interp_tag = f"i{int(self.interpolate_missing)}g{self.interpolate_max_gap}"
        sab_tag = "sab" if self.sabbarese_mode else ""
        fname = f"{split}_s{self.seq_len}_h{self.max_hands}_{tag}_{interp_tag}_ml{self.min_landmarks}{sab_tag}.pt"
        return os.path.join(self.cache_dir, fname)

    @staticmethod
    def _collate(batch: list[dict]) -> dict:
        """Custom collate that handles optional labels and Sabbarese multiple windows."""
        if (
            batch[0]["x"].dim() == 3
        ):  # Sabbarese val mode: (num_windows, seq_len, feature_dim)
            x = torch.cat([b["x"] for b in batch], dim=0)
            out: dict = {"x": x, "video_id": []}
            for b in batch:
                out["video_id"].extend([b["video_id"]] * len(b["x"]))
            if "y" in batch[0]:
                y = torch.cat([b["y"] for b in batch], dim=0)
                out["y"] = y
            return out
        else:
            x = torch.stack([b["x"] for b in batch])
            out = {"x": x, "video_id": [b["video_id"] for b in batch]}
            if "y" in batch[0]:
                out["y"] = torch.tensor([b["y"] for b in batch], dtype=torch.long)
            return out

    def _loader_kwargs(self, shuffle: bool) -> dict:
        """Common DataLoader kwargs, adapted for preloaded vs. lazy mode."""
        preloaded = self.preload_device is not None
        return {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            # Workers / pinning are pointless when data is already on device
            "num_workers": 0 if preloaded else self.num_workers,
            "pin_memory": not preloaded,
            "collate_fn": self._collate,
            "persistent_workers": (False if preloaded else self.num_workers > 0),
        }

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None, "Call setup('fit') first"
        return DataLoader(self._train_ds, **self._loader_kwargs(shuffle=True))

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None, "Call setup('fit') first"
        return DataLoader(self._val_ds, **self._loader_kwargs(shuffle=False))

    def test_dataloader(self) -> DataLoader:
        if self._test_ds is None:
            raise RuntimeError("Test dataset not loaded - call setup('test') first")
        return DataLoader(self._test_ds, **self._loader_kwargs(shuffle=False))
