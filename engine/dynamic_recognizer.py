"""Stacked inference pipeline: MediaPipe HandLandmarker → PyTorch dynamic gesture model.

This module provides:

- **LandmarkBuffer** — a thread-safe ring buffer that accumulates per-frame
  landmark vectors from MediaPipe's LIVE_STREAM callback.
- **DynamicGestureRecognizer** — loads a trained PyTorch checkpoint and
  runs inference on the buffered landmark window every time a new frame
  arrives, producing a ``(gesture_name, confidence)`` pair.

The two are glued together in :func:`run_stacked_pipeline` which is called
from ``main.py`` when the user configures ``dynamic_model`` in ``cfg.py``.
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

import numpy as np
import torch

from cfg.constants import COORDS_PER_LANDMARK, FEATURE_DIM
from utils.timeTracker import time_me


class SliderTracker:
    def __init__(self):
        self.active = False
        self.action = None
        self.start_dist = 0.0
        self.start_val = 0.0

    def process(self, left_lm, right_lm, action):
        import math

        from cfg import spot_config

        def get_center(lm_list):
            # Special case: If gesture is "Pointing_Up" (index extended, middle folded)
            # use the tip of the index finger (landmark 8) for precise tracking
            if len(lm_list) > 12:
                index_up = lm_list[8].y < lm_list[6].y
                middle_down = lm_list[12].y > lm_list[10].y
                if index_up and middle_down:
                    return lm_list[8].x, lm_list[8].y

            x = sum([lm.x for lm in lm_list]) / len(lm_list)
            y = sum([lm.y for lm in lm_list]) / len(lm_list)
            return x, y

        lx, ly = get_center(left_lm)
        rx, ry = get_center(right_lm)
        dist = math.hypot(lx - rx, ly - ry)

        param_name = action.split(".")[1]

        if not self.active or self.action != action:
            self.active = True
            self.action = action
            self.start_dist = dist
            self.start_val = getattr(spot_config, param_name, 0.0)

        if param_name == "BODY_HEIGHT_OFFSET":
            # Allow negative offsets for height, scale distance difference
            diff = dist - self.start_dist
            new_val = self.start_val + (diff * 0.5)
            new_val = max(-0.1, min(0.1, new_val))
        else:
            # Update value based on relative distance change
            # Let's say doubling the distance doubles the value, but we avoid div by zero
            if self.start_dist < 0.01:
                new_val = self.start_val
            else:
                # If start_val is 0, provide a default scale
                base_val = self.start_val if self.start_val != 0 else 1.0
                ratio = dist / self.start_dist
                new_val = max(0.0, base_val * ratio)

                # If start_val was 0, just use distance diff
                if self.start_val == 0:
                    new_val = max(0.0, (dist - self.start_dist) * 2.0)

        setattr(spot_config, param_name, new_val)

        return {
            "lx": lx,
            "ly": ly,
            "rx": rx,
            "ry": ry,
            "val": new_val,
            "param": param_name,
        }

    def reset(self):
        self.active = False
        self.action = None


_slider_tracker = SliderTracker()


class GestureSeriesTracker:
    def __init__(self):
        self.first_gesture = None       # e.g. "Closed_Fist" (for UI)
        self.current_prefix = None      # e.g. "A:Closed_Fist,A:Open_Palm" (for matching)
        self.start_time_ms = 0

    def reset(self):
        self.first_gesture = None
        self.current_prefix = None
        self.start_time_ms = 0

    def _get_state(self, timestamp_ms, timeout_ms, series_config):
        active_series = []
        prefix = f"{self.current_prefix},"
        for k, action in series_config.items():
            if k.startswith(prefix) or k == self.current_prefix:
                gestures = [g.split(':')[1] for g in k.split(',')]
                completed_count = len(self.current_prefix.split(',')) if self.current_prefix else 0
                
                active_series.append({
                    "key": k,
                    "gestures": gestures,
                    "completed_count": completed_count,
                    "action": action
                })

        if not active_series:
            return None

        return {
            "active_series": active_series,
            "timeout_ms": timeout_ms,
            "remaining_ms": max(0, timeout_ms - (timestamp_ms - self.start_time_ms))
        }

    def process(self, gesture, hand, timestamp_ms, series_config, timeout_ms, execute_cb, feedback_cb=None):
        candidates = []
        if gesture and hand:
            candidates = [f"{hand}:{gesture}", f"A:{gesture}"]

        if not candidates or not series_config:
            # Check timeout
            if self.current_prefix:
                if timestamp_ms - self.start_time_ms > timeout_ms:
                    print(f"[Series Tracker] Timeout reached for {self.current_prefix}")
                    self.reset()
                    if feedback_cb:
                        feedback_cb(0)
                    return None
                return self._get_state(timestamp_ms, timeout_ms, series_config)
            return None

        # Ignore duplicate bounce gestures
        if self.current_prefix:
            last_prefix_gesture = self.current_prefix.split(',')[-1]
            if f"A:{gesture}" == last_prefix_gesture or f"{hand}:{gesture}" == last_prefix_gesture:
                print(f"[Series Tracker] Ignoring duplicate bounce for {gesture}")
                return self._get_state(timestamp_ms, timeout_ms, series_config)

        # Check if gesture can complete or continue an active series
        if self.current_prefix:
            elapsed = timestamp_ms - self.start_time_ms
            if elapsed <= timeout_ms:
                for cand in candidates:
                    key = f"{self.current_prefix},{cand}"
                    if key in series_config:
                        action = series_config[key]
                        print(f"[Series Tracker] Executing series action: {key} -> {action}")
                        execute_cb(action)
                        self.reset()
                        if feedback_cb:
                            feedback_cb(0)
                        return None
                    
                    # Prefix match: continue series!
                    continues = any(k.startswith(f"{key},") for k in series_config.keys())
                    if continues:
                        print(f"[Series Tracker] Series continued: {key}")
                        self.current_prefix = key
                        self.start_time_ms = timestamp_ms # reset timeout for next gesture!
                        if feedback_cb:
                            feedback_cb(key.count(',') + 1)
                        return self._get_state(timestamp_ms, timeout_ms, series_config)
            print(f"[Series Tracker] Ignored intermediate gesture: {gesture}. Waiting for valid continuation...")
            return self._get_state(timestamp_ms, timeout_ms, series_config)

        # Check if gesture can START a series
        for cand in candidates:
            starts_series = any(k.startswith(f"{cand},") for k in series_config.keys())
            if starts_series:
                print(f"[Series Tracker] Started new series with {cand}")
                self.first_gesture = gesture
                self.current_prefix = cand
                self.start_time_ms = timestamp_ms
                if feedback_cb:
                    feedback_cb(1)
                return self._get_state(timestamp_ms, timeout_ms, series_config)

        return None


_series_tracker = GestureSeriesTracker()


_DYNAMIC_IGNORED_GESTURES = {"doing other things"}


# ══════════════════════════════════════════════════════════════════════════════
# Landmark ring buffer
# ══════════════════════════════════════════════════════════════════════════════
class LandmarkBuffer:
    """Thread-safe ring buffer that stores per-frame landmark vectors.

    Each slot holds a flat numpy array of shape ``(feature_dim,)`` —
    e.g. 63 for 1 hand (21 landmarks × 3 coords) or 126 for 2 hands.

    The buffer is written to by the MediaPipe callback thread and read
    from by the main inference loop.
    """

    def __init__(
        self,
        seq_len: int = 40,
        max_hands: int = 1,
        interpolate_missing: bool = True,
        interpolate_max_gap: int = 4,
    ):
        self.seq_len = seq_len
        self.max_hands = max_hands
        self.feature_dim = FEATURE_DIM * max_hands  # 63 or 126
        self.interpolate_missing = interpolate_missing
        self.interpolate_max_gap = interpolate_max_gap

        self._buf: deque[np.ndarray] = deque(maxlen=seq_len)
        self._valid: deque[bool] = deque(maxlen=seq_len)
        self._lock = threading.Lock()
        self._frame_count = 0

        # Latest raw MediaPipe data — used by the main loop for annotation
        self._latest_hand_landmarks: list = []
        self._latest_handedness: list = []
        self._imputation_stats = {
            "gap_segments_interpolated": 0,
            "gap_frames_interpolated": 0,
        }

    @property
    def ready(self) -> bool:
        """True once the buffer has been filled to ``seq_len`` frames."""
        with self._lock:
            return len(self._buf) >= self.seq_len

    @property
    def frame_count(self) -> int:
        with self._lock:
            return self._frame_count

    def push(self, landmarks: np.ndarray) -> None:
        """Push a single frame's flattened landmark vector.

        Parameters
        ----------
        landmarks : np.ndarray, shape ``(feature_dim,)``
            Zero-vector if no hand was detected in this frame.
        """
        assert landmarks.shape == (self.feature_dim,), (
            f"Expected ({self.feature_dim},), got {landmarks.shape}"
        )
        is_valid = bool(np.any(np.abs(landmarks) > 0))
        with self._lock:
            self._buf.append(landmarks)
            self._valid.append(is_valid)
            if self.interpolate_missing and is_valid:
                self._interpolate_recent_gap_locked()
            self._frame_count += 1

    def _interpolate_recent_gap_locked(self) -> None:
        """Fill the latest closed missing run, if it is short enough.

        Lock must already be held.
        """
        end_idx = len(self._buf) - 1
        if end_idx <= 0 or not self._valid[end_idx]:
            return

        j = end_idx - 1
        while j >= 0 and not self._valid[j]:
            j -= 1

        if j < 0:
            return

        gap_start = j + 1
        gap_end = end_idx - 1
        gap_len = gap_end - gap_start + 1
        if gap_len <= 0 or gap_len > self.interpolate_max_gap:
            return

        left_vec = self._buf[j]
        right_vec = self._buf[end_idx]
        denom = gap_len + 1
        for k in range(1, gap_len + 1):
            alpha = k / denom
            idx = j + k
            self._buf[idx] = (1.0 - alpha) * left_vec + alpha * right_vec
            self._valid[idx] = True

        self._imputation_stats["gap_segments_interpolated"] += 1
        self._imputation_stats["gap_frames_interpolated"] += gap_len

    def push_from_mediapipe(
        self,
        hand_landmarks: list,
        handedness: list,
    ) -> None:
        """Convert raw MediaPipe hand landmarks to a flat vector and push.

        Parameters
        ----------
        hand_landmarks : list of NormalizedLandmarkList
            ``result.hand_landmarks`` from MediaPipe.
        handedness : list of list[Category]
            ``result.handedness`` from MediaPipe.
        """
        vec = np.zeros(self.feature_dim, dtype=np.float32)

        if hand_landmarks:
            for h_idx in range(min(len(hand_landmarks), self.max_hands)):
                lm_list = hand_landmarks[h_idx]
                offset = h_idx * FEATURE_DIM
                for i, lm in enumerate(lm_list):
                    base = offset + i * COORDS_PER_LANDMARK
                    vec[base] = lm.x
                    vec[base + 1] = lm.y
                    vec[base + 2] = lm.z

        # Store latest raw data for annotation / frontend drawing
        with self._lock:
            self._latest_hand_landmarks = hand_landmarks
            self._latest_handedness = handedness

        self.push(vec)

    def update_static_gestures(self, gestures: list, handedness: list) -> None:
        """Extract and store the most recent static gestures for Left/Right hands."""
        left_static = None
        right_static = None
        for idx, h in enumerate(handedness):
            if not gestures or idx >= len(gestures) or not gestures[idx]:
                continue
            cat = h[0].category_name
            g_cat = gestures[idx][0].category_name
            g_score = gestures[idx][0].score
            if cat == "Left":
                left_static = (g_cat, g_score)
            elif cat == "Right":
                right_static = (g_cat, g_score)

        with self._lock:
            self._latest_left_static = left_static
            self._latest_right_static = right_static

    def get_latest_static_gestures(
        self,
    ) -> tuple[tuple[str, float] | None, tuple[str, float] | None]:
        """Return the most recent (left_static, right_static) gesture tuples."""
        with self._lock:
            return getattr(self, "_latest_left_static", None), getattr(
                self, "_latest_right_static", None
            )

    def get_latest_landmarks(self) -> tuple[list, list]:
        """Return the most recent ``(hand_landmarks, handedness)`` from MediaPipe."""
        with self._lock:
            return self._latest_hand_landmarks, self._latest_handedness

    def get_window(self) -> np.ndarray:
        """Return the current buffer as a ``(seq_len, feature_dim)`` array.

        If fewer than ``seq_len`` frames are available, the beginning is
        zero-padded (matching the training convention).
        """
        with self._lock:
            buf_list = list(self._buf)

        n = len(buf_list)
        window = np.zeros((self.seq_len, self.feature_dim), dtype=np.float32)
        if n > 0:
            start = max(0, self.seq_len - n)
            window[start:] = np.array(buf_list[-(self.seq_len - start) :])
        return window

    def valid_count_in_window(self) -> int:
        """Return how many frames in the current window contain non-empty landmarks."""
        with self._lock:
            return sum(self._valid)

    def get_and_reset_imputation_stats(self) -> dict[str, int]:
        with self._lock:
            out = dict(self._imputation_stats)
            self._imputation_stats = {
                "gap_segments_interpolated": 0,
                "gap_frames_interpolated": 0,
            }
            return out

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._valid.clear()
            self._frame_count = 0
            self._latest_hand_landmarks = []
            self._latest_handedness = []
            self._imputation_stats = {
                "gap_segments_interpolated": 0,
                "gap_frames_interpolated": 0,
            }


# ══════════════════════════════════════════════════════════════════════════════
# Dynamic gesture recognizer (PyTorch inference)
# ══════════════════════════════════════════════════════════════════════════════
class DynamicGestureRecognizer:
    """Wraps a trained model for real-time inference (supports PyTorch and ONNX)."""

    def __init__(
        self,
        model_path: str,
        labels_csv: str,
        device: str = "auto",
        confidence_threshold: float = 0.5,
    ):
        from jester.training.data import load_label_map
        from utils.model_loader import UnifiedModelRunner

        # ── Device ────────────────────────────────────────────────────────
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Model ─────────────────────────────────────────────────────────
        print(f"  Loading dynamic model: {model_path}")
        self.runner = UnifiedModelRunner(model_path, device=self.device)
        print(
            f"  Type: {self.runner.runner_type}, "
            f"classes: {self.runner.num_classes}, "
            f"device: {self.device}"
        )

        # ── Label map (sized to model's num_classes) ─────────────────────
        label_map = load_label_map(labels_csv)  # {name: idx}
        num_classes = self.runner.num_classes
        self.class_names: list[str] = [f"unknown_{i}" for i in range(num_classes)]
        for name, idx in label_map.items():
            if idx < num_classes:
                self.class_names[idx] = name
        if len(label_map) != num_classes:
            print(
                f"  WARNING: labels CSV has {len(label_map)} entries but "
                f"model has {num_classes} classes"
            )

        self.confidence_threshold = confidence_threshold
        self._warmup()

    def _warmup(self) -> None:
        """Run one dummy forward pass to trigger JIT compilation / CUDA init."""
        dummy = np.zeros((self.runner.seq_len, self.runner.input_dim), dtype=np.float32)
        self.runner(dummy)

    def predict(self, window: np.ndarray) -> tuple[str | None, float]:
        """Run inference on a landmark window.

        Parameters
        ----------
        window : np.ndarray, shape ``(seq_len, feature_dim)``

        Returns
        -------
        (gesture_name, confidence) or (None, 0.0) if below threshold.
        """
        logits = self.runner(window)  # (C,)

        # Softmax computation
        mx = np.max(logits)
        logits_exp = np.exp(logits - mx)
        probs = logits_exp / np.sum(logits_exp)

        conf = float(np.max(probs))
        idx = int(np.argmax(probs))
        gesture_name = self.class_names[idx]

        if conf < self.confidence_threshold:
            return None, conf

        if gesture_name.casefold() in _DYNAMIC_IGNORED_GESTURES:
            return None, conf

        return gesture_name, conf


# ══════════════════════════════════════════════════════════════════════════════
# Stacked pipeline (called from main.py)
# ══════════════════════════════════════════════════════════════════════════════

# Module-level shared buffer — written by callback, read by main loop
_landmark_buffer: LandmarkBuffer | None = None


def get_landmark_buffer() -> LandmarkBuffer:
    """Return the module-level landmark buffer (created by run_stacked_pipeline)."""
    assert _landmark_buffer is not None, (
        "LandmarkBuffer not initialised - call run_stacked_pipeline() first"
    )
    return _landmark_buffer


def run_stacked_pipeline() -> None:
    """Main entry-point for the stacked MediaPipe + PyTorch pipeline.

    1. Creates the landmark buffer and dynamic recognizer.
    2. Starts the MediaPipe HandLandmarker in LIVE_STREAM mode with a
       callback that feeds landmarks into the buffer.
    3. Loops reading webcam frames, sending them to MediaPipe, and
       every frame polls the buffer for PyTorch inference.
    """
    from cfg import cfg
    from cfg.options_loader import load_recognizer_options
    from data_loader.data_loader import load_video_frames, stream_webcam
    from engine.callback import recognizer_ready
    from engine.gesture_action_mapping import gesture_mapper
    from engine.processor import GestureFilter
    from utils.mediapipe_extensions import create_recognizer

    is_live_stream = cfg.running_mode == "LIVE_STREAM"
    if is_live_stream:
        from web_ui.stats import save_livestream_session_stats
        from web_ui.streaming_web_server.backend import (
            receive_dynamic,
            start_requested,
            stop_requested,
        )

    global _landmark_buffer

    dm_cfg = cfg.dynamic_model

    # ── Resolve paths ─────────────────────────────────────────────────────
    model_path = dm_cfg["model"]
    labels_csv = dm_cfg.get("labels_csv", "./models/dynamic_classes.csv")
    seq_len = dm_cfg.get("seq_len", 40)
    max_hands = dm_cfg.get("max_hands", 1)
    device = dm_cfg.get("device", "auto")
    confidence_threshold = dm_cfg.get("confidence_threshold", 0.5)
    show_fps = dm_cfg.get("show_fps", True)
    interpolate_missing = dm_cfg.get("interpolate_missing", True)
    interpolate_max_gap = dm_cfg.get("interpolate_max_gap", 4)
    min_valid_detections = dm_cfg.get("min_valid_detections", 10)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Dynamic model not found: {model_path}")

    # ── Create shared buffer ──────────────────────────────────────────────
    _landmark_buffer = LandmarkBuffer(
        seq_len=seq_len,
        max_hands=max_hands,
        interpolate_missing=interpolate_missing,
        interpolate_max_gap=interpolate_max_gap,
    )

    # ── Load Inference Engine ─────────────────────────────────────────────
    recognizer = DynamicGestureRecognizer(
        model_path=model_path,
        labels_csv=labels_csv,
        device=device,
        confidence_threshold=confidence_threshold,
    )

    # ── Gesture filter (temporal smoothing for the dynamic output) ────────
    gesture_filter = GestureFilter()
    static_left_filter = GestureFilter()
    static_right_filter = GestureFilter()

    # ── Create MediaPipe HandLandmarker (LIVE_STREAM) ─────────────────────
    # Override running mode to LIVE_STREAM
    original_mode = cfg.running_mode
    cfg.running_mode = "LIVE_STREAM"
    options = load_recognizer_options()
    cfg.running_mode = original_mode  # restore
    landmarker = create_recognizer(options)

    print("\n=== Stacked Pipeline: MediaPipe -> PyTorch ===")
    print(f"  Seq length : {seq_len}")
    print(f"  Max hands  : {max_hands}")
    print(f"  Mode       : {cfg.running_mode}")
    print(f"  Threshold  : {confidence_threshold}")
    print(f"  Min valid  : {min_valid_detections}")
    print(
        f"  Imputation : enabled={interpolate_missing}, max_gap={interpolate_max_gap}"
    )
    print(f"  Buffer fills in {seq_len} frames, then inference begins.\n")

    # ── Inference loop ────────────────────────────────────────────────────
    web_only = (
        bool(getattr(cfg, "livestream_web_only", False)) if is_live_stream else False
    )
    session_idx = 0
    total_frame_count = 0
    fps_interval = 30  # print FPS every N frames

    last_gesture = None
    last_left_static = None
    last_right_static = None
    t_start = time.perf_counter()

    try:
        while True:
            if is_live_stream and (web_only or session_idx > 0):
                print("Waiting for /start from web app (or Ctrl+C to exit)...")
                while not start_requested.is_set():
                    time.sleep(0.2)
                start_requested.clear()
                print("\n--- Starting stream ---\n")

            if is_live_stream:
                stop_requested.clear()
            _landmark_buffer.clear()
            gesture_filter.reset()
            static_left_filter.reset()
            static_right_filter.reset()
            _slider_tracker.reset()
            _series_tracker.reset()
            last_gesture: str | None = None
            current_gesture: str | None = None
            current_confidence: float = 0.0
            last_left_conf: float = 1.0
            last_right_conf: float = 1.0
            last_static_execute_time = 0.0
            static_retry_interval = getattr(cfg, "static_retry_interval", 1.5)
            frame_count = 0
            t_start = time.perf_counter()
            recognizer_ready.set()

            frame_iter = (
                stream_webcam() if is_live_stream else load_video_frames(cfg.data)
            )
            for mp_image, timestamp_ms in frame_iter:
                stable = None

                if is_live_stream and stop_requested.is_set():
                    print("\nStream stopped by frontend.")
                    break

                # Send frame to MediaPipe (async — callback fills the buffer)
                landmarker.run_auto(mp_image, timestamp_ms)

                # Wait for the async callback to finish processing this frame
                recognizer_ready.wait()
                recognizer_ready.clear()

                frame_count += 1
                total_frame_count += 1
                time_me("landmarks")

                # Run PyTorch inference when buffer is ready
                if _landmark_buffer.ready:
                    valid_detections = _landmark_buffer.valid_count_in_window()
                    if valid_detections >= min_valid_detections:
                        window = _landmark_buffer.get_window()
                        gesture_name, confidence = recognizer.predict(window)
                        current_gesture = gesture_name
                        current_confidence = confidence
                        if getattr(cfg, "debug_pipeline_prints", False) and gesture_name:
                            print(f"[Debug] PyTorch Output: {gesture_name} ({confidence:.2f})")

                        # Feed into temporal filter (reuse existing GestureFilter
                        # by wrapping the string as a simple object with .category_name)
                        gesture_filter.add_frame(
                            _GestureLabel(gesture_name) if gesture_name else None
                        )
                        stable = gesture_filter.get_stable_gesture()

                        if getattr(cfg, "debug_pipeline_prints", False) and gesture_name:
                            if stable:
                                print(f"[Debug] Temporal Filter Accepted (Dynamic): {stable}")
                            else:
                                print(f"[Debug] Temporal Filter Dropped (Dynamic Unstable): {gesture_name}")

                        # Handled in the prioritization block below
                    else:
                        current_gesture = None
                        current_confidence = 0.0
                        gesture_filter.add_frame(None)
                time_me("dynamic_gesture_detection")

                # ── Feed annotated frame to web frontend ──────────────────
                if is_live_stream:
                    mp_h_landmarks, _ = _landmark_buffer.get_latest_landmarks()
                    left_static, right_static = (
                        _landmark_buffer.get_latest_static_gestures()
                    )

                    stable_left = None
                    stable_right = None
                    slider_data = None
                    combo_action = None

                    if getattr(cfg, "debug_pipeline_prints", False):
                        if left_static: print(f"[Debug] MediaPipe Output (Left):  {left_static[0]} ({left_static[1]:.2f})")
                        if right_static: print(f"[Debug] MediaPipe Output (Right): {right_static[0]} ({right_static[1]:.2f})")

                    # ALWAYS update static filters
                    l_val = _GestureLabel(left_static[0]) if left_static else None
                    r_val = _GestureLabel(right_static[0]) if right_static else None
                    static_left_filter.add_frame(l_val)
                    static_right_filter.add_frame(r_val)

                    stable_left = static_left_filter.get_stable_gesture()
                    stable_right = static_right_filter.get_stable_gesture()
                    if getattr(cfg, "debug_pipeline_prints", False):
                        if left_static and not stable_left: print(f"[Debug] Temporal Filter Dropped (Left Unstable): {left_static[0]}")
                        if stable_left: print(f"[Debug] Temporal Filter Accepted (Left): {stable_left}")
                        if right_static and not stable_right: print(f"[Debug] Temporal Filter Dropped (Right Unstable): {right_static[0]}")
                        if stable_right: print(f"[Debug] Temporal Filter Accepted (Right): {stable_right}")

                    from cfg.gesture_config import gesture_mappings

                    combo_key = f"{stable_left}+{stable_right}" if stable_left and stable_right else None
                    combo_action = gesture_mappings.get("combinations", {}).get(combo_key) if combo_key else None
                    is_slider = combo_action and combo_action.startswith("config.")

                    series_config = gesture_mappings.get("series", {})
                    is_series_active = _series_tracker.current_prefix is not None

                    # Find out what changed for Single / Series actions
                    left_changed = (stable_left and stable_left != last_left_static)
                    right_changed = (stable_right and stable_right != last_right_static)
                    active_hand = None
                    active_gesture = None
                    if left_changed:
                        active_gesture = stable_left
                        active_hand = "L"
                    elif right_changed:
                        active_gesture = stable_right
                        active_hand = "R"

                    dynamic_action = gesture_mappings.get("dynamic", {}).get(stable) if stable else None
                    
                    # ----------------------------------------------------
                    # NEW PRIORITIZATION LOGIC
                    # 1. Slider (Live)
                    # 2. Series (Live)
                    # 3. Valid Mapped Dynamic
                    # 4. Valid Mapped Combo / Single (Handled by GestureActionMapper)
                    # ----------------------------------------------------

                    def visual_feedback_cb(step_index: int):
                        if "spot" not in gesture_mapper._handlers:
                            return
                        spot_handler = gesture_mapper._handlers["spot"]
                        try:
                            if step_index == 0:
                                spot_handler.reset_body_orientation()
                            elif step_index % 2 == 1:
                                spot_handler.micro_tilt_left()
                            else:
                                spot_handler.micro_tilt_right()
                        except Exception as e:
                            print(f"[Feedback] Spot feedback error: {e}")

                    series_state = None
                    handled = False

                    if is_slider:
                        # 1. Slider Mode
                        if mp_h_landmarks and len(mp_h_landmarks) >= 2:
                            lm1 = mp_h_landmarks[0]
                            lm2 = mp_h_landmarks[1]

                            cx1 = sum(lm.x for lm in lm1) / len(lm1)
                            cx2 = sum(lm.x for lm in lm2) / len(lm2)

                            if cx1 < cx2:
                                left_lm, right_lm = lm1, lm2
                            else:
                                right_lm, left_lm = lm1, lm2

                            slider_data = _slider_tracker.process(
                                left_lm, right_lm, combo_action
                            )

                            if slider_data["param"] == "BODY_HEIGHT_OFFSET":
                                now = time.time()
                                if not hasattr(_slider_tracker, "last_exec") or now - getattr(_slider_tracker, "last_exec", 0) > 0.1:
                                    from actions import spot
                                    if hasattr(spot, "_controller") and spot._controller.is_connected or spot._controller._simulation_mode:
                                        threading.Thread(target=spot.stand_height, args=(slider_data["val"],), daemon=True).start()
                                        _slider_tracker.last_exec = now

                        last_left_static = stable_left
                        last_right_static = stable_right
                        if is_series_active:
                            visual_feedback_cb(0)
                        _series_tracker.reset()
                        handled = True

                    else:
                        _slider_tracker.reset()

                        # 2. Series / Stacked Gestures
                        def exec_series(action):
                            gesture_mapper._invoke_action(action)
                        
                        series_timeout = getattr(cfg, "dynamic_model", {}).get("series_timeout_ms", 2000)

                        if is_series_active or (active_gesture and active_hand):
                            series_state = _series_tracker.process(
                                active_gesture,
                                active_hand,
                                timestamp_ms,
                                series_config,
                                series_timeout,
                                exec_series,
                                visual_feedback_cb,
                            )
                        
                        if series_state:
                            # Series is currently running or just started!
                            if left_changed: last_left_static = stable_left
                            if right_changed: last_right_static = stable_right
                            handled = True

                    if not handled:
                        # 3. Dynamic Action
                        if dynamic_action:
                            if stable != last_dynamic:
                                print(f"  >> Dynamic gesture: {stable}")
                                gesture_mapper.execute(stable, None)
                                last_dynamic = stable
                            last_left_static = stable_left
                            last_right_static = stable_right
                        else:
                            last_dynamic = stable # Sync dynamic so it doesn't trigger unexpectedly later if mapping is added
                            
                            # 4. Combo & 5. Single Hands
                            if not stable_left: last_left_static = None
                            if not stable_right: last_right_static = None

                            force_retry = False
                            if (stable_left or stable_right) and not left_changed and not right_changed:
                                if time.perf_counter() - last_static_execute_time > static_retry_interval:
                                    force_retry = True

                            if left_changed or right_changed or force_retry:
                                if force_retry:
                                    print(f"  >> Retrying Static gesture: L={stable_left}, R={stable_right}")
                                else:
                                    print(f"  >> Static gesture: L={stable_left}, R={stable_right}")
                                gesture_mapper.execute(stable_left, stable_right)
                                last_static_execute_time = time.perf_counter()
                                if left_changed: last_left_static = stable_left
                                if right_changed: last_right_static = stable_right

                    if left_static and left_static[0] == stable_left:
                        last_left_conf = left_static[1]
                    if right_static and right_static[0] == stable_right:
                        last_right_conf = right_static[1]

                    receive_dynamic(
                        image_np=mp_image.numpy_view(),
                        hand_landmarks=mp_h_landmarks,
                        gesture_text=current_gesture,
                        confidence=current_confidence,
                        timestamp_ms=timestamp_ms,
                        left_static=(stable_left, last_left_conf) if stable_left else None,
                        right_static=(stable_right, last_right_conf) if stable_right else None,
                        slider_data=slider_data,
                        series_state=series_state,
                    )

                # FPS counter
                if show_fps and frame_count % fps_interval == 0:
                    elapsed = time.perf_counter() - t_start
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    buf_fill = min(_landmark_buffer.frame_count, seq_len)
                    print(
                        f"  [{frame_count}] {fps:.1f} FPS | buffer: {buf_fill}/{seq_len}",
                        end="\r",
                    )

            # Explicitly close the generator to release the camera deterministically when breaking out of loop
            frame_iter.close()

            if is_live_stream:
                save_livestream_session_stats()
            imputation_stats = _landmark_buffer.get_and_reset_imputation_stats()
            print(
                "  Landmark-imputation impact (session): "
                f"segments={imputation_stats['gap_segments_interpolated']}, "
                f"frames={imputation_stats['gap_frames_interpolated']}"
            )

            session_idx += 1
            if not is_live_stream:
                break

    except KeyboardInterrupt:
        print("\n\nPipeline stopped by user (Ctrl+C).")
    finally:
        landmarker.close()
        print(f"\nProcessed {total_frame_count} frames total")


class _GestureLabel:
    """Minimal wrapper so dynamic gesture names work with GestureFilter.

    GestureFilter accesses ``.category_name`` on buffered items.
    """

    def __init__(self, name: str):
        self.category_name = name

    def __eq__(self, other):
        if isinstance(other, _GestureLabel):
            return self.category_name == other.category_name
        return NotImplemented

    def __hash__(self):
        return hash(self.category_name)
