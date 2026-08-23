from collections import Counter, deque

from cfg import cfg


class GestureFilter:
    """
    Temporal filter for MediaPipe gesture results to avoid false triggers.

    Uses a sliding buffer (Moving Window) and threshold logic to accept only stable
    gestures. Additionally implements hysteresis to prevent flickering during rapid switches.
    """

    def __init__(self):
        """
        Initializes the GestureFilter.
        configured via cfg.filter dictionary.
        """
        self.window_size = cfg.filter["window_size"]
        self.threshold = cfg.filter["threshold"]
        self.hysteresis_frames = cfg.filter["hysteresis_frames"]

        # Buffer for gesture labels
        self._gesture_buffer: deque = deque(maxlen=self.window_size)

        # Hysteresis tracking
        self._last_stable_gesture: str | None = None
        self._frames_since_change = 0

    def add_frame(self, gesture_label: str | None) -> None:
        """
        Adds a new gesture label to the buffer.

        Args:
            gesture_label: Label of the recognized gesture (e.g., "Open_Palm", "Closed_Fist")
        """
        self._gesture_buffer.append(gesture_label)

    def get_stable_gesture(self) -> str | None:
        """
        Checks if the most common gesture exceeds the threshold.
        Uses hysteresis logic to prevent flickering.

        Returns:
            Label of the stable gesture or None if no stable result is available
        """
        # Buffer must be filled
        if len(self._gesture_buffer) < self.window_size:
            return None

        # Exclude None and the string "None" from consideration
        filtered_buffer = [
            g
            for g in self._gesture_buffer
            if g is not None and g.category_name != "None"
        ]
        if not filtered_buffer:
            return None

        # Calculate gesture frequency on filtered data (ignoring None for the counting part)
        gesture_counts = Counter(g.category_name for g in filtered_buffer)
        most_common_gesture, count = gesture_counts.most_common(1)[0]

        # Calculate ratio of the most common gesture relative to the ENTIRE window size
        # This ensures that empty frames (None) dilute the stability score.
        stability_ratio = count / self.window_size

        # Threshold not reached
        if stability_ratio < self.threshold:
            return None

        # Hysteresis: threshold for gesture switching
        if most_common_gesture != self._last_stable_gesture:
            # New switch detected
            if self._frames_since_change < self.hysteresis_frames:
                self._frames_since_change += 1
                # Switch not yet accepted, return old gesture
                return self._last_stable_gesture
            else:
                # Enough frames have passed for switch
                self._last_stable_gesture = most_common_gesture
                self._frames_since_change = 0
        else:
            # Same gesture, reset counter
            self._frames_since_change = 0

        return self._last_stable_gesture

    def reset(self) -> None:
        """Resets the filter (buffer and hysteresis)."""
        self._gesture_buffer.clear()
        self._last_stable_gesture = None
        self._frames_since_change = 0

    def get_statistics(self) -> dict:
        """
        Returns statistics about the current buffer state (for debugging).

        Returns:
            Dictionary with buffer fill level and gesture frequencies
        """
        if len(self._gesture_buffer) == 0:
            return {"buffer_fill": 0, "gesture_counts": {}}

        gesture_counts = Counter(self._gesture_buffer)
        return {
            "buffer_fill": len(self._gesture_buffer) / self.window_size,
            "gesture_counts": dict(gesture_counts),
            "last_stable_gesture": self._last_stable_gesture,
        }
