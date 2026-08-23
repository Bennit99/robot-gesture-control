from mediapipe.tasks.python import vision


def is_landmark_model(model_path: str = None) -> bool:
    """Check if the configured model is a hand landmarker (vs gesture recognizer)."""
    from cfg import cfg

    path = model_path or cfg.model
    return "landmark" in path.lower()


def create_recognizer(options):
    """Factory: create the right recognizer based on model type in cfg."""
    if is_landmark_model():
        return AutoHandLandmarker.create_from_options(options)
    else:
        return AutoGestureRecognizer.create_from_options(options)


class AutoHandLandmarker(vision.HandLandmarker):
    """HandLandmarker with automatic mode selection for detect calls."""

    def run_auto(self, image, timestamp_ms=None, image_processing_options=None):
        mode = (
            self.options._running_mode
            if hasattr(self, "options")
            else self._running_mode
        )
        mode_name = mode.name if hasattr(mode, "name") else str(mode)
        if mode_name == "IMAGE":
            return self.detect(image, image_processing_options)
        elif mode_name == "VIDEO":
            if timestamp_ms is None:
                raise ValueError("timestamp_ms required for VIDEO mode")
            return self.detect_for_video(image, timestamp_ms, image_processing_options)
        elif mode_name == "LIVE_STREAM":
            if timestamp_ms is None:
                raise ValueError("timestamp_ms required for LIVE_STREAM mode")
            return self.detect_async(image, timestamp_ms, image_processing_options)
        else:
            raise ValueError(f"Unsupported running mode: {mode_name}")


class MyHandLandmarkerResult(vision.HandLandmarkerResult):
    """Extended HandLandmarkerResult with utility methods."""

    def __init__(self, base_result):
        super().__init__(
            handedness=base_result.handedness,
            hand_landmarks=base_result.hand_landmarks,
            hand_world_landmarks=base_result.hand_world_landmarks,
        )

    def summary_str(self):
        """Returns a list of strings summarizing detected hands."""
        return [
            f"{h[0].category_name}: {len(lm)} landmarks"
            for h, lm in zip(self.handedness, self.hand_landmarks)
        ]


def _wrap_result(result):
    """Wrap a raw mediapipe result in the appropriate My* wrapper."""
    if is_landmark_model():
        return MyHandLandmarkerResult(result)
    else:
        return MyGestureRecognizerResult(result)


class AutoGestureRecognizer(vision.GestureRecognizer):
    """GestureRecognizer with automatic mode selection for recognize calls."""

    def run_auto(self, image, timestamp_ms=None, image_processing_options=None):
        mode = (
            self.options._running_mode
            if hasattr(self, "options")
            else self._running_mode
        )
        mode_name = mode.name if hasattr(mode, "name") else str(mode)
        if mode_name == "IMAGE":
            return self.recognize(image, image_processing_options)
        elif mode_name == "VIDEO":
            if timestamp_ms is None:
                raise ValueError("timestamp_ms required for VIDEO mode")
            return self.recognize_for_video(
                image, timestamp_ms, image_processing_options
            )
        elif mode_name == "LIVE_STREAM":
            if timestamp_ms is None:
                raise ValueError("timestamp_ms required for LIVE_STREAM mode")
            return self.recognize_async(image, timestamp_ms, image_processing_options)
        else:
            raise ValueError(f"Unsupported running mode: {mode_name}")


class MyGestureRecognizerResult(vision.GestureRecognizerResult):
    """Extended GestureRecognizerResult with utility methods."""

    def __init__(self, base_result):
        """Initialize MyGestureRecognizerResult from a GestureRecognizerResult object."""
        super().__init__(
            gestures=base_result.gestures,
            handedness=base_result.handedness,
            hand_landmarks=base_result.hand_landmarks,
            hand_world_landmarks=base_result.hand_world_landmarks,
        )

    def gestures_str(self):
        """Prints the top gestures and their handedness."""
        top_gestures = [gesture_list[0] for gesture_list in self.gestures]
        handedness = [h[0].category_name for h in self.handedness]
        return [
            f"{handedness[idx]}: {gesture.category_name} {round(gesture.score, 2)}"
            for idx, gesture in enumerate(top_gestures)
        ]

    def summary_str(self):
        """Returns a list of strings summarizing detected hands and gestures."""
        return self.gestures_str()

    def left_gesture(self) -> str | None:
        """Returns the top gesture for the left hand, or None if not detected."""
        for idx, h in enumerate(self.handedness):
            if h[0].category_name == "Left":
                return self.gestures[idx][0]
        return None

    def right_gesture(self) -> str | None:
        """Returns the top gesture for the right hand, or None if not detected."""
        for idx, h in enumerate(self.handedness):
            if h[0].category_name == "Right":
                return self.gestures[idx][0]
        return None
