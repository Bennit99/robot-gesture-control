from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.processors import ClassifierOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)

from cfg import cfg
from engine.callback import result_callback
from utils.mediapipe_extensions import is_landmark_model


def load_recognizer_options(model_asset_path: str = cfg.model):
    """Auto-detect model type and return the appropriate options object."""
    if is_landmark_model(model_asset_path):
        return _load_hand_landmarker_options(model_asset_path)
    else:
        return _load_gesture_recognizer_options(model_asset_path)


def _load_hand_landmarker_options(
    model_asset_path: str = cfg.model,
) -> vision.HandLandmarkerOptions:
    """Loads HandLandmarkerOptions from cfg.py settings."""
    base_options = python.BaseOptions(model_asset_path=model_asset_path)

    running_mode_str = getattr(cfg, "running_mode", "IMAGE")
    print(f"Running mode: {running_mode_str} (HandLandmarker)")
    running_mode = VisionTaskRunningMode[running_mode_str]

    options_kwargs = dict(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=getattr(cfg, "num_hands", 1),
        min_hand_detection_confidence=getattr(
            cfg, "min_hand_detection_confidence", 0.5
        ),
        min_hand_presence_confidence=getattr(cfg, "min_hand_presence_confidence", 0.5),
        min_tracking_confidence=getattr(cfg, "min_tracking_confidence", 0.5),
    )
    if running_mode == VisionTaskRunningMode.LIVE_STREAM:
        # Use the dynamic-pipeline callback when dynamic_model is configured
        dm = getattr(cfg, "dynamic_model", None)
        # Accept either legacy 'checkpoint' key or the more common 'model' key
        if isinstance(dm, dict) and (dm.get("checkpoint") or dm.get("model")):
            from engine.callback import dynamic_landmark_callback

            options_kwargs["result_callback"] = dynamic_landmark_callback
        else:
            from engine.callback import landmark_result_callback

            options_kwargs["result_callback"] = landmark_result_callback
    return vision.HandLandmarkerOptions(**options_kwargs)


def _load_gesture_recognizer_options(
    model_asset_path: str = cfg.model,
) -> vision.GestureRecognizerOptions:  # type: ignore
    """
    Loads GestureRecognizerOptions from cfg.py settings.
    """
    base_options = python.BaseOptions(model_asset_path=model_asset_path)
    static_gestures = getattr(cfg, "static_gestures", {})
    confidence_threshold = static_gestures.get("confidence_threshold", 0.5)

    canned_options = ClassifierOptions(score_threshold=confidence_threshold)
    custom_options = ClassifierOptions(score_threshold=confidence_threshold)

    # Map string to VisionTaskRunningMode enum
    running_mode_str = getattr(cfg, "running_mode", "IMAGE")
    print(f"Running mode: {running_mode_str}")
    running_mode = VisionTaskRunningMode[running_mode_str]

    options_kwargs = dict(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=getattr(cfg, "num_hands", 1),
        min_hand_detection_confidence=getattr(
            cfg, "min_hand_detection_confidence", 0.5
        ),
        min_hand_presence_confidence=getattr(cfg, "min_hand_presence_confidence", 0.5),
        min_tracking_confidence=getattr(cfg, "min_tracking_confidence", 0.5),
        canned_gesture_classifier_options=canned_options,
        custom_gesture_classifier_options=custom_options,
    )
    if running_mode == VisionTaskRunningMode.LIVE_STREAM:
        # Use the dynamic-pipeline callback when dynamic_model is configured
        dm = getattr(cfg, "dynamic_model", None)
        if isinstance(dm, dict) and (dm.get("checkpoint") or dm.get("model")):
            from engine.callback import dynamic_landmark_callback

            options_kwargs["result_callback"] = dynamic_landmark_callback
        else:
            options_kwargs["result_callback"] = result_callback
    options = vision.GestureRecognizerOptions(**options_kwargs)
    return options
