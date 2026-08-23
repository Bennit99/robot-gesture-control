import threading

import mediapipe as mp
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizerResult
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult

from engine.gesture_action_mapping import gesture_mapper
from engine.processor import GestureFilter
from web_ui.streaming_web_server.backend import receive
from utils.mediapipe_extensions import MyGestureRecognizerResult, MyHandLandmarkerResult

gesture_filter_left = GestureFilter()
gesture_filter_right = GestureFilter()

# Event to signal that MediaPipe has finished processing the last frame.
# Starts set (ready) so the first frame is always sent.
recognizer_ready = threading.Event()
recognizer_ready.set()

# Storage for landmark results collected during a session
landmark_results = []
landmark_results_lock = threading.Lock()


def result_callback(
    result: GestureRecognizerResult, image: mp.Image, timestamp_ms: int
):
    """Callback function to process gesture recognition results."""
    result = MyGestureRecognizerResult(result)

    # Update gesture filter with new result
    gesture_filter_left.add_frame(result.left_gesture())
    gesture_filter_right.add_frame(result.right_gesture())

    # Get stable gesture from filter
    stable_left = gesture_filter_left.get_stable_gesture()
    stable_right = gesture_filter_right.get_stable_gesture()

    # Process stable gestures through action mapper
    if stable_left or stable_right:
        gesture_mapper.execute(stable_left, stable_right)
        # Reset filters after action executed
        if stable_left:
            gesture_filter_left.reset()
        if stable_right:
            gesture_filter_right.reset()

    receive(result, image, timestamp_ms)  # pass to backend receive function

    # Signal that the recognizer is ready for the next frame
    recognizer_ready.set()


def landmark_result_callback(
    result: HandLandmarkerResult, image: mp.Image, timestamp_ms: int
):
    """Callback function to collect hand landmark results (LIVE_STREAM mode)."""
    result = MyHandLandmarkerResult(result)
    with landmark_results_lock:
        landmark_results.append((result, timestamp_ms))

    if result.hand_landmarks:
        print(f"[{timestamp_ms}] " + ", ".join(result.summary_str()))

    # Signal that the recognizer is ready for the next frame
    recognizer_ready.set()


def dynamic_landmark_callback(
    result: HandLandmarkerResult, image: mp.Image, timestamp_ms: int
):
    """Callback that feeds landmarks into the stacked-pipeline ring buffer.

    Used when ``cfg.dynamic_model`` is configured.  Pushes every frame
    (including zero-vectors for empty detections) into the shared
    :class:`~engine.dynamic_recognizer.LandmarkBuffer`.
    """
    from engine.dynamic_recognizer import get_landmark_buffer

    buf = get_landmark_buffer()
    buf.push_from_mediapipe(
        hand_landmarks=result.hand_landmarks,
        handedness=result.handedness,
    )
    if hasattr(result, "gestures"):
        buf.update_static_gestures(result.gestures, result.handedness)
    # Signal ready for next frame
    recognizer_ready.set()
