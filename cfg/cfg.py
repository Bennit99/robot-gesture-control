verbose = False  # Show additional output
debug_pipeline_prints = True  # Print granular pipeline decision trace
model = "models/gesture_recognizer.task"  # Path to gesture recognizer model
data = "./data/base/"  # Path to input data (images or videos)
output = "./output/"  # Path to save output

# MediaPipe GestureRecognizer configuration ----------------------------------------------------------------------
running_mode = "LIVE_STREAM"  # 'IMAGE', 'VIDEO', or 'LIVE_STREAM'
num_hands = 2  # Max hands detected (int > 0)
min_hand_detection_confidence = 0.5  # Min confidence for hand detection
min_hand_presence_confidence = 0.6  # Min confidence for hand presence
min_tracking_confidence = 0.5  # Min confidence for hand tracking

# dynamic gesture model (stacked pipeline: MediaPipe landmarks → PyTorch classifier) --------------------------------
# Uncomment and set model path to enable the stacked pipeline.
# When set, LIVE_STREAM mode will use MediaPipe for landmark extraction
# and feed a sliding window of landmarks into this PyTorch model for
# dynamic gesture classification.
dynamic_model = {
    "model": "./models/gru-12_ml9.onnx",  # trained model (.ckpt, .pt, or .onnx)
    "labels_csv": "./models/dynamic_classes.csv",  # class label file
    "seq_len": 40,  # must match training seq_len
    "max_hands": 1,  # must match training max_hands (for dynamic model input)
    "min_valid_detections": 9,  # skip inference if fewer non-empty frames in buffer
    "interpolate_missing": True,  # linearly fill short missing-landmark gaps
    "interpolate_max_gap": 4,  # max consecutive missing frames to interpolate
    "device": "auto",  # 'auto', 'cuda', or 'cpu'
    "confidence_threshold": 0.7,  # min softmax confidence to accept
    "show_fps": True,  # print FPS counter
    "two_hand_wait_ms": 300,  # wait for second hand before firing single-hand action
    "series_timeout_ms": 4000,
}

# Static gesture recognition (MediaPipe per-frame, fallback when no dynamic gesture)
static_gestures = {
    "enabled": True,
    "model": "./models/gesture_recognizer.task",  # MediaPipe GestureRecognizer
    "confidence_threshold": 0.55,
    "show_unmapped": False,  # If False, unmapped static gestures will not be sent to the frontend UI
}

show_unmapped_gestures = False

# gesture engine settings ---------------------------------------------------------------------------------------------
filter = {"window_size": 7, "threshold": 0.7, "hysteresis_frames": 2}

# visualization --------------------------------------------------------------------------------------------------
# IMAGE mode
batch_size = 6  # Number of images to plot in a batch

# VIDEO mode
video_fps = 10  # Frames per second for video plotting

# LIVE_STREAM mode
stream_fps = 15  # max frames per second for live stream
# If True: start only web stack and wait for /start endpoint before running gesture pipeline.
# If False: start gesture pipeline immediately (web stack still starts in LIVE_STREAM).
livestream_web_only = True
# stream_exposure = -7  # Exposure value (2^n seconds, e.g. -8 = 1/256s ≈ 3.9ms)
# stream_brightness = 200  # ISO/Gain to compensate low exposure

from cfg.config_manager import load_overrides

overrides = load_overrides().get("cfg", {})
if overrides:
    if "dynamic_model" in overrides:
        dynamic_model.update(overrides["dynamic_model"])
    if "static_gestures" in overrides:
        static_gestures.update(overrides["static_gestures"])
    if "show_unmapped_gestures" in overrides:
        show_unmapped_gestures = overrides["show_unmapped_gestures"]
