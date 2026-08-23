import base64
import json
import os
import queue
import re
import threading
import time

import cv2
import mediapipe as mp
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from mediapipe.framework.formats import landmark_pb2

from cfg import cfg, gesture_config, spot_config
from web_ui.plot import annotate_image
from utils.mediapipe_extensions import MyGestureRecognizerResult

load_dotenv()

# Shared queue for annotated images
image_queue = queue.Queue(maxsize=10)
metadata_queue = queue.Queue(maxsize=10)

# Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

import logging


class EndpointFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "/simulation_mode" not in msg and "/stream_state" not in msg and "/stats_report" not in msg

logging.getLogger("werkzeug").addFilter(EndpointFilter())

# ----------------- Log Interception -----------------
import sys
from collections import deque


class UnifiedStreamInterceptor:
    def __init__(self, max_history=2000):
        self.max_history = max_history
        self.history = deque(maxlen=max_history)
        self.subscribers = []
        self.lock = threading.Lock()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def write(self, data, stream_type='stdout'):
        if stream_type == 'stdout':
            self.original_stdout.write(data)
        else:
            self.original_stderr.write(data)
            
        with self.lock:
            self.history.append(data)
            for sub in self.subscribers:
                try:
                    sub.put_nowait(data)
                except queue.Full:
                    pass

    def flush(self, stream_type='stdout'):
        if stream_type == 'stdout':
            self.original_stdout.flush()
        else:
            self.original_stderr.flush()

    def subscribe(self):
        q = queue.Queue(maxsize=1000)
        with self.lock:
            for chunk in self.history:
                q.put_nowait(chunk)
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

class StreamProxy:
    def __init__(self, interceptor, stream_type):
        self.interceptor = interceptor
        self.stream_type = stream_type

    def write(self, data):
        self.interceptor.write(data, self.stream_type)
        return len(data)

    def flush(self):
        self.interceptor.flush(self.stream_type)

    def __getattr__(self, name):
        if self.stream_type == 'stdout':
            return getattr(self.interceptor.original_stdout, name)
        else:
            return getattr(self.interceptor.original_stderr, name)

log_interceptor = UnifiedStreamInterceptor()
sys.stdout = StreamProxy(log_interceptor, 'stdout')
sys.stderr = StreamProxy(log_interceptor, 'stderr')
# ----------------------------------------------------

# Global variables for stats
gesture_stats = {}

# Flag to signal that stats have been generated and images are ready
stats_ready = threading.Event()

# Flag to signal that the frontend requested a stop (like Ctrl+C)
stop_requested = threading.Event()

# Flag to signal that a start was requested
start_requested = threading.Event()

# Guard to prevent starting backend server thread multiple times
backend_started = threading.Event()

# Tracks whether a stream session is currently active (between /start and /stop)
session_active = threading.Event()

# Livestream stats (incremental)
livestream_stats = {
    "total_frames": 0,
    "total_hands_detected": 0,
    "total_gestures_detected": 0,
    "gesture_distribution": {},
    "avg_gesture_score": 0.0,
    "total_scores": 0.0,
    "fps_values": [],
}
livestream_stats_lock = threading.Lock()


def _select_preview_profile(queue_fill_ratio: float) -> tuple[float, int]:
    """Choose preview scale and JPEG quality from queue pressure.

    The goal is to reduce bandwidth and encode cost when the frontend
    falls behind, while keeping the stream at the same frame rate.
    """
    if queue_fill_ratio >= 0.9:
        return 0.55, 60
    if queue_fill_ratio >= 0.7:
        return 0.7, 70
    if queue_fill_ratio >= 0.4:
        return 0.85, 80
    return 1.0, 90


def _encode_preview_frame(annotated_image, queue_fill_ratio: float) -> bytes:
    """Resize and JPEG-encode a frame for the MJPEG frontend stream."""
    scale, jpeg_quality = _select_preview_profile(queue_fill_ratio)

    if scale < 1.0:
        height, width = annotated_image.shape[:2]
        resized_width = max(1, int(width * scale))
        resized_height = max(1, int(height * scale))
        preview_image = cv2.resize(
            annotated_image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview_image = annotated_image

    rgb_image = cv2.cvtColor(preview_image, cv2.COLOR_BGR2RGB)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, buffer = cv2.imencode(".jpg", rgb_image, encode_params)
    return buffer.tobytes()


def receive(result: MyGestureRecognizerResult, image: mp.Image, timestamp_ms: int = 0):
    """Receives results from the gesture recognizer and puts annotated images and metadata into the queue."""
    global livestream_stats

    # Static variables for warning logic
    if not hasattr(receive, "full_count"):
        receive.full_count = 0
        receive.suppressed = False
        receive.last_time = None

    # calculate fps
    if receive.last_time is not None:
        elapsed = time.time() - receive.last_time
        fps = 1 / elapsed if elapsed > 0 else 0
        with livestream_stats_lock:
            livestream_stats["fps_values"].append(fps)
    receive.last_time = time.time()

    result = MyGestureRecognizerResult(result)
    annotated_image = annotate_image(image.numpy_view(), result, labels=False)

    if not image_queue.full():
        # If warnings were suppressed, print recovery message
        if receive.suppressed:
            print("Frames are being collected again! The party resumes!")
            receive.full_count = 0
            receive.suppressed = False

        # Join the list of strings and clean up brackets and quotes
        cleaned_gesture_str = (
            ", ".join(result.gestures_str())
            .replace("[", "")
            .replace("]", "")
            .replace("'", "")
        )

        # Update livestream stats incrementally
        with livestream_stats_lock:
            livestream_stats["total_frames"] += 1

            # Count hands and gestures
            for gesture_list in result.gestures:
                if gesture_list:
                    livestream_stats["total_hands_detected"] += 1
                    gesture_name = gesture_list[0].category_name
                    score = gesture_list[0].score

                    # Update gesture distribution
                    if gesture_name not in livestream_stats["gesture_distribution"]:
                        livestream_stats["gesture_distribution"][gesture_name] = 0
                    livestream_stats["gesture_distribution"][gesture_name] += 1

                    # Update gesture score tracking
                    if gesture_name != "None":
                        livestream_stats["total_gestures_detected"] += 1

                    livestream_stats["total_scores"] += score

            # Calculate average gesture score
            total_detections = livestream_stats["total_hands_detected"]
            if total_detections > 0:
                livestream_stats["avg_gesture_score"] = (
                    livestream_stats["total_scores"] / total_detections
                )

        image_queue.put(annotated_image)

        # Put metadata in separate queue
        if not metadata_queue.full():
            metadata_queue.put(cleaned_gesture_str)
    else:
        if not receive.suppressed:
            if receive.full_count < 10:
                print("Warning: Image queue is full. Dropping frame.")
                receive.full_count += 1
                if receive.full_count == 10:
                    print(
                        "Seems like no one collects the frames... Is anyone out there?"
                    )
                    receive.suppressed = True


def receive_dynamic(
    image_np,
    hand_landmarks: list,
    gesture_text: str | None,
    confidence: float,
    timestamp_ms: int = 0,
    left_static: tuple[str, float] | None = None,
    right_static: tuple[str, float] | None = None,
    slider_data: dict | None = None,
    series_state: str | None = None,
):
    """Feed a frame from the dynamic (stacked) pipeline into the frontend queues.

    Parameters
    ----------
    image_np : numpy array (BGR)
        Raw camera frame.
    hand_landmarks : list of NormalizedLandmarkList
        ``result.hand_landmarks`` from MediaPipe HandLandmarker.
    gesture_text : str or None
        Predicted dynamic gesture name (from PyTorch), or *None*.
    confidence : float
        Prediction confidence (0-1).
    timestamp_ms : int
        Frame timestamp in milliseconds.
    left_static : tuple(str, float) or None
        Left hand static gesture and confidence.
    right_static : tuple(str, float) or None
        Right hand static gesture and confidence.
    handedness : list or None
        MediaPipe handedness data for stats counting.
    """
    global livestream_stats

    # Static variables for warning logic (shared naming avoids extra attrs)
    if not hasattr(receive_dynamic, "full_count"):
        receive_dynamic.full_count = 0
        receive_dynamic.suppressed = False
        receive_dynamic.last_time = None

    # calculate fps
    if receive_dynamic.last_time is not None:
        elapsed = time.time() - receive_dynamic.last_time
        fps = 1 / elapsed if elapsed > 0 else 0
        with livestream_stats_lock:
            livestream_stats["fps_values"].append(fps)
    receive_dynamic.last_time = time.time()

    # ---------- Annotate ----------
    annotated = image_np.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_drawing_styles = mp.solutions.drawing_styles

    for lm_list in hand_landmarks:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend(
            [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in lm_list]
        )
        mp_drawing.draw_landmarks(
            annotated,
            proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    # Slider data is now sent to frontend via metadata stream

    # ---------- Enqueue ----------
    # Handle "None" class from models
    if left_static and left_static[0] == "None":
        left_static = None
    if right_static and right_static[0] == "None":
        right_static = None
    if gesture_text == "None":
        gesture_text = None

    # Filter gestures if show_unmapped is False
    show_unmapped = getattr(cfg, "show_unmapped_gestures", False)
    left_mapping = gesture_config.gesture_mappings.get("left", {})
    right_mapping = gesture_config.gesture_mappings.get("right", {})
    dynamic_mapping = gesture_config.gesture_mappings.get("dynamic", {})

    # Gestures used in combinations should not be dropped
    used_in_combinations = set()
    for combo_key in gesture_config.gesture_mappings.get("combinations", {}).keys():
        try:
            l_gest, r_gest = combo_key.split("+")
            used_in_combinations.add(l_gest)
            used_in_combinations.add(r_gest)
        except ValueError:
            pass

    # Gestures used in series should not be dropped
    for series_key in gesture_config.gesture_mappings.get("series", {}).keys():
        try:
            parts = series_key.split(",")
            for p in parts:
                if ":" in p:
                    hand, gest = p.split(":")
                    used_in_combinations.add(gest)
        except ValueError:
            pass

    if (
        left_static
        and not show_unmapped
        and not left_mapping.get(left_static[0])
        and left_static[0] not in used_in_combinations
    ):
        left_static = None
    if (
        right_static
        and not show_unmapped
        and not right_mapping.get(right_static[0])
        and right_static[0] not in used_in_combinations
    ):
        right_static = None
    if gesture_text and not show_unmapped and not dynamic_mapping.get(gesture_text):
        gesture_text = None

    # Format the gestures with their confidence scores so the frontend can display them cleanly
    gesture_str = f"{gesture_text} ({confidence:.0%})" if gesture_text else ""
    left_static_str = (
        f"{left_static[0].replace('_', ' ')} ({left_static[1]:.0%})"
        if left_static
        else ""
    )
    right_static_str = (
        f"{right_static[0].replace('_', ' ')} ({right_static[1]:.0%})"
        if right_static
        else ""
    )

    if not image_queue.full():
        if receive_dynamic.suppressed:
            print("Frames are being collected again! The party resumes!")
            receive_dynamic.full_count = 0
            receive_dynamic.suppressed = False

        with livestream_stats_lock:
            livestream_stats["total_frames"] += 1
            if hand_landmarks:
                livestream_stats["total_hands_detected"] += len(hand_landmarks)
            if gesture_text:
                livestream_stats["total_gestures_detected"] += 1
                if gesture_text not in livestream_stats["gesture_distribution"]:
                    livestream_stats["gesture_distribution"][gesture_text] = 0
                livestream_stats["gesture_distribution"][gesture_text] += 1
                livestream_stats["total_scores"] += confidence
                total = livestream_stats["total_hands_detected"]
                if total > 0:
                    livestream_stats["avg_gesture_score"] = (
                        livestream_stats["total_scores"] / total
                    )

        image_queue.put(annotated)
        if not metadata_queue.full():
            metadata_queue.put(
                (gesture_str, left_static_str, right_static_str, series_state, slider_data)
            )
    else:
        if not receive_dynamic.suppressed:
            if receive_dynamic.full_count < 10:
                print("Warning: Image queue is full. Dropping frame.")
                receive_dynamic.full_count += 1
                if receive_dynamic.full_count == 10:
                    print(
                        "Seems like no one collects the frames... Is anyone out there?"
                    )
                    receive_dynamic.suppressed = True


# generate_frames() streams images to the frontend
def generate_frames():
    """Yields frames from the shared queue. Drains queue to always serve the latest frame."""
    while not stats_ready.is_set():
        try:
            queue_fill_ratio = (
                image_queue.qsize() / image_queue.maxsize
                if image_queue.maxsize > 0
                else 0.0
            )
            # Get the first available frame
            annotated_image = image_queue.get(timeout=1)
            # Drain the queue to skip stale frames and get the latest one
            while not image_queue.empty():
                try:
                    annotated_image = image_queue.get_nowait()
                except queue.Empty:
                    break
            frame = _encode_preview_frame(annotated_image, queue_fill_ratio)
            # Yield the frame as part of the MJPEG stream
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        except queue.Empty:
            continue


# generate_metadata() streams metadata to the frontend
def generate_metadata():
    """Yields metadata from the shared queue. Drains queue to always serve the latest."""
    while not stats_ready.is_set():
        try:
            # Get the first available metadata
            metadata_tuple = metadata_queue.get(timeout=1)
            # Drain the queue to skip stale metadata and get the latest one
            while not metadata_queue.empty():
                try:
                    metadata_tuple = metadata_queue.get_nowait()
                except queue.Empty:
                    break

            if isinstance(metadata_tuple, tuple) and len(metadata_tuple) >= 3:
                gestures_str = metadata_tuple[0]
                left_static_str = metadata_tuple[1]
                right_static_str = metadata_tuple[2]
                series_state_dict = metadata_tuple[3] if len(metadata_tuple) > 3 else None
                slider_data_dict = metadata_tuple[4] if len(metadata_tuple) > 4 else None
            else:
                gestures_str = metadata_tuple
                left_static_str = ""
                right_static_str = ""
                series_state_dict = None
                slider_data_dict = None
                
            slider_json = json.dumps(slider_data_dict) if slider_data_dict else "{}"
            series_json = json.dumps(series_state_dict) if series_state_dict else "{}"

            # Get current stats
            with livestream_stats_lock:
                stats_copy = livestream_stats.copy()

            # Calculate average hands per frame
            avg_hands_per_frame = (
                stats_copy["total_hands_detected"] / stats_copy["total_frames"]
                if stats_copy["total_frames"] > 0
                else 0
            )

            # Calculate current FPS from latest value
            fps_vals = stats_copy.get("fps_values", [])
            fps = fps_vals[-1] if fps_vals else 0
            # Current detection and frame-load times
            detection_vals = stats_copy.get("detection_ms_values", [])
            current_detection = detection_vals[-1] if detection_vals else 0
            frame_load_vals = stats_copy.get("frame_load_ms_values", [])
            current_frame_load = frame_load_vals[-1] if frame_load_vals else 0

            # Create stats JSON
            stats_json = json.dumps(
                {
                    "total_frames": stats_copy["total_frames"],
                    "total_hands_detected": stats_copy["total_hands_detected"],
                    "total_gestures_detected": stats_copy["total_gestures_detected"],
                    "avg_hands_per_frame": round(avg_hands_per_frame, 2),
                    "avg_gesture_score": round(stats_copy["avg_gesture_score"], 2),
                    "gesture_distribution": stats_copy["gesture_distribution"],
                }
            )

            # Get current active action details from gesture_mapper (avoiding circular imports)
            from engine.gesture_action_mapping import gesture_mapper

            active_action = None
            with gesture_mapper._lock:
                if gesture_mapper._active_action:
                    start_time = gesture_mapper._active_action["start_time"]
                    duration = gesture_mapper._active_action["duration"]
                    elapsed = time.time() - start_time
                    remaining = max(0.0, duration - elapsed)
                    active_action = {
                        "action_key": gesture_mapper._active_action["action_key"],
                        "display_name": gesture_mapper._active_action["display_name"],
                        "description": gesture_mapper._active_action["description"],
                        "duration": duration,
                        "elapsed": round(elapsed, 2),
                        "remaining": round(remaining, 2),
                    }
            active_action_json = json.dumps(active_action)

            # Yield the metadata as part of the stream
            yield (
                b"--metadata\r\n"
                b"Content-Type: application/json\r\n"
                + f"Gesture-String: {gestures_str}\r\n".encode()
                + f"Left-Static: {left_static_str}\r\n".encode()
                + f"Right-Static: {right_static_str}\r\n".encode()
                + f"Series-Data: {series_json}\r\n".encode()
                + f"Active-Action: {active_action_json}\r\n".encode()
                + f"FPS: {fps:.0f}\r\n".encode()
                + f"FrameLoad: {current_frame_load:.0f}\r\n".encode()
                + f"Detection: {current_detection:.0f}\r\n".encode()
                + f"Stats: {stats_json}\r\n".encode()
                + f"Slider-Data: {slider_json}\r\n".encode()
                + b"\r\n"
            )
        except queue.Empty:
            continue


# Flask route for video feed
@app.route("/video_feed")
def video_feed():
    """Endpoint for the video stream."""
    response = Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["X-Accel-Buffering"] = "no"
    return response


# Flask route for metadata stream
@app.route("/metadata_feed")
def metadata_feed():
    """Endpoint for the metadata stream."""
    response = Response(
        generate_metadata(), mimetype="multipart/x-mixed-replace; boundary=metadata"
    )
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/logs")
def logs_page():
    """Frontend page to view live process logs."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Process Logs</title>
        <style>
            body { background: #1e1e1e; color: #d4d4d4; font-family: monospace; padding: 10px; margin: 0; font-size: 14px; }
            #logs { white-space: pre-wrap; word-wrap: break-word; }
        </style>
    </head>
    <body>
        <div id="logs"></div>
        <script>
            const logsDiv = document.getElementById('logs');
            const eventSource = new EventSource('/logs/stream');
            
            eventSource.onmessage = function(event) {
                try {
                    const text = JSON.parse(event.data);
                    logsDiv.appendChild(document.createTextNode(text));
                    window.scrollTo(0, document.body.scrollHeight);
                } catch(e) {
                    logsDiv.appendChild(document.createTextNode(event.data));
                }
            };
            
            eventSource.onerror = function(e) {
                console.error("Stream closed or error:", e);
            };
        </script>
    </body>
    </html>
    """
    return html

@app.route("/logs/stream")
def stream_logs():
    """Chunked streaming endpoint for process logs."""
    def generate():
        q = log_interceptor.subscribe()
        try:
            while True:
                try:
                    chunk = q.get(timeout=15)
                    if isinstance(chunk, bytes):
                        chunk = chunk.decode("utf-8", errors="replace")
                    yield f"data: {json.dumps(chunk)}\n\n"
                except queue.Empty:
                    yield ": ping\n\n"
        except GeneratorExit:
            pass
        finally:
            log_interceptor.unsubscribe(q)

    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Connection"] = "keep-alive"
    response.headers["X-Accel-Buffering"] = "no"
    return response


# The host the Flask backend actually bound to (set by start_backend)
active_host: str = "0.0.0.0"
active_port: int = 5000


# Start the Flask app in a separate thread
def start_backend():
    """Starts the Flask backend in a separate thread.

    Tries each comma-separated IP in ``FLASK_HOST`` and falls back to
    ``127.0.0.1`` if none of them can be bound.
    """
    import socket

    if backend_started.is_set():
        return

    global active_host, active_port

    raw = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5000))
    active_port = port

    candidates = [h.strip() for h in raw.split(",") if h.strip()]
    if "0.0.0.0" not in candidates:
        candidates.append("0.0.0.0")  # fall back to all interfaces

    chosen = "0.0.0.0"
    for host in candidates:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind((host, port))
            sock.close()
            chosen = host
            break
        except OSError:
            print(f"[Backend] Cannot bind to {host}:{port}, trying next...")

    active_host = chosen
    print(f"[Backend] Starting Flask on {chosen}:{port}")
    threading.Thread(
        target=lambda: app.run(host=chosen, port=port), daemon=True
    ).start()
    backend_started.set()


@app.route("/estop", methods=["POST"])
def estop():
    """Assert Spot's emergency stop."""
    from actions import spot

    spot.estop()
    return jsonify({"status": "ok"})


@app.route("/manual_control", methods=["POST"])
def manual_control():
    """Trigger manual actions using WASD and QE."""
    from actions import spot

    data = request.json
    action = data.get("action")

    if action == "w":
        spot.walk_forward()
    elif action == "s":
        spot.walk_backward()
    elif action == "a":
        spot.strafe_left()
    elif action == "d":
        spot.strafe_right()
    elif action == "q":
        spot.turn_left()
    elif action == "e":
        spot.turn_right()
    elif action == " ":
        spot.stop()
    else:
        return jsonify({"status": "error", "message": "Unknown action"}), 400

    return jsonify({"status": "ok"})


@app.route("/stop", methods=["POST"])
def stop():
    """Endpoint to stop the stream (same as Ctrl+C)."""
    session_active.clear()
    stop_requested.set()
    return Response(status=200)


@app.route("/start", methods=["POST"])
def start():
    """Endpoint to start a fresh stream session.

    Works for first start and subsequent starts.
    """
    reset_state()
    session_active.set()
    start_requested.set()
    return Response(status=200)


def reset_state():
    """Resets all shared state for a fresh run."""
    global livestream_stats
    stats_ready.clear()
    stop_requested.clear()
    start_requested.clear()
    session_active.clear()
    # Drain queues
    for q in (image_queue, metadata_queue):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break
    # Reset livestream stats
    with livestream_stats_lock:
        livestream_stats = {
            "total_frames": 0,
            "total_hands_detected": 0,
            "total_gestures_detected": 0,
            "gesture_distribution": {},
            "avg_gesture_score": 0.0,
            "total_scores": 0.0,
            "fps_values": [],
        }
    # Reset receive() static vars
    if hasattr(receive, "full_count"):
        receive.full_count = 0
        receive.suppressed = False
        receive.last_time = None


def get_stream_state() -> str:
    """Return high-level stream state for frontend UI orchestration.

    States:
    - idle: no active session
    - starting: start requested, waiting for first frames
    - stopping: stop requested, waiting for stats generation
    - running: active stream with frame flow
    - ready: stats are generated and available
    """
    # Check if we were actively running but lost the lease (e.g., tablet hijack)
    if session_active.is_set() and not stop_requested.is_set():
        try:
            from actions import spot as spot_module
            if hasattr(spot_module, "_controller") and spot_module._controller.is_connected and spot_module._controller.has_control is False:
                print("[Backend] Lease hijacked (tablet taking over?). Forcing stream stop.")
                stop_requested.set()
        except Exception as e:
            print(f"[Backend] Error checking lease: {e}")

    if stats_ready.is_set():
        return "ready"

    if stop_requested.is_set():
        return "stopping"

    if start_requested.is_set():
        return "starting"

    with livestream_stats_lock:
        has_frames = livestream_stats.get("total_frames", 0) > 0

    if has_frames:
        return "running"

    if session_active.is_set():
        return "starting"

    return "idle"


@app.route("/stream_state")
def stream_state():
    return jsonify({"state": get_stream_state()})


@app.route("/stats_report")
def stats_report():
    """Returns a JSON object with base64-encoded stats images, or 204 if not ready yet."""
    if not stats_ready.is_set():
        return Response(status=204)
    images = []
    for filename in sorted(os.listdir(cfg.output)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(cfg.output, filename)
            with open(filepath, "rb") as f:
                data = base64.b64encode(f.read()).decode("utf-8")
            ext = os.path.splitext(filename)[1].lower().lstrip(".")
            mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
            images.append(
                {
                    "filename": filename,
                    "data": f"data:{mime};base64,{data}",
                }
            )
    return jsonify(images)


@app.route("/spot_config", methods=["GET"])
def get_spot_config():
    return jsonify(
        {
            "VELOCITY_BASE_SPEED": spot_config.VELOCITY_BASE_SPEED,
            "VELOCITY_BASE_ANGULAR": spot_config.VELOCITY_BASE_ANGULAR,
            "VELOCITY_CMD_DURATION": spot_config.VELOCITY_CMD_DURATION,
            "BODY_PITCH_ANGLE": spot_config.BODY_PITCH_ANGLE,
            "BODY_YAW_ANGLE": spot_config.BODY_YAW_ANGLE,
        }
    )


@app.route("/spot_config", methods=["POST"])
def update_spot_config():
    from cfg.config_manager import save_overrides
    data = request.json
    if "VELOCITY_BASE_SPEED" in data:
        spot_config.VELOCITY_BASE_SPEED = float(data["VELOCITY_BASE_SPEED"])
    if "VELOCITY_BASE_ANGULAR" in data:
        spot_config.VELOCITY_BASE_ANGULAR = float(data["VELOCITY_BASE_ANGULAR"])
    if "VELOCITY_CMD_DURATION" in data:
        spot_config.VELOCITY_CMD_DURATION = float(data["VELOCITY_CMD_DURATION"])
    if "BODY_PITCH_ANGLE" in data:
        spot_config.BODY_PITCH_ANGLE = float(data["BODY_PITCH_ANGLE"])
    if "BODY_YAW_ANGLE" in data:
        spot_config.BODY_YAW_ANGLE = float(data["BODY_YAW_ANGLE"])
    
    save_overrides({"spot": data})
    return Response(status=200)


@app.route("/has_control", methods=["GET"])
def get_has_control():
    """Get the true connection and lease status."""
    from actions import spot as spot_module
    has_control = False
    debug_info = {}
    if hasattr(spot_module, "_controller"):
        ctrl = spot_module._controller
        debug_info = {
            "sim_mode": ctrl._simulation_mode,
            "connected": ctrl.is_connected,
            "lease": ctrl.has_control
        }
        if not ctrl._simulation_mode:
            if ctrl.is_connected and ctrl.has_control:
                has_control = True
            else:
                # We lost connection or control while in live mode
                ctrl._simulation_mode = True
    
    print(f"[DEBUG] GET /has_control => {has_control} | State: {debug_info}")
    return jsonify({"has_control": has_control})


@app.route("/has_control", methods=["POST"])
def update_has_control():
    """Toggle or set the live mode and perform live connect/disconnect."""
    import json
    import queue
    import threading

    from flask import stream_with_context

    from actions import spot as spot_module

    data = request.json
    if "has_control" not in data:
        return Response(status=400)
        
    want_control = bool(data["has_control"])

    current_control = False
    if hasattr(spot_module, "_controller"):
        ctrl = spot_module._controller
        if not ctrl._simulation_mode:
            if ctrl.is_connected and ctrl.has_control:
                current_control = True
            else:
                ctrl._simulation_mode = True

    if want_control == current_control:
        return jsonify({
            "has_control": current_control,
        })

    def generate():
        q = queue.Queue()
        
        def log_cb(msg):
            q.put({"log": msg})
            
        def worker():
            try:
                if not want_control:
                    log_cb("[Backend] Disconnecting from robot...")
                    spot_module.disconnect(log_callback=log_cb)
                else:
                    log_cb("[Backend] Attempting to connect to real robot...")
                    spot_module.connect(exit_on_fail=False, log_callback=log_cb)
                
                # Determine true state
                has_control_result = False
                ctrl = spot_module._controller
                
                log_cb(f"[DEBUG] Post-connect State -> sim: {ctrl._simulation_mode}, connected: {ctrl.is_connected}, lease: {ctrl.has_control}")
                
                if not ctrl._simulation_mode and ctrl.is_connected and ctrl.has_control:
                    has_control_result = True
                elif not ctrl._simulation_mode:
                    ctrl._simulation_mode = True
                
                log_cb(f"[DEBUG] Final has_control result => {has_control_result}")
                
                # Signal success
                q.put({
                    "has_control": has_control_result,
                })
            except Exception as e:
                # Force fallback to simulation
                spot_module._controller._simulation_mode = True
                q.put({
                    "error": str(e),
                    "has_control": False
                })
            finally:
                q.put(None) # EOF marker
                
        threading.Thread(target=worker, daemon=True).start()
        
        while True:
            item = q.get()
            if item is None:
                break
            yield json.dumps(item) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


@app.route("/gesture_config", methods=["GET"])
def get_gesture_config():
    import inspect

    from actions import spot

    # Dynamically fetch public action functions from the spot module
    ignored_functions = {"connect", "disconnect", "estop", "main", "load_dotenv"}
    spot_actions = [
        f"spot.{name}"
        for name, obj in inspect.getmembers(spot, inspect.isfunction)
        if not name.startswith("_")
        and name not in ignored_functions
        and obj.__module__ == "actions.spot"
    ]

    # Add configurable spot parameters
    spot_actions.extend(
        [
            "config.VELOCITY_BASE_SPEED",
            "config.VELOCITY_BASE_ANGULAR",
            "config.VELOCITY_CMD_DURATION",
            "config.BODY_PITCH_ANGLE",
            "config.BODY_YAW_ANGLE",
            "config.BODY_HEIGHT_OFFSET",
        ]
    )

    return jsonify(
        {
            "mappings": gesture_config.gesture_mappings["dynamic"],
            "actions": spot_actions,
            "confidence_threshold": cfg.dynamic_model.get("confidence_threshold", 0.5),
            "show_unmapped": getattr(cfg, "show_unmapped_gestures", False),
        }
    )


@app.route("/gesture_config", methods=["POST"])
def update_gesture_config():
    from cfg.config_manager import save_overrides
    data = request.json
    
    overrides_gesture = {}
    overrides_cfg = {}
    
    if "mappings" in data:
        gesture_config.gesture_mappings["dynamic"].update(data["mappings"])
        for k, v in gesture_config.gesture_mappings["dynamic"].items():
            if not v or v.lower() == "none":
                gesture_config.gesture_mappings["dynamic"][k] = None
        overrides_gesture["dynamic"] = gesture_config.gesture_mappings["dynamic"]
        
    if "confidence_threshold" in data:
        cfg.dynamic_model["confidence_threshold"] = float(data["confidence_threshold"])
        overrides_cfg["dynamic_model"] = {"confidence_threshold": cfg.dynamic_model["confidence_threshold"]}
        
    if "show_unmapped" in data:
        cfg.show_unmapped_gestures = bool(data["show_unmapped"])
        overrides_cfg["show_unmapped_gestures"] = bool(data["show_unmapped"])
        
    if overrides_gesture:
        save_overrides({"gesture": overrides_gesture})
    if overrides_cfg:
        save_overrides({"cfg": overrides_cfg})
        
    return Response(status=200)


@app.route("/static_gesture_config", methods=["GET"])
def get_static_gesture_config():
    """Get static gesture configuration and mappings."""
    import inspect

    from actions import spot

    # Dynamically fetch public action functions
    ignored_functions = {"connect", "disconnect", "estop", "main", "load_dotenv"}
    spot_actions = [
        f"spot.{name}"
        for name, obj in inspect.getmembers(spot, inspect.isfunction)
        if not name.startswith("_")
        and name not in ignored_functions
        and obj.__module__ == "actions.spot"
    ]

    static_cfg = getattr(cfg, "static_gestures", {})

    return jsonify(
        {
            "enabled": static_cfg.get("enabled", True),
            "confidence_threshold": static_cfg.get("confidence_threshold", 0.6),
            "mappings_left": gesture_config.gesture_mappings.get("left", {}),
            "mappings_right": gesture_config.gesture_mappings.get("right", {}),
            "actions": spot_actions,
        }
    )


@app.route("/static_gesture_config", methods=["POST"])
def update_static_gesture_config():
    """Update static gesture configuration and mappings."""
    from cfg.config_manager import save_overrides
    data = request.json

    static_cfg = getattr(cfg, "static_gestures", {})
    if "enabled" in data:
        static_cfg["enabled"] = bool(data["enabled"])
    if "confidence_threshold" in data:
        static_cfg["confidence_threshold"] = float(data["confidence_threshold"])

    if "static_gestures" not in dir(cfg):
        cfg.static_gestures = static_cfg

    overrides_gesture = {}

    if "mappings_left" in data:
        gesture_config.gesture_mappings.setdefault("left", {}).update(
            data["mappings_left"]
        )
        for k, v in gesture_config.gesture_mappings["left"].items():
            if not v or v.lower() == "none":
                gesture_config.gesture_mappings["left"][k] = None
        overrides_gesture["left"] = gesture_config.gesture_mappings["left"]

    if "mappings_right" in data:
        gesture_config.gesture_mappings.setdefault("right", {}).update(
            data["mappings_right"]
        )
        for k, v in gesture_config.gesture_mappings["right"].items():
            if not v or v.lower() == "none":
                gesture_config.gesture_mappings["right"][k] = None
        overrides_gesture["right"] = gesture_config.gesture_mappings["right"]

    save_overrides({"cfg": {"static_gestures": static_cfg}})
    if overrides_gesture:
        save_overrides({"gesture": overrides_gesture})

    return Response(status=200)


@app.route("/two_hand_config", methods=["GET"])
def get_two_hand_config():
    """Get two-hand combination mappings and settings."""
    import inspect

    from actions import spot

    # Dynamically fetch public action functions
    ignored_functions = {"connect", "disconnect", "estop", "main", "load_dotenv"}
    spot_actions = [
        f"spot.{name}"
        for name, obj in inspect.getmembers(spot, inspect.isfunction)
        if not name.startswith("_")
        and name not in ignored_functions
        and obj.__module__ == "actions.spot"
    ]

    # Add configurable spot parameters
    spot_actions.extend(
        [
            "config.VELOCITY_BASE_SPEED",
            "config.VELOCITY_BASE_ANGULAR",
            "config.VELOCITY_CMD_DURATION",
            "config.BODY_PITCH_ANGLE",
            "config.BODY_YAW_ANGLE",
            "config.BODY_HEIGHT_OFFSET",
        ]
    )

    return jsonify(
        {
            "two_hand_wait_ms": cfg.dynamic_model.get("two_hand_wait_ms", 300),
            "series_timeout_ms": cfg.dynamic_model.get("series_timeout_ms", 2000),
            "combinations": gesture_config.gesture_mappings.get("combinations", {}),
            "series": gesture_config.gesture_mappings.get("series", {}),
            "actions": spot_actions,
        }
    )


@app.route("/two_hand_config", methods=["POST"])
def update_two_hand_config():
    """Update two-hand combination mappings and settings."""
    from cfg.config_manager import save_overrides
    data = request.json

    overrides_cfg = {}
    overrides_gesture = {}

    if "two_hand_wait_ms" in data:
        cfg.dynamic_model["two_hand_wait_ms"] = int(data["two_hand_wait_ms"])
        overrides_cfg["two_hand_wait_ms"] = cfg.dynamic_model["two_hand_wait_ms"]

    if "series_timeout_ms" in data:
        cfg.dynamic_model["series_timeout_ms"] = int(data["series_timeout_ms"])
        overrides_cfg["series_timeout_ms"] = cfg.dynamic_model["series_timeout_ms"]

    if "combinations" in data:
        gesture_config.gesture_mappings["combinations"] = data["combinations"]
        overrides_gesture["combinations"] = data["combinations"]

    if "series" in data:
        gesture_config.gesture_mappings["series"] = data["series"]
        overrides_gesture["series"] = data["series"]

    if overrides_cfg:
        save_overrides({"cfg": {"dynamic_model": overrides_cfg}})
    if overrides_gesture:
        save_overrides({"gesture": overrides_gesture})

    return Response(status=200)


# ── Example gesture videos ───────────────────────────────────────────────
_EXAMPLE_VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "example_videos")


@app.route("/example_video/<class_name>", methods=["GET"])
def get_example_video(class_name):
    """Serve a pre-composed example video for a given gesture class."""
    # Sanitize: lowercase, replace non-alphanumeric with underscores
    sanitized = re.sub(r"[^a-z0-9]+", "_", class_name.lower()).strip("_")

    # Try .mp4 first, then .webm
    for ext, mime in [("mp4", "video/mp4"), ("webm", "video/webm")]:
        filename = f"{sanitized}.{ext}"
        filepath = os.path.join(_EXAMPLE_VIDEOS_DIR, filename)
        if os.path.isfile(filepath):
            return send_from_directory(_EXAMPLE_VIDEOS_DIR, filename, mimetype=mime)

    return Response(f"No example video for '{class_name}'", status=404)
