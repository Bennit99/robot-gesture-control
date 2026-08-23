# Contextual Architecture Document: Quadruped Robot Gesture Control

This document provides a comprehensive technical breakdown of the gesture recognition architecture deployed on the Jetson Orin Nano / Boston Dynamics Spot hardware. It explains the end-to-end design decisions, software stack, networking, and machine learning components required for deploying a continuous, low-latency, dynamic gesture recognition system on Edge AI hardware.

---

## 1. System Overview
The gesture control pipeline maps a real-time camera feed to physical robotic movements while adhering to strict edge-compute constraints. The overall pipeline operates as follows:
* **Video Ingestion:** Local camera or IP stream ingestion routed via Docker to the Python backend.
* **Spatial Feature Extraction:** MediaPipe isolates the hands and converts raw RGB pixels into sparse 3D skeletal landmarks.
* **Temporal Classification:** A custom PyTorch Recurrent Neural Network (GRU) analyzes the rolling window of skeletal data to classify dynamic gestures over time.
* **Hardware Translation:** The identified gestures are parsed by the bosdyn SDK and translated into physical `RobotCommand` velocity commands or trajectory actions for the Spot robot.
* **User Interface:** A local Flask web server provides real-time MJPEG streams of the internal MediaPipe skeletal tracking alongside a dynamic settings dashboard.

---

## 2. User Interface & Networking Architecture
The user interface serves as the command center for monitoring the AI's "brain" and tuning configuration flags on the fly.

### Tech Stack
* **Backend:** Flask (Python) operates as the primary REST API and MJPEG stream provider.
* **Frontend:** Vanilla HTML/JavaScript utilizing Tailwind CSS for responsive styling.
* **Offline Capability:** To ensure deployment robustness in edge environments (e.g., deep outdoors alongside the Spot robot without internet access), Tailwind CSS is compiled locally. No external CDNs are required, meaning the entire UI is functional on an isolated Local Area Network (LAN).

### Network Architecture
The Flask server runs directly on the compute node (Jetson Orin Nano). Developers or operators connect to the UI via the node's local IP address using any standard web browser. The backend pushes raw MediaPipe-annotated frames over HTTP via an MJPEG (Motion JPEG) socket, allowing low-latency visual debugging of the gesture recognizer.

### Functionality & Options
The Web UI permits dynamic interaction with the running python scripts:
* Real-time video visualization of the MediaPipe hand landmarks and bounding boxes.
* Real-time updating of gesture-to-action bindings.

### Dynamic Resolution Throttling
To maintain a real-time experience even under poor network conditions, the backend implements a dynamic resolution throttle. It monitors the internal `queue_fill_ratio` of the MJPEG stream buffer. If the client's connection speed drops and the queue begins filling up with unsent frames, the system automatically scales down the resolution (down to 55%) and decreases JPEG quality compression (down to 60) to aggressively reduce bandwidth. This ensures the frame rate remains stable and latency remains low without dropping the connection.

---

## 3. Docker Deployment & Platform Abstraction
Deploying Python hardware pipelines directly on edge devices can lead to severe dependency conflicts, especially when mixing CUDA, hardware-accelerated OpenCV, and standard PyTorch. The system utilizes Docker to ensure reproducible environments.

### The start.sh Orchestrator
To handle the discrepancy between developing on a Windows Machine and deploying on a Linux Jetson, the `start.sh` bash script orchestrates the Docker deployment by automatically selecting hardware presets:
* **Jetson Profile (arm64):** Automatically links to `docker-compose.jetson.yml`. This injects Linux-specific GPU bindings (`/dev/video0` for cameras, native TensorRT access, etc.).
* **Windows Profile (amd64):** Automatically links to `docker-compose.windows.yml`. Because Docker Desktop for Windows cannot pass through USB webcams natively to the container, the script silently spins up an ffmpeg UDP bridge on the host (`udp://localhost:1234`) and pipes the local camera feed directly into the Docker container.

---

## 4. Dependency Management: uv Package Manager
Given the strict PyTorch and CUDA versioning required across arm64 and amd64, the standard `pip` installer was replaced with the `uv` package manager.

### Why UV?
* **Speed:** Dependency resolution is nearly instantaneous.
* **Platform specific dependency graphs:** The `pyproject.toml` leverages standard python environment markers (`sys_platform` and `platform_machine`) to automatically pull the correct dependencies. For example, it utilizes `tool.uv.sources` to map torch to the `cu124` index exclusively on AMD64 architectures, while falling back to standard Linux wheels for the Jetson.

---

## 5. Spot Robot Control (Bosdyn SDK)
The translation from a classified PyTorch gesture string (e.g., "swipe_left") to physical actuation is managed by `actions/spot.py`.

### Architecture
* **Bosdyn SDK:** The official Boston Dynamics python API is the backbone of the communication layer.
* **Lease & Estop Management:** Before executing commands, the module transparently acquires a "Lease" (control authority over Spot) and instantiates an EstopKeepAlive thread to satisfy Spot's safety heartbeat requirements.
* **Velocity Translation:** Dynamic gestures are commonly mapped to EulerZXY velocity commands via the `RobotCommandBuilder`, telling Spot to rotate, strafe, or walk based on the continuous gesture evaluation.

---

## 6. Core Inference Modes
The python engine can be executed in three distinct modes depending on the context:
* **LIVE_STREAM:** Designed for physical deployment. Runs MediaPipe asynchronously. Bypasses the expensive full-frame BlazePalm object detector by tracking the Region of Interest (ROI) from previous frames, feeding directly into the lightweight Hand Landmark model.
* **VIDEO:** Designed for offline testing. Processes pre-recorded Jester MP4s frame-by-frame chronologically to ensure predictable debugging.
* **IMAGE:** Designed for debugging the pure spatial extraction layer. Processes single static frames without feeding the temporal GRU.

---

## 7. Model Architecture: Feature Extraction & The GRU
The GRU ingests `(B, T, D)` sequences (Batch, Time, Dimension). Because MediaPipe extracts 21 landmarks containing 3 coordinates (x,y,z), a single hand represents a 63-Dimensional vector. The sliding window buffers T frames (e.g., 40).
The Bidirectional GRU processes the temporal relationships of this 40×63 matrix. Its outputs are mean-pooled across the temporal dimension before passing to a standard Linear classification head (softmax).

---

## 8. Training & Inference Evaluation
The pipeline provides robust monitoring and statistical analysis for both the model training phase and edge-inference execution.

### Training (WandB)
Weights & Biases (wandb) is integrated directly into the `jester/training/train.py` loop. It pushes metrics to the cloud in real-time, allowing monitoring of the GRU's training loss, validation accuracy, and learning rate decay over epochs, preventing overfitting on the Jester dataset.

### Inference Stats (web_ui/stats.py)
After an inference session concludes, the pipeline generates a suite of offline diagnostic visuals:
* **FPS Histograms:** To monitor edge-compute bottlenecks.
* **Latency Breakdowns:** A Gantt-style chart showing the exact millisecond delay between the Camera ingestion, MediaPipe spatial extraction, PyTorch GRU execution, and the UDP socket delay.
* **Hand Position Heatmaps:** Generates a 2D spatial heatmap demonstrating where the user's wrist spent the majority of its time in the camera frame, highly useful for diagnosing blind spots in the UI.
* **Handedness Distributions:** Pie charts mapping the ratio of Left vs. Right hand usage.

---

## 9. Skeletal Data Augmentation
Training dynamic gestures requires robustness against variances in how users perform the gesture (e.g., distance to camera, positioning in frame). Because the GRU's input is purely skeletal coordinates (not raw pixels), mathematical augmentations are applied directly to the 3D tensors:
* **Spatial Shift (Translation):** Simulates the user standing in different areas of the frame by adding a random (x,y) offset to all coordinates simultaneously.
* **Isotropic Scaling:** Simulates the user moving closer or further from the camera by multiplying all (x,y) coordinates by a randomized scale factor (e.g., 0.8x to 1.2x).

---

## 10. Advanced Gesture Control Features
The system supports a hierarchical variety of gesture types, evaluated per-frame. To prevent conflicting commands, a strict priority system governs execution. The priority order is: `Slider (Highest) > Stacked Gestures > Two-Hand Combinations > Dynamic Gestures > Static Gestures (Lowest)`.

### 10.1 Static Gestures (Lowest Priority)
These rely entirely on single-frame spatial analysis from MediaPipe. If a defined hand shape (e.g., `Open_Palm`, `Thumb_Up`) is detected on either the left or right hand, it triggers a mapped action. This acts as a reliable fallback when no dynamic movement is occurring.

### 10.2 Dynamic Gestures
These are analyzed over a rolling time window by the PyTorch GRU model. The model identifies continuous motion gestures such as "Swiping Left" or "Rolling Hand Forward." They override static gestures to prevent a static snapshot (like an open palm mid-swipe) from accidentally triggering a command while the user is actively performing a motion.

### 10.3 Two-Hand Combinations
When both hands perform specific static gestures simultaneously (e.g., `Victory` + `Victory`), the combination triggers a unique action.
* **Timer Used (two_hand_wait_ms):** Because human hands rarely enter the camera frame or form shapes at the exact same millisecond, a `two_hand_wait_ms` timer (typically 300ms) is employed. When a single hand static gesture is detected, the system delays executing its single-hand action for this duration to see if the second hand enters the frame to complete a combination.

### 10.4 Stacked Gestures (Series)
Users can string together a sequence of static gestures (e.g., `Pointing_Up → Victory → Pointing_Up`) to trigger complex actions.
* **Priority:** Once a sequence begins, the `GestureSeriesTracker` locks execution and ignores all other gesture types to prevent accidental interruptions.
* **Timer Used (series_timeout_ms):** The sequence must be completed within the `series_timeout_ms` window (e.g., 4000ms) from the start of the first gesture. If the timer expires, the sequence aborts.
* **Edge Cases Handled:** The tracker dynamically ignores "wrong" or transitional gestures performed in the middle of a sequence. If a user accidentally flashes an `Open_Palm` while transitioning between steps, the sequence simply waits rather than aborting, making it highly fault-tolerant.

### 10.5 Slider Control (Highest Priority)
The slider allows for continuous, analog control over numeric configuration parameters (e.g., robot height offset or speed) by mapping the physical distance between the user's two hands to a numeric value.
* **Priority:** Because it requires continuous adjustment, the slider commands absolute priority. While active, all other gesture recognition is suppressed.
* **Edge Cases Handled:**
  * By default, the distance is calculated between the center point of all landmarks on each hand. However, if the user performs a "Pointing_Up" gesture (index finger extended, middle finger folded down), the system tracks the absolute tip of the index finger (landmark 8) instead of the hand's center. This allows for extremely precise, granular distance adjustments.
  * The parameter scaling dynamically handles starting values of 0.0 by applying an absolute multiplier instead of a relative ratio.