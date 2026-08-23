#!/bin/bash
set -e

# ── Defaults ──────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_FLAG=""
RUN_APP=false
DAEMON=true
ATTACH_LOGS=true

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Starts the gesture-control Docker stack.

Options:
  --preset <name>   Use a hardware preset:
                      jetson   – Jetson / Linux  (arm64, /dev/video0)
                      windows  – Windows Desktop (amd64, GPU, UDP camera)
  --build           (Re)build the Docker image before starting
  --run             Start main.py interactively in foreground
  --bash            Start an interactive bash shell in foreground
  --daemon          Run container in the background (DEFAULT)
  --detach          Run container in the background and do NOT attach logs
  -h, --help        Show this help

Examples:
  ./start.sh --preset jetson
  ./start.sh --preset windows --run
  ./start.sh --preset jetson --bash
EOF
    exit 0
}

# ── Parse args ────────────────────────────────────────────────────────────
PRESET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)  PRESET="$2"; shift 2 ;;
        --build)   BUILD_FLAG="--build"; shift ;;
        --run)     RUN_APP=true; DAEMON=false; shift ;;
        --bash)    DAEMON=false; shift ;;
        --daemon)  DAEMON=true; ATTACH_LOGS=true; shift ;;
        --detach)  DAEMON=true; ATTACH_LOGS=false; shift ;;
        -h|--help) usage ;;
        *)         echo "Unknown option: $1"; usage ;;
    esac
done

# ── Resolve preset ────────────────────────────────────────────────────────
if [[ -z "$PRESET" ]]; then
    # Auto-detect
    case "$(uname -m)" in
        aarch64) PRESET="jetson" ;;
        *)       PRESET="windows" ;;
    esac
    echo "Auto-detected preset: $PRESET"
fi

case "$PRESET" in
    jetson)
        export ARCH="arm64"
        OVERRIDE="docker-compose.jetson.yml"
        ;;
    windows)
        export ARCH="amd64"
        OVERRIDE="docker-compose.windows.yml"
        ;;
    *)
        echo "Unknown preset: $PRESET (available: jetson, windows)"
        exit 1
        ;;
esac

echo "Preset: $PRESET  |  ARCH=$ARCH  |  Override: $OVERRIDE"

# ── Load .env into shell (so Compose port interpolation + app vars match) ─
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a; source <(grep -v '^#' "$SCRIPT_DIR/.env" | grep '='); set +a
fi

# ── First-boot TensorRT compilation (Jetson / ONNX only) ─────────────────
if [[ "$PRESET" == "jetson" && -x "$SCRIPT_DIR/scripts/prepare_tensorrt_engine.sh" ]]; then
    "$SCRIPT_DIR/scripts/prepare_tensorrt_engine.sh"
fi

# ── Windows-only: start ffmpeg UDP camera bridge ──────────────────────────
FFMPEG_PID=""
if [[ "$PRESET" == "windows" ]]; then

    if [[ -z "$CAMERA_NAME" ]]; then
        echo ""
        echo "No CAMERA_NAME in .env — listing available devices:"
        ffmpeg -list_devices true -f dshow -i dummy 2>&1 || true
        echo ""
        read -rp "Enter your camera name (e.g. USB Camera): " CAMERA_NAME
    fi

    echo "Starting ffmpeg camera bridge: $CAMERA_NAME → udp://localhost:1234"
    ffmpeg -f dshow -i "video=$CAMERA_NAME" -pix_fmt yuv420p \
           -f mpegts udp://host.docker.internal:1234 &
    FFMPEG_PID=$!
    echo "ffmpeg PID: $FFMPEG_PID"
fi

# ── Docker Compose ────────────────────────────────────────────────────────
DOCKER_DIR="$SCRIPT_DIR/docker"
COMPOSE_FILES="-f $DOCKER_DIR/docker-compose.yml -f $DOCKER_DIR/$OVERRIDE"

if [[ -n "$BUILD_FLAG" ]]; then
    echo "Building image…"
    docker compose $COMPOSE_FILES build
fi

if [[ "$DAEMON" == true ]]; then
    echo "Starting container in background (daemon mode)…"
    docker compose $COMPOSE_FILES up -d
    if [[ "$ATTACH_LOGS" == true ]]; then
        echo "Attaching to container logs (Press Ctrl+C to detach without stopping the container)…"
        docker compose $COMPOSE_FILES logs -f gesture
    else
        echo "Container started in detached mode. Logs will not be shown."
    fi
elif [[ "$RUN_APP" == true ]]; then
    echo "Starting container (running main.py)…"
    docker compose $COMPOSE_FILES run --rm --service-ports gesture
else
    echo "Starting container (interactive shell)…"
    docker compose $COMPOSE_FILES run --rm --service-ports gesture bash
fi

# ── Cleanup ───────────────────────────────────────────────────────────────
if [[ "$DAEMON" == true && -n "$FFMPEG_PID" ]]; then
    echo "ffmpeg is running in the background (PID $FFMPEG_PID)."
    echo "You will need to stop it manually when done."
elif [[ -n "$FFMPEG_PID" ]]; then
    echo "Stopping ffmpeg (PID $FFMPEG_PID)…"
    kill "$FFMPEG_PID" 2>/dev/null || true
fi

echo "Done."
