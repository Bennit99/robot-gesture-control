#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

MODEL_PATH="$(python3 - <<'PY'
from pathlib import Path
import runpy
import sys
# When this script runs, its working dir is the `scripts/` folder. Resolve the
# project-level `cfg/cfg.py` by referencing the parent directory.
sys.path.insert(0, str(Path("..").resolve()))
ns = runpy.run_path(str(Path("../cfg/cfg.py")))
dynamic_model = ns.get("dynamic_model", {})
model_path = ""
if isinstance(dynamic_model, dict):
    model_path = dynamic_model.get("model", "") or ""
print(model_path)
PY
)"

if [[ -z "$MODEL_PATH" ]]; then
    echo "[TensorRT] No dynamic_model.model configured, skipping compilation."
    exit 0
fi

if [[ "$MODEL_PATH" != *.onnx ]]; then
    echo "[TensorRT] Configured model is not ONNX: $MODEL_PATH"
    exit 0
fi

PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ONNX_FILE="$PROJECT_ROOT/${MODEL_PATH#./}"
ENGINE_FILE="${ONNX_FILE%.onnx}.engine"

echo "[TensorRT] ONNX model: $ONNX_FILE"

if [[ ! -f "$ONNX_FILE" ]]; then
    echo "[TensorRT] ONNX file not found, skipping compilation."
    exit 0
fi

if [[ -f "$ENGINE_FILE" ]]; then
    echo "[TensorRT] Engine already exists, skipping compilation."
    exit 0
fi

if [[ ! -x /usr/src/tensorrt/bin/trtexec ]]; then
    echo "[TensorRT] trtexec not available, skipping compilation."
    exit 0
fi

echo "[TensorRT] Compiling FP16 engine..."
/usr/src/tensorrt/bin/trtexec \
    --onnx="$ONNX_FILE" \
    --saveEngine="$ENGINE_FILE" \
    --fp16
echo "[TensorRT] Engine written to $ENGINE_FILE"