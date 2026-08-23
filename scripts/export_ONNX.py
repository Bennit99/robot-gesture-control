import argparse
import os
import sys
from pathlib import Path

import torch

# Ensure the root of the project is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jester.training.module import GestureClassifier


def export_to_onnx(checkpoint_path: str, output_path: str = None):
    print(f"Loading checkpoint: {checkpoint_path}")

    # Needs to be loaded via Lightning module if it's our gesture model
    model = GestureClassifier.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    # Inspect model's expected shape from hyperparameters
    seq_len = model.hparams.seq_len
    input_dim = model.hparams.input_dim
    print(
        f"Model expects input shape: [batch_size=1, seq_len={seq_len}, input_dim={input_dim}]"
    )

    # Generate dummy input with exact shape
    dummy_input = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    if output_path is None:
        base_name = Path(checkpoint_path).stem
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        output_path = os.path.join(models_dir, f"{base_name}.onnx")

    print(f"Exporting ONNX model to: {output_path}")

    # Export using required flags
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    print("Export successful!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input .ckpt or .pt file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output .onnx file (defaults to models/ directory)",
    )
    args = parser.parse_args()

    export_to_onnx(args.input, args.output)
