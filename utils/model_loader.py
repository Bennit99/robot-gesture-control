import os

import numpy as np
import torch


def load_pytorch_model(model_path: str, device="cpu"):
    """Loads a PyTorch Lightning GestureClassifier from a checkpoint."""
    from jester.training.module import GestureClassifier

    model = GestureClassifier.load_from_checkpoint(model_path, map_location=device)
    model.eval()
    if device != "cpu":
        model.to(device)
    return model


class UnifiedModelRunner:
    """A generic runner that can execute `.ckpt`, `.pt`, and `.onnx` models."""

    def __init__(self, model_path: str, device="cpu"):
        self.model_path = model_path
        self.ext = os.path.splitext(model_path)[1].lower()
        self.device = torch.device(device) if isinstance(device, str) else device

        if self.ext in [".ckpt", ".pt", ".pth"]:
            self.torch_model = load_pytorch_model(model_path, device=self.device)
            self.seq_len = self.torch_model.hparams.seq_len
            self.input_dim = self.torch_model.hparams.input_dim
            self.num_classes = self.torch_model.hparams.num_classes
            self.runner_type = "pytorch"

        elif self.ext == ".onnx":
            import onnxruntime as ort

            providers = (
                ["CUDAExecutionProvider"]
                if self.device.type == "cuda"
                else ["CPUExecutionProvider"]
            )
            try:
                self.ort_session = ort.InferenceSession(model_path, providers=providers)
            except Exception:
                print(
                    "CUDA failed for ONNX Runtime. Falling back to CPUExecutionProvider."
                )
                self.ort_session = ort.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )

            input_shape = self.ort_session.get_inputs()[0].shape
            output_shape = self.ort_session.get_outputs()[0].shape

            # Using defaults if the ONNX graph specifies dynamic shapes (like 'batch_size', etc.)
            self.seq_len = input_shape[1] if isinstance(input_shape[1], int) else 40
            self.input_dim = input_shape[2] if isinstance(input_shape[2], int) else 63
            self.num_classes = (
                output_shape[1] if isinstance(output_shape[1], int) else 27
            )
            self.runner_type = "onnx"
        else:
            raise ValueError(f"Unsupported model extension: {self.ext}")

    def __call__(self, window_np: np.ndarray) -> np.ndarray:
        """Predict a single window of shape (T, D). Returns logits (C,)."""
        if self.runner_type == "pytorch":
            # Pass through PyTorch natively
            x = torch.from_numpy(window_np).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.torch_model(x)
            return logits.squeeze(0).cpu().numpy()

        elif self.runner_type == "onnx":
            # Pass through ONNXRuntime natively
            x = np.expand_dims(window_np, axis=0).astype(np.float32)
            ort_inputs = {self.ort_session.get_inputs()[0].name: x}
            logits = self.ort_session.run(None, ort_inputs)[0]
            return logits[0]
