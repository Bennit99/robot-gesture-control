"""Jester training package - PyTorch Lightning + W&B gesture classification.

Imports are lazy to avoid pulling heavy training-only dependencies
(wandb, tqdm, evaluate, etc.) when only inference modules are needed.
"""


def __getattr__(name: str):
    """Lazy import - only resolve symbols when actually accessed."""
    _imports = {
        "ARCHITECTURE_REGISTRY": (
            "jester.training.architectures",
            "ARCHITECTURE_REGISTRY",
        ),
        "build_model": ("jester.training.architectures", "build_model"),
        "JesterLandmarkDataModule": (
            "jester.training.data",
            "JesterLandmarkDataModule",
        ),
        "JesterLandmarkDataset": ("jester.training.data", "JesterLandmarkDataset"),
        "GestureClassifier": ("jester.training.module", "GestureClassifier"),
        "train": ("jester.training.train", "train"),
    }
    if name in _imports:
        import importlib

        mod_path, attr = _imports[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ARCHITECTURE_REGISTRY",
    "GestureClassifier",
    "JesterLandmarkDataModule",
    "JesterLandmarkDataset",
    "build_model",
    "train",
]
