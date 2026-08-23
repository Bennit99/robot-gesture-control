"""
Standalone parallel Jester landmark extraction pipeline.

Separated from the main gesture-control system because:
  - Jester processing is an **offline batch pipeline** (multiprocessing, parallel I/O)
  - The main system handles **real-time / interactive** modes (single-process, callbacks)
  - Different execution models, dependencies, and scaling concerns
  - Can be run independently without the rest of the system

Usage
-----
CLI::

    python -m jester                           # process all unprocessed videos
    python -m jester --video 4/107946          # process a single video
    python -m jester --workers 8               # custom worker count
    python -m jester --cpu                     # force CPU delegate

Python::

    from jester import process_video, run_parallel

    # Process one video (creates its own MediaPipe instance)
    result = process_video("107946")

    # Batch-process everything in parallel
    run_parallel("./data/jester/images", "./data/jester/landmarks", "models/hand_landmarker.task")
"""


def __getattr__(name: str):
    """Lazy import - avoid pulling heavy extraction deps at package init."""
    _imports = {
        "process_video": ("jester.extract_landmarks", "process_video"),
        "run_parallel": ("jester.extract_landmarks", "run_parallel"),
        "gather_video_folders": ("jester.extract_landmarks", "gather_video_folders"),
        "get_optimal_workers": ("jester.extract_landmarks", "get_optimal_workers"),
    }
    if name in _imports:
        import importlib

        mod_path, attr = _imports[name]
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "gather_video_folders",
    "get_optimal_workers",
    "process_video",
    "run_parallel",
]
