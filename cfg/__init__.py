import os
import shutil

# Import the actual configuration module as a submodule and expose it as `cfg`.
# Use a relative import to avoid recursive package import issues when other
# modules do `from cfg import cfg`.
from . import cfg as cfg

try:
    # Ensure the output directory exists and create a new numeric subdirectory
    os.makedirs(cfg.output, exist_ok=True)

    existing_dirs = [
        int(folder) for folder in os.listdir(cfg.output) if folder.isdigit()
    ]
    next_dir = str(max(existing_dirs) + 1) if existing_dirs else "0"

    new_dir_path = os.path.join(cfg.output, next_dir)
    os.makedirs(new_dir_path, exist_ok=True)

    # Copy the cfg.py file to the new output directory for record-keeping
    config_file_path = os.path.join(os.path.dirname(__file__), "cfg.py")
    shutil.copy(config_file_path, new_dir_path)

    # Update cfg.output to point to the new directory
    cfg.output = new_dir_path
except Exception as e:
    # Initialization should not crash import; log a warning instead.
    print(f"Warning initializing cfg package: {e}")
