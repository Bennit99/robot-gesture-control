import json
import threading
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.local.json"
_lock = threading.Lock()

def load_overrides():
    """Load user configuration overrides from the local JSON file."""
    with _lock:
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[ConfigManager] Failed to load {CONFIG_FILE}: {e}")
        return {}

def save_overrides(data):
    """Save user configuration overrides to the local JSON file."""
    # First load without lock to avoid deadlock if load_overrides locked again
    # But wait, we want atomic read-modify-write!
    with _lock:
        overrides = {}
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    overrides = json.load(f)
            except Exception as e:
                print(f"[ConfigManager] Failed to load before save: {e}")
                # We can choose to overwrite or just start fresh
                
        # Deep merge the data dict into overrides
        for category, values in data.items():
            if category not in overrides:
                overrides[category] = {}
            overrides[category].update(values)
            
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(overrides, f, indent=4)
        except Exception as e:
            print(f"[ConfigManager] Failed to save {CONFIG_FILE}: {e}")
