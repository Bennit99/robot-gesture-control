"""
Gesture to Action Mapping

Maps detected gesture combinations to executable actions.
Supports single-hand and two-hand gesture combinations.
"""

import threading
import time
from typing import Any

from cfg import cfg
from cfg.gesture_config import gesture_mappings

ACTION_METADATA = {
    "spot.sit": {"display_name": "Sit", "description": "Sitting down", "duration": 2.5},
    "spot.stand": {
        "display_name": "Stand",
        "description": "Standing up",
        "duration": 2.5,
    },
    "spot.walk_forward": {
        "display_name": "Walk Forward",
        "description": "Walking forward",
        "duration": 2.5,
    },
    "spot.walk_backward": {
        "display_name": "Walk Backward",
        "description": "Walking backward",
        "duration": 2.5,
    },
    "spot.strafe_left": {
        "display_name": "Strafe Left",
        "description": "Strafing left",
        "duration": 2.5,
    },
    "spot.strafe_right": {
        "display_name": "Strafe Right",
        "description": "Strafing right",
        "duration": 2.5,
    },
    "spot.turn_left": {
        "display_name": "Turn Left",
        "description": "Turning left",
        "duration": 2.5,
    },
    "spot.turn_right": {
        "display_name": "Turn Right",
        "description": "Turning right",
        "duration": 2.5,
    },
    "spot.look_up": {
        "display_name": "Look Up",
        "description": "Looking up",
        "duration": 2.5,
    },
    "spot.look_down": {
        "display_name": "Look Down",
        "description": "Looking down",
        "duration": 2.5,
    },
    "spot.bow": {
        "display_name": "Bow",
        "description": "Bowing front body",
        "duration": 2.5,
    },
    "spot.wave": {
        "display_name": "Wave",
        "description": "Waving body side to side",
        "duration": 2.5,
    },
    "spot.stop": {
        "display_name": "Stop",
        "description": "Emergency stopping",
        "duration": 1.0,
    },
    "spot.turn_360": {
        "display_name": "Turn 360",
        "description": "Turning a full 360 degrees",
        "duration": 8.0,
    },
    "spot.yoga": {
        "display_name": "Yoga",
        "description": "Performing yoga routine",
        "duration": 8.5,
    },
}


class GestureActionMapper:
    """Maps gesture combinations to executable actions from cfg."""

    def __init__(self):
        self.mappings = gesture_mappings
        self._handlers: dict[str, Any] = {}
        self._load_handlers()
        self._active_action = None
        self._lock = threading.Lock()

        # Two-hand wait state
        self._pending_left: str | None = None
        self._pending_right: str | None = None
        self._left_timer: threading.Timer | None = None
        self._right_timer: threading.Timer | None = None
        self._state_lock = threading.Lock()

    def _load_handlers(self):
        """Dynamically load action handler modules and connect if needed."""
        try:
            from actions import spot

            self._handlers["spot"] = spot
        except ImportError as e:
            print(f"Warning: Could not load spot handler: {e}")

        # Auto-connect handlers that are referenced in gesture mappings
        if self._uses_handler("spot") and "spot" in self._handlers:
            print("[Spot] Spot actions mapped in config - ready for simulation or live mode.")

    def _uses_handler(self, handler_name: str) -> bool:
        """Check if any gesture mapping references the given handler."""
        for section in self.mappings.values():
            if isinstance(section, dict):
                for action in section.values():
                    if action and action.startswith(f"{handler_name}."):
                        return True
        return False

        # Add more handlers here as needed:
        # from actions import ui
        # self._handlers["ui"] = ui

    def _invoke_action(self, action_string: str) -> bool:
        """
        Invoke action from string like 'spot.sit' in a separate thread.

        Args:
            action_string: Action in format 'module.function'

        Returns:
            True if action was initiated or bypasses blocking, False otherwise
        """
        if not action_string:
            return False

        # spot.stop should bypass blocking and execute immediately
        is_stop = action_string == "spot.stop"

        with self._lock:
            if self._active_action is not None and not is_stop:
                # Ignore new action while one is active
                if getattr(cfg, "debug_pipeline_prints", False):
                    print(f"[Debug] Action Mapped: {action_string} (BLOCKED by {self._active_action['action_key']})")
                return False

        try:
            import re

            # Parse 'module.function(arg)'
            match = re.match(
                r"^([a-zA-Z_0-9]+)\.([a-zA-Z_0-9_]+)(?:\((.*)\))?$", action_string
            )
            if not match:
                print(f"Warning: Invalid action format '{action_string}'")
                return False

            module_name = match.group(1)
            func_name = match.group(2)
            args_str = match.group(3)

            args = []
            if args_str:
                for arg in args_str.split(","):
                    arg = arg.strip()
                    try:
                        args.append(float(arg))
                    except ValueError:
                        args.append(arg)

            if module_name == "config":
                from cfg import spot_config

                if hasattr(spot_config, func_name):
                    current_val = getattr(spot_config, func_name)
                    if args and isinstance(args[0], (int, float)):
                        # Add as a flat percentage (e.g., 25 -> 0.25)
                        new_val = current_val + (args[0] / 100.0)
                        setattr(spot_config, func_name, new_val)
                        print(f"[Config] Updated {func_name} to {new_val:.3f}")

                        # Set active action to visualize in UI
                        display_name = func_name.replace("_", " ").title()
                        with self._lock:
                            self._active_action = {
                                "action_key": action_string,
                                "display_name": display_name,
                                "description": f"Updated to {new_val:.2f}",
                                "duration": 2.0,
                                "start_time": time.time(),
                            }

                        # Clear it after 2 seconds
                        def clear_action():
                            time.sleep(2.0)
                            with self._lock:
                                if (
                                    self._active_action
                                    and self._active_action.get("action_key")
                                    == action_string
                                ):
                                    self._active_action = None

                        threading.Thread(target=clear_action, daemon=True).start()

                    return True
                else:
                    print(f"Warning: Unknown config parameter '{func_name}'")
                    return False

            handler = self._handlers.get(module_name)
            if handler is None:
                print(f"Warning: Unknown handler module '{module_name}'")
                return False

            func = getattr(handler, func_name, None)
            if func is None:
                print(f"Warning: Unknown function '{func_name}' in {module_name}")
                return False

            # Get action metadata (fallback to func_name if not listed)
            meta = ACTION_METADATA.get(
                action_string,
                {
                    "display_name": func_name.replace("_", " ").title(),
                    "description": f"Executing {func_name}",
                    "duration": 2.5,
                },
            )

            if getattr(cfg, "debug_pipeline_prints", False):
                print(f"[Debug] Action Executing: {action_string} (bypassed lock: {is_stop})")

            start_time_worker = time.time()

            def worker():
                try:
                    func(*args)
                    elapsed_func = time.time() - start_time_worker
                    remaining_sleep = max(0.0, meta["duration"] - elapsed_func)
                    if remaining_sleep > 0:
                        time.sleep(remaining_sleep)
                except Exception as e:
                    print(f"Error executing action function '{action_string}': {e}")
                finally:
                    with self._lock:
                        # Only clear if it is the action we started (avoid clearing a stop command that interrupted us)
                        if (
                            self._active_action
                            and self._active_action.get("action_key") == action_string
                        ):
                            self._active_action = None

            if is_stop:
                # If it's a stop command, clear current action state immediately and run spot.stop synchronously
                with self._lock:
                    self._active_action = {
                        "action_key": action_string,
                        "display_name": meta["display_name"],
                        "description": meta["description"],
                        "duration": meta["duration"],
                        "start_time": time.time(),
                    }
                func(
                    *args
                )  # Run stop immediately in current thread to ensure fast stop
                
                def clear_stop():
                    time.sleep(meta["duration"])
                    with self._lock:
                        if (
                            self._active_action
                            and self._active_action.get("action_key") == action_string
                        ):
                            self._active_action = None

                threading.Thread(target=clear_stop, daemon=True).start()
                return True

            # Set active action info
            with self._lock:
                self._active_action = {
                    "action_key": action_string,
                    "display_name": meta["display_name"],
                    "description": meta["description"],
                    "duration": meta["duration"],
                    "start_time": time.time(),
                }

            # Start background thread
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            return True

        except Exception as e:
            print(f"Error preparing action '{action_string}': {e}")
            with self._lock:
                self._active_action = None
            return False

    def execute(
        self, left_gesture: str | None, right_gesture: str | None
    ) -> bool:
        """
        Execute action based on gesture combination.

        Priority:
        1. Dynamic gesture mappings (from stacked PyTorch pipeline)
        2. Two-hand combinations (if both hands detected)
        3. Single-hand gestures
        """
        executed = False

        # 0. Check dynamic gesture mappings (from stacked pipeline)
        if left_gesture and right_gesture is None:
            dynamic_mappings = self.mappings.get("dynamic", {})
            if left_gesture in dynamic_mappings:
                action = dynamic_mappings[left_gesture]
                print(f"[GestureMapper] Dynamic: {left_gesture} -> {action}")
                if self._invoke_action(action):
                    return True

        timeout_sec = cfg.dynamic_model.get("two_hand_wait_ms", 300) / 1000.0

        with self._state_lock:
            # Update pending states based on incoming gestures
            if left_gesture:
                self._pending_left = left_gesture
                if self._left_timer:
                    self._left_timer.cancel()
                    self._left_timer = None
            if right_gesture:
                self._pending_right = right_gesture
                if self._right_timer:
                    self._right_timer.cancel()
                    self._right_timer = None

            # 1. Check two-hand combinations (highest priority)
            if self._pending_left and self._pending_right:
                combo_key = f"{self._pending_left}+{self._pending_right}"
                
                # We have both hands, process combo
                self._pending_left = None
                self._pending_right = None

                combinations = self.mappings.get("combinations", {})
                if combo_key in combinations:
                    action = combinations[combo_key]
                    print(f"[GestureMapper] Combo: {combo_key} -> {action}")
                    if self._invoke_action(action):
                        return True
                return False

            # We only have one hand pending. We will wait timeout_sec to see if the other arrives.
            def fire_single(hand: str, gesture: str):
                with self._state_lock:
                    if hand == "left":
                        if self._pending_left != gesture:
                            return
                        self._pending_left = None
                        self._left_timer = None
                    else:
                        if self._pending_right != gesture:
                            return
                        self._pending_right = None
                        self._right_timer = None

                # Fire the single-hand gesture
                mappings = self.mappings.get(hand, {})
                if gesture in mappings:
                    action = mappings[gesture]
                    print(f"[GestureMapper] {hand.capitalize()}: {gesture} -> {action}")
                    self._invoke_action(action)

            if self._pending_left and not self._left_timer:
                self._left_timer = threading.Timer(timeout_sec, fire_single, args=["left", self._pending_left])
                self._left_timer.start()
                executed = True

            if self._pending_right and not self._right_timer:
                self._right_timer = threading.Timer(timeout_sec, fire_single, args=["right", self._pending_right])
                self._right_timer.start()
                executed = True

        return executed


# Singleton instance for easy import
gesture_mapper = GestureActionMapper()
