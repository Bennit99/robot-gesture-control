gesture_mappings = {
    # Dynamic gestures (from stacked PyTorch pipeline)
    "dynamic": {
        "Sliding Two Fingers Left": "spot.yoga",
        "Sliding Two Fingers Right": None,
        "Sliding Two Fingers Up": None,
        "Sliding Two Fingers Down": None,
        "Swiping Left": None,
        "Swiping Right": None,
        "Swiping Up": None,
        "Swiping Down": None,
        "Stop Sign": None,
        "Thumb Up": None,
        "Thumb Down": None,
        "Pushing Hand Away": None,
        "Pulling Hand In": None,
        "Shaking Hand": None,
        "Drumming Fingers": None,
        "Pulling Two Fingers In": None,
        "Pushing Two Fingers Away": None,
        "Rolling Hand Backward": None,
        "Rolling Hand Forward": None,
        "Turning Hand Clockwise": None,
        "Turning Hand Counterclockwise": None,
        "Zooming In With Full Hand": None,
        "Zooming In With Two Fingers": None,
        "Zooming Out With Full Hand": None,
        "Zooming Out With Two Fingers": None,
    },
    # Single left hand gestures (static MediaPipe)
    "left": {
        "Open_Palm": None,
        "Closed_Fist": None,
        "Pointing_Up": None,
        "Thumb_Down": "spot.sit",
        "Thumb_Up": "spot.stand",
        "Victory": None,
        "ILoveYou": None,
    },
    # Single right hand gestures
    "right": {
        "Open_Palm": None,
        "Closed_Fist": None,
        "Pointing_Up": None,
        "Thumb_Down": "spot.sit",
        "Thumb_Up": "spot.stand",
        "Victory": None,
        "ILoveYou": None,
    },
    # Two-hand combinations (format: "left_gesture+right_gesture")
    "combinations": {
        "Victory+Victory": "spot.turn_360",
        "Open_Palm+Pointing_Up": "config.BODY_HEIGHT_OFFSET",
    },
    "series": {
        "A:Pointing_Up,A:Victory,A:Pointing_Up": "spot.turn_left",
        "A:Pointing_Up,A:ILoveYou,A:Pointing_Up": "spot.turn_right",
    },
}

from cfg.config_manager import load_overrides

overrides = load_overrides().get("gesture", {})
if overrides:
    # Fixed-key dictionaries can be updated
    for key in ["dynamic", "left", "right"]:
        if key in overrides:
            if key not in gesture_mappings:
                gesture_mappings[key] = {}
            gesture_mappings[key].update(overrides[key])

    # User-defined dictionaries must be fully replaced so deletions persist
    for key in ["combinations", "series"]:
        if key in overrides:
            gesture_mappings[key] = overrides[key]
