from dotenv import load_dotenv

load_dotenv()


# Movement parameters
VELOCITY_BASE_SPEED = 1  # m/s for walking
VELOCITY_BASE_ANGULAR = 0.8  # rad/s for turning
VELOCITY_CMD_DURATION = 1  # seconds per velocity command
BODY_PITCH_ANGLE = 0.4  # radians for look up/down and bow
BODY_YAW_ANGLE = 0.4  # radians for look left/right
BODY_HEIGHT_OFFSET = 0.0  # meters for standing height

from cfg.config_manager import load_overrides

overrides = load_overrides().get("spot", {})
if overrides:
    if "VELOCITY_BASE_SPEED" in overrides:
        VELOCITY_BASE_SPEED = overrides["VELOCITY_BASE_SPEED"]
    if "VELOCITY_BASE_ANGULAR" in overrides:
        VELOCITY_BASE_ANGULAR = overrides["VELOCITY_BASE_ANGULAR"]
    if "VELOCITY_CMD_DURATION" in overrides:
        VELOCITY_CMD_DURATION = overrides["VELOCITY_CMD_DURATION"]
    if "BODY_PITCH_ANGLE" in overrides:
        BODY_PITCH_ANGLE = overrides["BODY_PITCH_ANGLE"]
    if "BODY_YAW_ANGLE" in overrides:
        BODY_YAW_ANGLE = overrides["BODY_YAW_ANGLE"]
    if "BODY_HEIGHT_OFFSET" in overrides:
        BODY_HEIGHT_OFFSET = overrides["BODY_HEIGHT_OFFSET"]
