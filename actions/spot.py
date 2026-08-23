"""
Spot Action Module

Provides gesture-triggered actions for the Boston Dynamics Spot robot.
Uses the bosdyn SDK for robot control. Connects lazily on first action call.
Connection details are loaded from .env (SPOT_USERNAME, SPOT_PASSWORD, IP_ADRESSE).
"""

import os
import time

import bosdyn.client
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
)
from bosdyn.geometry import EulerZXY
from dotenv import load_dotenv

from cfg import spot_config

load_dotenv()


class SpotController:
    """Manages the connection and command interface to the Spot robot."""

    def __init__(self):
        self._robot = None
        self._command_client = None
        self._lease_client = None
        self._lease_keep_alive = None
        self._estop_client = None
        self._estop_endpoint = None
        self._estop_keep_alive = None
        self._simulation_mode = True
        self._lease_error_occurred = False

    def _on_lease_error(self, exc):
        print(f"[Spot] Lease error caught by keep-alive! {exc}")
        self._lease_error_occurred = True

    @property
    def is_connected(self) -> bool:
        return self._robot is not None

    @property
    def has_control(self) -> bool:
        """Check if we still actively hold the lease."""
        if self._simulation_mode:
            return True
        if getattr(self, "_lease_error_occurred", False):
            return False
        if not self.is_connected or not self._lease_keep_alive:
            return False
        try:
            return self._lease_keep_alive.is_alive()
        except AttributeError:
            return True

    def connect(self, exit_on_fail: bool = True, log_callback=None):
        """Connect to Spot, acquire lease, start e-stop, and power on.
        Simulation mode behavior depends on spot_config.SIMULATION_MODE."""
        def log(msg):
            print(msg)
            if log_callback:
                log_callback(msg)

        if self.is_connected:
            if self._simulation_mode:
                log("[Spot] SDK already initialized. Attempting to reclaim lease...")
                try:
                    self._lease_client.take()
                    self._lease_error_occurred = False
                    self._simulation_mode = False
                    log("[Spot] Successfully reclaimed lease and returned to Live Mode.")
                except Exception as e:
                    log(f"[Spot] Failed to reclaim lease: {e}")
                    self._simulation_mode = True
                    if exit_on_fail:
                        import sys
                        sys.exit(1)
                    raise e
            return

        hostname = os.getenv("IP_ADRESSE", "192.168.50.3")
        username = os.getenv("SPOT_USERNAME", "spot")
        password = os.getenv("SPOT_PASSWORD", "")

        log(f"[Spot] Connecting to {hostname}...")

        try:
            # Create robot object and authenticate
            sdk = bosdyn.client.create_standard_sdk("GestureControlClient")
            self._robot = sdk.create_robot(hostname)
            self._robot.authenticate(username, password)
            self._robot.time_sync.wait_for_sync()

            # E-Stop — try to register, but don't fail if motors are
            # already on (e.g. tablet is active).  The tablet keeps its
            # own E-Stop so we can safely skip ours.
            try:
                self._estop_client = self._robot.ensure_client(
                    EstopClient.default_service_name
                )
                self._estop_endpoint = EstopEndpoint(
                    client=self._estop_client,
                    name="gesture-control-estop",
                    estop_timeout=9.0,
                )
                self._estop_endpoint.force_simple_setup()
                self._estop_keep_alive = EstopKeepAlive(self._estop_endpoint)
                log("[Spot] E-Stop endpoint registered.")
            except Exception as e:
                log(f"[Spot] E-Stop setup skipped (tablet active?): {e}")
                self._estop_client = None
                self._estop_endpoint = None
                self._estop_keep_alive = None

            # Lease — use take() to grab control even if tablet holds it.
            # The tablet can re-take the lease back at any time.
            self._lease_client = self._robot.ensure_client(
                LeaseClient.default_service_name
            )
            try:
                self._lease_client.take()
            except Exception as e:
                log(f"[Spot] Warning: Failed to forcefully take lease: {e}")
                
            self._lease_error_occurred = False
            self._lease_keep_alive = LeaseKeepAlive(
                self._lease_client, 
                must_acquire=True, 
                return_at_exit=True,
                on_failure_callback=self._on_lease_error
            )

            # Command client
            self._command_client = self._robot.ensure_client(
                RobotCommandClient.default_service_name
            )

            # Power on motors if not already powered
            if not self._robot.is_powered_on():
                log("[Spot] Powering on motors (this may take a few seconds)...")
                self._robot.power_on(timeout_sec=20)
            log(f"[Spot] Motors powered on: {self._robot.is_powered_on()}")

            self._simulation_mode = False
            self._powered_on = True
            log("[Spot] Connected and ready.")

        except (Exception, KeyboardInterrupt) as e:
            self._robot = None
            self._simulation_mode = True
            log("\033[91;1m  [Spot] CONNECTION FAILED.\033[0m")
            log(f"\033[93m  Error: {e}\033[0m")
            if exit_on_fail:
                log("\033[91;1m  EXITING.\033[0m")
                import sys
                sys.exit(1)
            else:
                raise e

    def disconnect(self, log_callback=None):
        """Release all resources."""
        def log(msg):
            print(msg)
            if log_callback:
                log_callback(msg)

        if self._estop_keep_alive:
            self._estop_keep_alive.shutdown()
            self._estop_keep_alive = None

        if self._lease_keep_alive:
            self._lease_keep_alive.shutdown()
            self._lease_keep_alive = None

        self._robot = None
        self._simulation_mode = True
        log("[Spot] Disconnected and reverted to Simulation Mode.")

    def _ensure_connected(self):
        """Ensure we are connected (or in simulation mode) before sending a command."""
        if not self.is_connected and not self._simulation_mode:
            self.connect()

    # ------------------------------------------------------------------
    # Command helpers
    # ------------------------------------------------------------------

    def _issue_robot_command(self, command, end_time_secs=None):
        """Send a command to the robot, or log it in simulation mode."""
        self._ensure_connected()
        if self._simulation_mode:
            return  # action is already printed by the calling method
        if end_time_secs:
            end_time = time.time() + end_time_secs
            self._command_client.robot_command(command, end_time_secs=end_time)
        else:
            self._command_client.robot_command(command)

    def _velocity_command(self, v_x=0.0, v_y=0.0, v_rot=0.0, duration=None):
        """Send a velocity command (walking/turning)."""
        if duration is None:
            duration = spot_config.VELOCITY_CMD_DURATION
        cmd = RobotCommandBuilder.synchro_velocity_command(
            v_x=v_x, v_y=v_y, v_rot=v_rot
        )
        self._issue_robot_command(cmd, end_time_secs=duration)

    def _stand_with_body_orientation(self, yaw=0.0, roll=0.0, pitch=0.0):
        """Stand with a specific body orientation."""
        orientation = EulerZXY(yaw=yaw, roll=roll, pitch=pitch)
        cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=orientation)
        self._issue_robot_command(cmd)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def sit(self):
        """Tell Spot to sit."""
        print("[Spot] Sitting")
        cmd = RobotCommandBuilder.synchro_sit_command()
        self._issue_robot_command(cmd)

    def stand(self):
        """Tell Spot to stand."""
        print("[Spot] Standing")
        cmd = RobotCommandBuilder.synchro_stand_command()
        self._issue_robot_command(cmd)

    def reset_body_orientation(self):
        """Reset the body orientation to flat."""
        self._stand_with_body_orientation(roll=0.0)

    def micro_tilt_left(self):
        """Tilt slightly left as visual feedback."""
        self._stand_with_body_orientation(roll=-0.08)

    def micro_tilt_right(self):
        """Tilt slightly right as visual feedback."""
        self._stand_with_body_orientation(roll=0.08)

    def stand_height(self, height: float):
        """Adjust Spot's standing height."""
        print(f"[Spot] Standing height set to {height:.2f}")
        cmd = RobotCommandBuilder.synchro_stand_command(body_height=height)
        self._issue_robot_command(cmd)

    def walk_forward(self):
        """Move Spot forward."""
        print("[Spot] Walking forward")
        self._velocity_command(v_x=spot_config.VELOCITY_BASE_SPEED)

    def walk_backward(self):
        """Move Spot backward."""
        print("[Spot] Walking backward")
        self._velocity_command(v_x=-spot_config.VELOCITY_BASE_SPEED)

    def strafe_left(self):
        """Move Spot sideways to the left."""
        print("[Spot] Strafing left")
        self._velocity_command(v_y=spot_config.VELOCITY_BASE_SPEED)

    def strafe_right(self):
        """Move Spot sideways to the right."""
        print("[Spot] Strafing right")
        self._velocity_command(v_y=-spot_config.VELOCITY_BASE_SPEED)

    def turn_left(self):
        """Rotate Spot to the left."""
        print("[Spot] Turning left")
        self._velocity_command(v_rot=spot_config.VELOCITY_BASE_ANGULAR)

    def turn_right(self):
        """Rotate Spot to the right."""
        print("[Spot] Turning right")
        self._velocity_command(v_rot=-spot_config.VELOCITY_BASE_ANGULAR)

    def turn_360(self):
        """Perform a full 360-degree rotation.
        
        This splits the velocity command into 1-second chunks to avoid the
        TooDistantError caused by Spot's maximum allowed command duration.
        """
        import math
        import time
        print("[Spot] Turning 360 degrees")
        speed = spot_config.VELOCITY_BASE_ANGULAR if spot_config.VELOCITY_BASE_ANGULAR > 0 else 0.5
        total_duration = ((2 * math.pi) / speed) * 1.01
        
        chunks = int(total_duration)
        remainder = total_duration - chunks
        
        for _ in range(chunks):
            self._velocity_command(v_rot=speed, duration=1.0)
            time.sleep(1.0)
            
        if remainder > 0:
            self._velocity_command(v_rot=speed, duration=remainder)
            time.sleep(remainder)

    def look_up(self):
        """Tilt Spot's body to look up."""
        print("[Spot] Looking up")
        self._stand_with_body_orientation(pitch=spot_config.BODY_PITCH_ANGLE)

    def look_down(self):
        """Tilt Spot's body to look down."""
        print("[Spot] Looking down")
        self._stand_with_body_orientation(pitch=-spot_config.BODY_PITCH_ANGLE)

    def look_left(self):
        """Tilt Spot's body to look left."""
        print("[Spot] Looking left")
        self._stand_with_body_orientation(yaw=spot_config.BODY_YAW_ANGLE)

    def look_right(self):
        """Tilt Spot's body to look right."""
        print("[Spot] Looking right")
        self._stand_with_body_orientation(yaw=-spot_config.BODY_YAW_ANGLE)

    def bow(self):
        """Lower Spot's front body in a bow."""
        print("[Spot] Bowing")
        self._stand_with_body_orientation(pitch=-spot_config.BODY_PITCH_ANGLE * 1.5)

    def self_right(self):
        """Spot gets up if it has fallen."""
        print("[Spot] Self-righting")
        cmd = RobotCommandBuilder.selfright_command()
        self._issue_robot_command(cmd)

    def wave(self):
        """Spot performs a wave gesture. Falls back to look-up if no choreography."""
        print("[Spot] Waving (body wiggle)")
        # Wiggle body side to side as a simple "wave"
        for _ in range(3):
            self._stand_with_body_orientation(roll=0.3)
            time.sleep(0.3)
            self._stand_with_body_orientation(roll=-0.3)
            time.sleep(0.3)
        self._stand_with_body_orientation()  # reset to neutral

    def yoga(self):
        """Spot performs a yoga routine."""
        import time
        print("[Spot] Performing yoga (stretches)")
        # Twisting stretch (combine yaw and pitch)
        self._stand_with_body_orientation(yaw=1, pitch=1)
        time.sleep(1.5)
        self._stand_with_body_orientation(yaw=-1, pitch=1)
        time.sleep(1.5)
        self._stand_with_body_orientation(yaw=1, pitch=-1)
        time.sleep(1.5)
        self._stand_with_body_orientation(yaw=-1, pitch=-1)
        time.sleep(1.5)
        
        # Base position
        self._stand_with_body_orientation()

    def cat_cow(self):
        """Spot performs a cat-cow stretch."""
        import time
        print("[Spot] Performing cat-cow")
        # Cow (Look up)
        self._stand_with_body_orientation(pitch=spot_config.BODY_PITCH_ANGLE * 1.5)
        time.sleep(1.5)
        # Cat (Bow)
        self._stand_with_body_orientation(pitch=-spot_config.BODY_PITCH_ANGLE * 1.5)
        time.sleep(1.5)
        
        # Base position
        self._stand_with_body_orientation()

    def stop(self):
        """Emergency stop. Spot freezes in place."""
        print("[Spot] Emergency stop")
        cmd = RobotCommandBuilder.stop_command()
        self._issue_robot_command(cmd)

        self._issue_robot_command(cmd)


# ---------------------------------------------------------------------------
# Singleton controller and module-level convenience functions
#
# WARNING: All public functions defined below (that don't start with '_')
# are dynamically discovered by the frontend API and exposed as selectable
# gesture actions in the Web Settings UI.
# ---------------------------------------------------------------------------

_controller = SpotController()


def connect(exit_on_fail: bool = True, log_callback=None):
    """Manually connect to Spot (optional, auto-connects on first command)."""
    _controller.connect(exit_on_fail=exit_on_fail, log_callback=log_callback)


def disconnect(log_callback=None):
    """Disconnect and power off Spot."""
    _controller.disconnect(log_callback=log_callback)


def sit():
    """Tell Spot to sit."""
    _controller.sit()


def stand():
    """Tell Spot to stand."""
    _controller.stand()


def stand_height(height: float):
    """Adjust Spot's standing height."""
    _controller.stand_height(height)


def reset_body_orientation():
    """Reset the body orientation to flat."""
    _controller.reset_body_orientation()


def micro_tilt_left():
    """Tilt slightly left as visual feedback."""
    _controller.micro_tilt_left()


def micro_tilt_right():
    """Tilt slightly right as visual feedback."""
    _controller.micro_tilt_right()


def walk_forward():
    """Move Spot forward."""
    _controller.walk_forward()


def walk_backward():
    """Move Spot backward."""
    _controller.walk_backward()


def strafe_left():
    """Move Spot sideways to the left."""
    _controller.strafe_left()


def strafe_right():
    """Move Spot sideways to the right."""
    _controller.strafe_right()


def turn_left():
    """Rotate Spot to the left."""
    _controller.turn_left()


def turn_right():
    """Rotate Spot to the right."""
    _controller.turn_right()


def turn_360():
    """Perform a full 360-degree rotation."""
    _controller.turn_360()


def look_up():
    """Tilt Spot's body to look up."""
    _controller.look_up()


def look_down():
    """Tilt Spot's body to look down."""
    _controller.look_down()


def look_left():
    """Tilt Spot's body to look left."""
    _controller.look_left()


def look_right():
    """Tilt Spot's body to look right."""
    _controller.look_right()


def bow():
    """Lower Spot's front body in a bow."""
    _controller.bow()


def self_right():
    """Spot gets up if it has fallen."""
    _controller.self_right()


def wave():
    """Spot performs a wave gesture."""
    _controller.wave()


def yoga():
    """Spot performs a yoga routine."""
    _controller.yoga()


def cat_cow():
    """Spot performs a cat-cow stretch."""
    _controller.cat_cow()


def stop():
    """Emergency stop. Spot freezes in place."""
    _controller.stop()


# ---------------------------------------------------------------------------
# Interactive test mode (run with: python -m actions.spot)
# ---------------------------------------------------------------------------


def _getch_cross_platform():
    """Read a single keypress without waiting for Enter. Works on Windows, Linux, macOS.

    On Unix, the caller must set the terminal to raw mode beforehand
    (see ``main()``).
    """
    try:
        # Windows
        import msvcrt

        ch = msvcrt.getch()
        # Arrow keys on Windows: 0xE0 or 0x00 prefix + second byte
        if ch in (b"\xe0", b"\x00"):
            ch2 = msvcrt.getch()
            arrow_map = {
                b"H": b"\x1b[A",
                b"P": b"\x1b[B",
                b"K": b"\x1b[D",
                b"M": b"\x1b[C",
            }
            return arrow_map.get(ch2, b"")
        return ch
    except ImportError:
        pass

    # Unix / Linux / macOS — terminal already in raw mode
    import sys

    ch = sys.stdin.buffer.read(1)
    # ESC sequence for arrow keys: \x1b [ A/B/C/D
    if ch == b"\x1b":
        import select

        # Check if more bytes are coming (arrow key sequence)
        if select.select([sys.stdin], [], [], 0.05)[0]:
            seq = sys.stdin.buffer.read(1)
            if seq == b"[":
                code = sys.stdin.buffer.read(1)
                return b"\x1b[" + code
        return ch  # plain ESC
    return ch


def _kbhit_cross_platform() -> bool:
    """Check if a keypress is available (non-blocking). Works on Windows, Linux, macOS.

    On Unix, the terminal must already be in raw mode for this to work
    without requiring Enter.
    """
    try:
        import msvcrt

        return msvcrt.kbhit()
    except ImportError:
        pass

    import select
    import sys

    return select.select([sys.stdin], [], [], 0)[0] != []


def main():
    """Interactive keyboard controller for testing the Spot connection.

    Uses arrow keys / WASD for movement, with a live terminal UI.
    Works on Windows, Linux, macOS, SSH, and Docker.
    """
    import sys

    HELP = """
╔══════════════════════════════════════════════╗
║         Spot Keyboard Controller             ║
╠══════════════════════════════════════════════╣
║                                              ║
║         ↑ / W      Forward                   ║
║    ← / A       → / D   Strafe L / R          ║
║         ↓ / S      Backward                  ║
║                                              ║
║    Q  Turn left        E  Turn right         ║
║    Z  Look up          X  Look down          ║
║    R  Stand             F  Sit               ║
║    B  Bow               V  Wave              ║
║    T  Self-right                             ║
║    SPACE  Emergency stop                     ║
║    ESC    Quit                               ║
╚══════════════════════════════════════════════╝
"""
    print(HELP)
    print("[Test] Connecting to Spot...")
    connect()
    print("[Test] Ready - press keys to send commands (ESC to quit)\n")

    # Arrow key ANSI escape sequences (unified across platforms)
    ARROW_MAP = {
        b"\x1b[A": "up",
        b"\x1b[B": "down",
        b"\x1b[D": "left",
        b"\x1b[C": "right",
    }

    KEY_ACTIONS = {
        "up": walk_forward,
        "down": walk_backward,
        "left": strafe_left,
        "right": strafe_right,
        b"w": walk_forward,
        b"a": strafe_left,
        b"s": walk_backward,
        b"d": strafe_right,
        b"q": turn_left,
        b"e": turn_right,
        b"z": look_up,
        b"x": look_down,
        b"r": stand,
        b"f": sit,
        b"b": bow,
        b"v": wave,
        b"t": self_right,
        b" ": stop,
    }

    # On Unix/Linux/Docker: put terminal into raw mode for the entire
    # session so keypresses arrive instantly without needing Enter.
    _old_term_settings = None
    try:
        import termios
        import tty

        fd = sys.stdin.fileno()
        _old_term_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)  # cbreak = raw keys but still allow Ctrl+C
    except (ImportError, OSError):
        pass  # Windows or non-TTY — handled by msvcrt

    try:
        while True:
            if _kbhit_cross_platform():
                ch = _getch_cross_platform()

                # ESC (plain, not part of arrow sequence)
                if ch == b"\x1b":
                    print("\n[Test] Exiting.")
                    break

                # Arrow keys (ANSI sequences)
                arrow = ARROW_MAP.get(ch)
                if arrow:
                    KEY_ACTIONS[arrow]()
                    continue

                # Regular keys (lowercase)
                key = ch.lower() if hasattr(ch, "lower") else ch
                action = KEY_ACTIONS.get(key)
                if action:
                    action()
                else:
                    print(f"  (unknown key: {ch!r})")

            time.sleep(0.05)  # ~20 Hz poll rate

    except KeyboardInterrupt:
        print("\n[Test] Interrupted.")
    finally:
        # Restore terminal settings on Unix before cleanup prints
        if _old_term_settings is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _old_term_settings)
        stop()
        disconnect()


if __name__ == "__main__":
    main()
