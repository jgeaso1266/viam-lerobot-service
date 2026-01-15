"""Teleoperation methods"""
from logging import getLogger
from typing import Mapping, Optional
from dataclasses import dataclass, field
import asyncio
import time
import uuid

from viam.components.arm import Arm
from viam.components.input import Controller
from viam.utils import ValueTypes

from lerobot_robot_viam_robot.viam_robot import ViamRobotWrapper
from lerobot_teleoperator_viam_teleoperator.config_viam_teleoperator import ViamTeleoperatorConfig
from lerobot_teleoperator_viam_teleoperator.viam_teleoperator import ViamTeleopWrapper

LOGGER = getLogger(__name__)

@dataclass
class TeleopSession:
    """Tracks state for a teleoperation session."""
    teleop: 'ViamTeleopWrapper'  # Teleoperator device
    robot: 'ViamRobotWrapper'  # Robot wrapper
    fps: int
    start_time: float = field(default_factory=time.time)
    stop_requested: bool = False
    stopped: bool = False
    duration_s: float = 0.0
    teleop_wrapper: Optional['ViamTeleopWrapper'] = None  # Wrapper for leader arm

    async def wait_for_stop(self):
        """Wait until the session is fully stopped."""
        while not self.stopped:
            await asyncio.sleep(0.1)

async def start_teleoperation(
    self,
    teleop_device_type: str,
    fps: int = 30,
    *,
    extra: Optional[Mapping[str, ValueTypes]] = None,
) -> str:
    """
    Starts a teleoperation session.
    Equivalent to lerobot-teleoperate.
    """

    # Connect robot and configure speed/acceleration
    if not self.robot_wrapper.is_connected:
        self.robot_wrapper.connect()

    # Configure follower arm for responsiveness
    follower_arm = self.robot_wrapper.arm
    if follower_arm:
        await follower_arm.do_command({'set_speed': 180.0})
        await follower_arm.do_command({'set_acceleration': 500.0})

    # Create teleop wrapper and disable torque on leader
    if not self.teleop_wrapper.is_connected:
        self.teleop_wrapper.connect()

    # Create session
    session_id = str(uuid.uuid4())
    session = TeleopSession(
        teleop=self.teleop_device,
        robot=self.robot_wrapper,
        fps=fps,
    )
    session.teleop_wrapper = self.teleop_wrapper  # Store wrapper for cleanup
    self.teleop_sessions[session_id] = session

    # Start teleoperation loop in background
    asyncio.create_task(self._teleoperation_loop(session_id))

    LOGGER.info(
        "Started teleoperation session %s with device %s",
        session_id, teleop_device_type
    )
    return session_id

async def stop_teleoperation(
    self,
    session_id: str,
    *,
    extra: Optional[Mapping[str, ValueTypes]] = None,
) -> float:
    """Stops a teleoperation session."""
    session = self.teleop_sessions.get(session_id)
    if not session:
        raise ValueError(f"Unknown session: {session_id}")

    session.stop_requested = True
    await session.wait_for_stop()

    # Stop both arms
    if session.teleop and isinstance(session.teleop, Arm):
        await session.teleop.stop()
        # Re-enable torque on leader
        await session.teleop.do_command({'command': 'set_torque', 'enable': True})

    if self.robot_wrapper.arm:
        await self.robot_wrapper.arm.stop()

    # Disconnect teleop wrapper if present
    if hasattr(session, 'teleop_wrapper') and session.teleop_wrapper:
        session.teleop_wrapper.disconnect()

    duration_s = session.duration_s
    del self.teleop_sessions[session_id]

    LOGGER.info("Stopped teleoperation session %s, duration: %.2fs", session_id, duration_s)
    return duration_s

async def _teleoperation_loop(self, session_id: str):
    """Background teleoperation control loop with automatic retry on errors."""
    session = self.teleop_sessions[session_id]

    if session.teleop is None or self.robot_wrapper.arm is None:
        LOGGER.error(
            "Teleoperation session %s has no valid teleop device or robot arm",
            session_id
        )
        session.stopped = True
        return

    start_time = time.time()
    error_count = 0

    # Get the teleop wrapper (created in start_teleoperation)
    teleop_wrapper = getattr(session, 'teleop_wrapper', None)
    if teleop_wrapper is None and isinstance(session.teleop, Arm):
        # Fallback: create wrapper if not already created
        teleop_config = ViamTeleoperatorConfig(fps=session.fps)
        teleop_wrapper = ViamTeleopWrapper(teleop_config, session.teleop)
        await teleop_wrapper.connect_async()

    try:
        if isinstance(session.teleop, Arm) and teleop_wrapper:
            # Arm-to-arm teleoperation using get_action/send_action with retry
            while not session.stop_requested:
                try:
                    # Retry delay after errors
                    if error_count > 0:
                        LOGGER.info("Retrying (attempt %d)...", error_count + 1)
                        await asyncio.sleep(0.5)
                        error_count = 0  # Reset on successful iteration

                    # Get action from leader arm
                    action = await teleop_wrapper.get_action_async()

                    # Send action to follower arm
                    await self.robot_wrapper.send_action_async(action)

                except Exception as e:
                    error_count += 1
                    error_msg = str(e).lower()
                    if 'voltage' in error_msg:
                        LOGGER.warning("Error (voltage issue, retrying): %s", e)
                    else:
                        LOGGER.warning("Error (retrying): %s", e)
                    continue

        elif isinstance(session.teleop, Controller):
            teleop_controller = session.teleop
            follower_arm = self.robot_wrapper.arm

            # Get initial position
            current_pos = await follower_arm.get_end_position()
            next_pos = None

            while not session.stop_requested:
                try:
                    if error_count > 0:
                        await asyncio.sleep(0.5)
                        error_count = 0

                    # Get control events from controller
                    events = await teleop_controller.get_events()

                    # Extract delta values from controller axes
                    delta_x = 0.0
                    delta_y = 0.0
                    delta_z = 0.0
                    gripper_action = 1  # STAY = 1

                    for event in events:
                        if event.control == "AbsoluteX":
                            delta_x = event.value * 0.01
                        elif event.control == "AbsoluteY":
                            delta_y = event.value * 0.01
                        elif event.control == "AbsoluteRZ":
                            delta_z = event.value * 0.01
                        elif event.control == "ButtonSouth" and event.value > 0:
                            gripper_action = 0  # CLOSE
                        elif event.control == "ButtonWest" and event.value > 0:
                            gripper_action = 2  # OPEN

                    # Apply deltas to current position
                    next_pos.x = current_pos.x + delta_x
                    next_pos.y = current_pos.y + delta_y
                    next_pos.z = current_pos.z + delta_z

                    # Handle gripper
                    if gripper_action != 1:
                        # TODO: handle gripper open/close
                        pass

                    # Send updated positions to robot
                    await follower_arm.move_to_position(next_pos)
                    current_pos = next_pos

                    await asyncio.sleep(1.0 / session.fps)

                except Exception as e:
                    error_count += 1
                    LOGGER.warning("Controller error (retrying): %s", e)
                    continue

    finally:
        session.duration_s = time.time() - start_time
        session.stopped = True
