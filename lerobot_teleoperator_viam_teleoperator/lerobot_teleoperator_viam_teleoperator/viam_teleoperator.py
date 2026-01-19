"""Viam implementation of Teleoperator interface for LeRobot."""
import asyncio
import math
import os
from typing import Any, Dict, List, Optional

from viam.components.arm import Arm
from viam.robot.client import RobotClient
from viam.logging import logging

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_viam_teleoperator import ViamTeleoperatorConfig

LOGGER = logging.getLogger(__name__)

async def viam_connect(
    api_key_value: str,
    api_key_id: str,
    robot_address: str,
):
    """Connect to a Viam robot using API key credentials."""

    opts = RobotClient.Options.with_api_key(api_key_value, api_key_id)
    return await RobotClient.at_address(robot_address, opts)

class ViamTeleoperator(Teleoperator):
    """
    Wraps a Viam Arm component to provide a LeRobot-compatible teleop interface.
    """

    config_class = ViamTeleoperatorConfig
    name = "viam_teleoperator"

    def __init__(self, config: ViamTeleoperatorConfig):
        super().__init__(config)
        self.id = config.id
        self._is_connected = False
        self._last_positions: Optional[List[float]] = None
        self._teleop_device_name = config.teleop_device_name
        self._num_joints = config.num_joints
        self._machine = None
        self._teleop_device: Arm = None
        self._loop = asyncio.new_event_loop()

    @property
    def action_features(self) -> Dict[str, type]:
        """Return action features - joint positions based on config."""
        return {f"joint_{i}.pos": float for i in range(self._num_joints)}

    @property
    def feedback_features(self) -> Dict[str, type]:
        """No feedback features supported."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        """Connect to the teleoperator device and disable torque for free movement."""
        self._machine = self._loop.run_until_complete(viam_connect(
            api_key_value=os.environ.get("VIAM_API_KEY", ""),
            api_key_id=os.environ.get("VIAM_API_KEY_ID", ""),
            robot_address=os.environ.get("VIAM_MACHINE_FQDN", "")
        ))

        self._teleop_device: Arm = Arm.from_robot(self._machine, self._teleop_device_name)
        if not self._teleop_device:
            raise ValueError(f"Arm '{self._teleop_device_name}' not found on robot.")

        self._is_connected = True

        # Disable torque so leader arm can be moved freely
        LOGGER.info("ViamTeleoperator connected (torque disabled)")

    @property
    def is_calibrated(self) -> bool:
        """Viam arms handle their own calibration."""
        return True

    def calibrate(self) -> None:
        """No calibration needed - handled by Viam arm component."""

    def configure(self) -> None:
        """No additional configuration needed."""
        self._loop.run_until_complete(self._set_torque(False))

    def get_action(self) -> Dict[str, float]:
        """
        Get current action (joint positions) from the teleoperator.

        Returns dict mapping joint names to positions in radians.
        """
        joint_positions = self._loop.run_until_complete(self._teleop_device.get_joint_positions())
        LOGGER.debug(f"Teleoperator joint positions: {joint_positions.values}")
        return {
            f"joint_{i}.pos": math.radians(pos)
            for i, pos in enumerate(joint_positions.values)
        }

    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        """Send feedback to the teleoperator (not implemented)."""
        # Force feedback not supported on Viam arms

    def disconnect(self) -> None:
        """Disconnect from the teleoperator."""
        if self._machine:
            self._loop.run_until_complete(self._machine.close())

        # Cancel any remaining tasks on our loop
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()

        self._is_connected = False
        self._teleop_device = None
        LOGGER.info("ViamTeleoperator disconnected")

    async def _set_torque(self, enabled: bool) -> None:
        """Enable or disable torque on the teleoperator arm."""
        await self._teleop_device.do_command({"command": "set_torque", "enable": enabled})
