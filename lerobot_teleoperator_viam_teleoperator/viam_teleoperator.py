"""Viam implementation of Teleoperator interface for LeRobot."""
import asyncio
import math
from typing import Any, Dict, List, Mapping, Optional
import logging

from viam.components.arm import Arm
from viam.components.input import Controller
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase

from lerobot.teleoperators.teleoperator import Teleoperator

from .config_viam_teleoperator import ViamTeleoperatorConfig

class ViamTeleopWrapper(Teleoperator):
    """
    Wraps a Viam Arm component to provide a LeRobot-compatible teleop interface.
    """

    config_class = ViamTeleoperatorConfig
    name = "viam_arm"

    def __init__(self, config: ViamTeleoperatorConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        super().__init__(config)
        self.config = config
        self.dependencies = dependencies
        self.id = config.id
        self._teleop_device = None
        self._is_connected = False
        self._stream = None
        self._last_positions: Optional[List[float]] = None
        self._logger = config.logger if config.logger else logging.getLogger(__name__)

    @property
    def action_features(self) -> Dict[str, type]:
        """Return action features - joint positions."""
        return {f"joint_{i}.pos": float for i in range(self.config.num_joints)}

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
        for name, resource in self.dependencies.items():
            if (isinstance(resource, Arm) or isinstance(resource, Controller)) and name.name == self.config.teleop_device:
                self._teleop_device = resource

        self._is_connected = True
        # Disable torque so leader arm can be moved freely
        asyncio.create_task(self._set_torque(False))
        self._logger.info("ViamTeleopWrapper connected (torque disabled)")

    @property
    def is_calibrated(self) -> bool:
        """Viam arms handle their own calibration."""
        return True

    def calibrate(self) -> None:
        """No calibration needed - handled by Viam arm component."""

    def configure(self) -> None:
        """No additional configuration needed."""

    def get_action(self) -> Dict[str, float]:
        """
        Get current action (joint positions) from the teleoperator.

        Returns dict mapping joint names to positions in radians.
        """
        joint_positions = asyncio.run(self._teleop_device.get_joint_positions())
        return {
            f"joint_{i}.pos": math.radians(pos)
            for i, pos in enumerate(joint_positions.values)
        }

    async def get_action_async(self) -> Dict[str, float]:
        """
        Async version of get_action.

        Returns dict mapping joint names to positions in radians.
        """
        joint_positions = await self._teleop_device.get_joint_positions()
        return {
            f"joint_{i}.pos": math.radians(pos)
            for i, pos in enumerate(joint_positions.values)
        }

    async def stream_actions(self) -> Any:
        """
        Stream joint positions from the teleoperator at configured FPS.

        Yields dicts mapping joint names to positions in radians.
        """
        stream = await self._teleop_device.stream_joint_positions(fps=self.config.fps)

        async for position_frame in stream:
            # Convert from degrees to radians
            positions_radians = [
                math.radians(deg)
                for deg in position_frame.positions.values
            ]
            self._last_positions = positions_radians

            yield {
                f"joint_{i}.pos": pos
                for i, pos in enumerate(positions_radians)
            }

    def send_feedback(self, feedback: Dict[str, Any]) -> None:
        """Send feedback to the teleoperator (not implemented)."""
        # Force feedback not supported on Viam arms

    def disconnect(self) -> None:
        """Disconnect from the teleoperator."""
        self._is_connected = False
        self._stream = None
        self._logger.info("ViamTeleopWrapper disconnected")

    async def _set_torque(self, enabled: bool) -> None:
        """Enable or disable torque on the teleoperator arm."""
        await self._teleop_device.do_command({"command": "set_torque", "enable": enabled})

    async def connect_async(self, calibrate: bool = True) -> None:
        """Async version of connect - disables torque for free movement."""
        self.connect(calibrate)

        self._is_connected = True
        await self._set_torque(False)
        self._logger.info("ViamTeleopWrapper connected (torque disabled)")
