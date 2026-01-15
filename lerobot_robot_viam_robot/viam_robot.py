"""Viam implementation of Robot interface for LeRobot."""
import asyncio
from functools import cached_property
from typing import Any, Dict, Mapping, Optional

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.proto.common import ResourceName
from viam.proto.component.arm import JointPositions
from viam.resource.base import ResourceBase

from lerobot.robots.robot import Robot

from .config_viam_robot import ViamRobotConfig

class ViamRobotWrapper(Robot):
    """
    Wraps Viam robot components to provide a LeRobot-compatible interface.

    This adapter translates between Viam's component API and LeRobot's
    expected robot interface (get_observation, send_action).
    """

    config_class = ViamRobotConfig
    name = "viam_robot"

    @property
    def arm(self) -> Optional[Arm]:
        """Public accessor for the protected _arm attribute."""
        return self._arm

    def __init__(self, config: ViamRobotConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        super().__init__(config)
        self.dependencies = dependencies
        self.config = config
        self._is_connected = False
        self._arm: Arm = None
        self._cameras: Dict[str, Camera] = {}
        self._logger = config.logger

    def connect(self, calibrate: bool = True):
        """Initialize connections to robot components."""
        for name, resource in self.dependencies.items():
            if isinstance(resource, Arm) and name.name == self.config.arm:
                self._arm = resource
            elif isinstance(resource, Camera) and name.name in self.config.cameras:
                self._cameras[name.name] = resource

        self._is_connected = True

    async def connect_async(self, calibrate: bool = True):
        """Async version of connect - configures arm speed/acceleration."""
        self.connect(calibrate)

        # Set speed and acceleration for responsiveness
        await self._arm.do_command({'set_speed': 180.0})  # Max speed: 180 deg/s
        await self._arm.do_command({'set_acceleration': 500.0})  # Max accel: 500 deg/s²

    def disconnect(self):
        """Disconnect from robot components."""
        self._is_connected = False
        self._arm = None
        self._cameras.clear()

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._is_connected

    def configure(self):
        """Configure robot components if needed."""
        # For now, assume components are pre-configured

    def calibrate(self):
        """Calibrate robot components if needed."""
        # For now, assume components are pre-calibrated

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return True  # Assume always calibrated for now

    def get_observation(self) -> Dict[str, Any]:
        """
        Get current observation from all robot components.

        Returns dict with:
          - "state": joint positions as list of floats
          - "images": dict mapping camera name to image array
        """
        observation = {"state": [], "images": {}}

        # Get joint positions from arms
        try:
            joint_positions = asyncio.run(self._arm.get_joint_positions())
            for i, pos in enumerate(joint_positions.values):
                observation[f"joint_{i}.pos"] = pos
        except asyncio.TimeoutError:
            self._logger.warning("Failed to get joint positions from arm.")

        # Get camera images
        for cam_name, camera in self._cameras.items():
            try:
                image = asyncio.run(camera.get_image())
                observation["images"][cam_name] = image
            except asyncio.TimeoutError:
                self._logger.warning(f"Failed to get image from camera {cam_name}.")

        return observation

    def send_action(self, action: Dict[str, float]) -> Dict[str, float]:
        """
        Send action to robot components (sync version).

        Args:
            action: Dict mapping joint names to target positions

        Returns:
            The action that was sent
        """
        try:
            positions = JointPositions(values=list(action.values()))
            asyncio.run(self._arm.move_to_joint_positions(positions))
        except asyncio.TimeoutError:
            if self._logger:
                self._logger.warning("Failed to send action to arm.")
        return action

    async def send_action_async(self, action: Dict[str, float]) -> Dict[str, float]:
        """
        Send action to robot components (async version).

        Args:
            action: Dict mapping joint names to target positions (in radians)

        Returns:
            The action that was sent
        """
        try:
            # Use non-blocking set_goal_positions command
            # Action values are already in radians from the teleoperator
            positions_radians = list(action.values())
            await self._arm.do_command({
                'command': 'set_goal_positions',
                'positions': positions_radians
            })
        except asyncio.TimeoutError as e:
            if self._logger:
                self._logger.warning("Failed to send action to arm: %s", e)
        return action

    @property
    def _joint_pos_ft(self) -> dict[str, type]:
        # Only support 6-DOF arms for now (including gripper)
        joint_pos_fts = {f"joint_{i}.pos": float for i in range(6)}

        return joint_pos_fts

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        cam_fts = {}
        for cam_name, cam in self._cameras.items():
            img = asyncio.run(cam.get_images())[0][0]
            cam_fts[cam_name] = (img.height, img.width, 3)
        return cam_fts

    @cached_property
    def observation_features(self) -> dict:
        """Return observation features structure."""
        return {**self._joint_pos_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._joint_pos_ft
