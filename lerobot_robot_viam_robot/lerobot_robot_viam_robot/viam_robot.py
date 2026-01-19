"""Viam implementation of Robot interface for LeRobot."""
import asyncio
import math
import os
from functools import cached_property
from typing import Any, Dict

from viam.components.arm import Arm
from viam.proto.component.arm import JointPositions
from viam.robot.client import RobotClient
from viam.logging import logging

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot

from .config_viam_robot import ViamRobotConfig

LOGGER = logging.getLogger(__name__)

async def viam_connect(
    api_key_value: str,
    api_key_id: str,
    robot_address: str,
):
    """Connect to a Viam robot using API key credentials."""
    opts = RobotClient.Options.with_api_key(api_key_value, api_key_id)
    return await RobotClient.at_address(robot_address, opts)

class ViamRobot(Robot):
    """
    Wraps Viam robot components to provide a LeRobot-compatible interface.

    This adapter translates between Viam's component API and LeRobot's
    expected robot interface (get_observation, send_action).
    """

    config_class = ViamRobotConfig
    name = "viam_robot"

    def __init__(self, config: ViamRobotConfig):
        super().__init__(config)
        self.id = config.id
        self._is_connected = False
        self._robot_device_name = config.robot_device_name
        self._robot_device: Arm = None
        self._num_joints = config.num_joints
        self._camera_device_configs = config.cameras
        self.cameras = make_cameras_from_configs(self._camera_device_configs)
        self._machine = None
        self._loop = asyncio.new_event_loop()

    def connect(self, calibrate: bool = True):
        """Initialize connections to robot components."""
        self._machine = self._loop.run_until_complete(viam_connect(
            api_key_value=os.environ.get("VIAM_API_KEY", ""),
            api_key_id=os.environ.get("VIAM_API_KEY_ID", ""),
            robot_address=os.environ.get("VIAM_MACHINE_FQDN", "")
        ))

        self._robot_device: Arm = Arm.from_robot(self._machine, self._robot_device_name)

        if not self._robot_device:
            raise ValueError(f"Arm '{self._robot_device_name}' not found on robot.")

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True

    def disconnect(self):
        """Disconnect from robot components."""
        if self._machine:
            self._loop.run_until_complete(self._machine.close())

        # Cancel any remaining tasks on our loop
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()

        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
        self._robot_device = None

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

        Returns flat dict with:
          - joint positions
          - dict mapping camera name to image array
        """
        observation: Dict[str, Any] = {}

        # Get joint positions from arms
        try:
            joint_positions = self._loop.run_until_complete(self._robot_device.get_joint_positions())
            # joint positions are in degrees, convert to radians
            joint_positions_rad = [math.radians(pos) for pos in joint_positions.values]
            LOGGER.debug(f"Robot joint positions: {joint_positions_rad}")
            for i, pos in enumerate(joint_positions_rad):
                observation[f"joint_{i}.pos"] = pos
        except asyncio.TimeoutError:
            LOGGER.warning("Failed to get joint positions from arm.")

        # Get images from cameras
        for camera_name, camera_device in self.cameras.items():
            try:
                observation[camera_name] = camera_device.async_read()
                LOGGER.debug(f"Got image from camera '{camera_name}'.")
            except asyncio.TimeoutError:
                LOGGER.warning("Failed to get image from camera '%s'.", camera_name)

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
            ## joint positions are in radians, switch to degrees for Viam
            actions_deg = [math.floor(math.degrees(pos)) for pos in action.values()]
            positions = JointPositions(values=actions_deg)
            LOGGER.debug(f"Sending joint positions: {positions.values}")
            start_time = asyncio.get_event_loop().time()
            self._loop.run_until_complete(self._robot_device.move_to_joint_positions(positions))
            end_time = asyncio.get_event_loop().time()
            LOGGER.debug(f"Action sent in {end_time - start_time:.4f} seconds.")
        except asyncio.TimeoutError:
            LOGGER.warning("Failed to send action to arm.")
        return action

    @property
    def _joint_pos_ft(self) -> dict[str, type]:
        """Return joint position features based on config."""
        return {f"joint_{i}.pos": float for i in range(self._num_joints)}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Return camera features based on config (height, width from CameraConfig)."""
        return {
            cam_name: (cam_cfg.height, cam_cfg.width, 3)
            for cam_name, cam_cfg in self._camera_device_configs.items()
        }

    @cached_property
    def observation_features(self) -> dict:
        """Return observation features structure."""
        return {**self._joint_pos_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._joint_pos_ft
