"""Viam implementation of Robot interface for LeRobot."""
import asyncio
from functools import cached_property
from typing import Any, Dict
import logging

from viam.components.arm import Arm
from viam.proto.component.arm import JointPositions
from viam.robot.client import RobotClient

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

        self._api_key_id = config.api_key_id
        self._api_key_secret = config.api_key_secret
        self._robot_address = config.robot_address

        self._robot_device_name = config.robot_device_name
        self._robot_device: Arm = None
        self._num_joints = config.num_joints

        self._camera_device_configs = config.cameras
        self._cameras = make_cameras_from_configs(
            self._camera_device_configs
        )

        self._machine = None

    def connect(self, calibrate: bool = True):
        """Initialize connections to robot components."""
        self._machine = asyncio.run(viam_connect(
            api_key_value=self._api_key_secret,
            api_key_id=self._api_key_id,
            robot_address=self._robot_address
        ))

        self._robot_device: Arm = Arm.from_robot(self._machine, self._robot_device_name)

        if not self._robot_device:
            raise ValueError(f"Arm '{self._robot_device_name}' not found on robot.")

        self._is_connected = True

    def disconnect(self):
        """Disconnect from robot components."""
        asyncio.run(self._machine.close())
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

        Returns dict with:
          - "state": joint positions as list of floats
          - "images": dict mapping camera name to image array
        """
        observation: Dict[str, Any] = {}

        # Get joint positions from arms
        try:
            joint_positions = asyncio.run(self._robot_device.get_joint_positions())
            for i, pos in enumerate(joint_positions.values):
                observation[f"joint_{i}.pos"] = pos
        except asyncio.TimeoutError:
            LOGGER.warning("Failed to get joint positions from arm.")

        # Get images from cameras
        for camera_name, camera_device in self._cameras.items():
            try:
                images = asyncio.run(camera_device.get_images())
                if images:
                    # Assuming we want the first image from the list
                    observation[camera_name] = camera_device.async_read()
            except asyncio.TimeoutError:
                LOGGER.warning(f"Failed to get image from camera '{camera_name}'.")

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
            asyncio.run(self._robot_device.move_to_joint_positions(positions))
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
