"""Viam implementation of Robot interface for LeRobot."""
import asyncio
from typing import Any
import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from viam.components.camera import Camera as ViamCameraComponent
from viam.robot.client import RobotClient

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import ColorMode

from .config_viam_camera import ViamCameraConfig

LOGGER = logging.getLogger(__name__)

async def viam_connect(
    api_key_value: str,
    api_key_id: str,
    robot_address: str,
):
    """Connect to a Viam robot using API key credentials."""
    opts = RobotClient.Options.with_api_key(api_key_value, api_key_id)
    return await RobotClient.at_address(robot_address, opts)

class ViamCamera(Camera):
    """
    Wraps Viam camera components to provide a LeRobot-compatible interface.

    This adapter translates between Viam's component API and LeRobot's
    expected robot interface.
    """

    config_class = ViamCameraConfig
    name = "viam_camera"

    def __init__(self, config: ViamCameraConfig):
        super().__init__(config)
        self.id = config.id
        self._is_connected = False

        self._api_key_id = config.api_key_id
        self._api_key_secret = config.api_key_secret
        self._robot_address = config.robot_address

        self._camera_name = config.camera_device_name

        self._machine = None
        self._camera: ViamCameraComponent = None

    def connect(self, warmup: bool = True):
        """Initialize connections to camera components."""
        self._machine = asyncio.run(viam_connect(
            api_key_value=self._api_key_secret,
            api_key_id=self._api_key_id,
            robot_address=self._robot_address
        ))

        self._camera = ViamCameraComponent.from_robot(self._machine, self._camera_name)

        if not self._camera:
            raise ValueError(f"Camera '{self._camera_name}' not found on robot.")

        self._is_connected = True

        props = asyncio.run(self._camera.get_properties())
        self.fps = props.frame_rate
        self.width = props.intrinsic_parameters.width_px
        self.height = props.intrinsic_parameters.height_px

        if warmup:
            # Warm up camera by capturing an initial image
            try:
                asyncio.run(self._camera.get_images())
            except asyncio.TimeoutError:
                LOGGER.warning("Failed to warm up camera.")

    def disconnect(self):
        """Disconnect from camera components."""
        asyncio.run(self._machine.close())
        self._is_connected = False
        self._camera_name = None

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._is_connected

    def configure(self):
        """Configure robot components if needed."""
        # For now, assume components are pre-configured

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Detect available Viam cameras connected to the robot.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains information about a detected camera.
        """
        # Note: Viam does not provide a direct way to list cameras without connecting to a robot.
        # This method would typically require robot connection details.
        raise NotImplementedError("Camera detection requires robot connection details.")

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """Capture and return a single frame from the camera.

        Args:
            color_mode: Desired color mode for the output frame. If None,
                        uses the camera's default color mode.
        Returns:
            NDArray[Any]: Captured image frame as a NumPy array.
        """
        images = asyncio.run(self._camera.get_images())
        if not images or len(images) == 0:
            raise RuntimeError("Failed to capture image from camera.")

        # For simplicity, return the first image
        frame_bytes = images[0][0].data

        # Convert bytes to NumPy array
        frame_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
        image = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)

        if color_mode == ColorMode.RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def async_read(self, timeout_ms: float = 1000) -> NDArray[Any]:
        """Asynchronously capture and return a single frame from the camera.

        Args:
            timeout_ms: Maximum time to wait for a frame in milliseconds.
                        Defaults to 1000ms.
        Returns:
            NDArray[Any]: Captured image frame as a NumPy array.
        """
        try:
            images = asyncio.run(asyncio.wait_for(
                self._camera.get_images(),
                timeout=timeout_ms / 1000.0
            ))
        except asyncio.TimeoutError:
            raise RuntimeError("Timeout while capturing image from camera.")

        if not images or len(images) == 0:
            raise RuntimeError("Failed to capture image from camera.")

        # For simplicity, return the first image
        frame_bytes = images[0][0].data

        # Convert bytes to NumPy array
        frame_buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
        image = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)

        return image
