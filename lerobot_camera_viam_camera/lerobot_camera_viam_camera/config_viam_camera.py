"""Configuration for ViamRobot."""
from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig



@CameraConfig.register_subclass("viam_camera")
@dataclass
class ViamCameraConfig(CameraConfig):
    """Configuration for ViamCamera."""
    api_key_id: str = ""
    api_key_secret: str = ""
    robot_address: str = ""
    camera_device_name: str = ""
    width: int = 640
    height: int = 480
