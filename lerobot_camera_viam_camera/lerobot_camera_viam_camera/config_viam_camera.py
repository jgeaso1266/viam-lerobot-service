"""Configuration for ViamRobot."""
from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.configs import ColorMode


@CameraConfig.register_subclass("viam_camera")
@dataclass
class ViamCameraConfig(CameraConfig):
    """Configuration for ViamCamera."""
    camera_device_name: str = ""
    width: int = 640
    height: int = 480
    color_mode: ColorMode = ColorMode.RGB

    def __post_init__(self) -> None:
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )
