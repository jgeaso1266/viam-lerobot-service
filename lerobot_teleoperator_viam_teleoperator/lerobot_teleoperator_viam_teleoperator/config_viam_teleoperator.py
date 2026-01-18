"""Configuration for ViamTeleoperator."""
from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("viam_teleoperator")
@dataclass
class ViamTeleoperatorConfig(TeleoperatorConfig):
    """Configuration for ViamTeleoperator."""
    api_key_id: str = ""
    api_key_secret: str = ""
    robot_address: str = ""
    teleop_device_name: str = ""
    num_joints: int = 6
