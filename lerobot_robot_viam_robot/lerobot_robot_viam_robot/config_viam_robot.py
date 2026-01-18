"""Configuration for ViamRobot."""
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras.configs import CameraConfig



@RobotConfig.register_subclass("viam_robot")
@dataclass
class ViamRobotConfig(RobotConfig):
    """Configuration for ViamRobot."""
    api_key_id: str = ""
    api_key_secret: str = ""
    robot_address: str = ""
    robot_device_name: str = ""
    num_joints: int = 6
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
