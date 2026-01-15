"""Configuration for ViamRobotWrapper."""
from dataclasses import dataclass, field
from logging import Logger, getLogger

from lerobot.robots import RobotConfig



@RobotConfig.register_subclass("viam_robot")
@dataclass
class ViamRobotConfig(RobotConfig):
    """Configuration for ViamRobotWrapper."""
    arm: str = ""
    # cameras
    cameras: list[str] = field(default_factory=list)
    logger: Logger = getLogger(__name__)
