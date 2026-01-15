"""Configuration for ViamTeleopWrapper."""
from dataclasses import dataclass
from logging import Logger, getLogger

from lerobot.teleoperators import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("viam_teleoperator")
@dataclass
class ViamTeleoperatorConfig(TeleoperatorConfig):
    """Configuration for ViamTeleopWrapper."""
    teleop_device: str = ""
    # Number of joints (6 DOF by default including gripper)
    num_joints: int = 6
    # FPS for streaming
    fps: int = 30
    logger: Logger = getLogger(__name__)
