"""Configuration for ViamTeleoperator."""
from dataclasses import dataclass
from typing import Optional

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("viam_teleoperator")
@dataclass
class ViamTeleoperatorConfig(TeleoperatorConfig):
    """Configuration for ViamTeleoperator.

    Supports two modes:
    - "arm" mode: Leader arm teleoperation returning joint positions
    - "controller" mode: Gamepad/controller returning delta values for end-effector control
    """
    teleop_device_name: str = ""
    teleop_device_type: str = "arm"  # "arm" or "controller"
    num_joints: int = 6  # Only relevant for arm mode

    # Controller-specific configuration
    use_gripper: bool = True  # Include gripper control in output
    axis_x: str = "AbsoluteX"  # Control for delta_x (left stick X)
    axis_y: str = "AbsoluteY"  # Control for delta_y (left stick Y)
    axis_z: str = "AbsoluteRZ"  # Control for delta_z (right stick Y)
    gripper_close_button: str = "ButtonSouth"  # A button (Xbox) / X (PlayStation)
    gripper_open_button: str = "ButtonWest"  # X button (Xbox) / Square (PlayStation)
    delta_scale: float = 10.0  # Movement in mm per full axis deflection
    delta_scale_x: Optional[float] = None  # Per-axis scale override
    delta_scale_y: Optional[float] = None
    delta_scale_z: Optional[float] = None
    axis_deadzone: float = 0.1  # Ignore inputs below this threshold

    def __post_init__(self) -> None:
        if self.teleop_device_type not in ["arm", "controller"]:
            raise ValueError(
                f"`teleop_device_type` is expected to be 'arm' or 'controller', but {self.teleop_device_type} is provided."
            )
