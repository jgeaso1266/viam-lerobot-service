"""Configuration for ViamRobot."""
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras.configs import CameraConfig



@RobotConfig.register_subclass("viam_robot")
@dataclass
class ViamRobotConfig(RobotConfig):
    """Configuration for ViamRobot."""
    robot_device_name: str = ""
    num_joints: int = 6
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("viam_robot_ee")
@dataclass
class ViamRobotEndEffectorConfig(ViamRobotConfig):
    """Configuration for ViamRobotEndEffector (end-effector delta control).

    Inherits all fields from ViamRobotConfig. Use this with controller
    teleoperation to accept delta_x, delta_y, delta_z actions.
    """
    # Safety bounds for end-effector position (min/max for x, y, z)
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-1000.0, -1000.0, -1000.0],
            "max": [1000.0, 1000.0, 1000.0],
        }
    )
    # Maximum step size per axis (clamps delta values)
    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 10.0,
            "y": 10.0,
            "z": 10.0,
        }
    )
    # Gripper configuration
    use_gripper: bool = False
    max_gripper_pos: float = 50.0
    min_gripper_pos: float = 5.0
