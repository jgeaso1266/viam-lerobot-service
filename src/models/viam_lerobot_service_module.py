"""
Viam LeRobot Service Module

This module implements the LeRobot service interface for robot learning
and teleoperation using the LeRobot framework.
"""

import asyncio
import time
from typing import Any, ClassVar, Dict, Mapping, Optional, Sequence, Tuple, cast
from dataclasses import dataclass, field
import os
import uuid

from typing_extensions import Self

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.logging import getLogger
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model
from viam.services.generic import Generic
from viam.utils import ValueTypes

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.scripts.lerobot_teleoperate import teleoperate, TeleoperateConfig

from lerobot_camera_viam_camera.lerobot_camera_viam_camera.config_viam_camera import ViamCameraConfig
from lerobot_robot_viam_robot.lerobot_robot_viam_robot.config_viam_robot import ViamRobotConfig
from lerobot_teleoperator_viam_teleoperator.lerobot_teleoperator_viam_teleoperator.config_viam_teleoperator import ViamTeleoperatorConfig

LOGGER = getLogger(__name__)

@dataclass
class TeleopSession:
    """Tracks state for a teleoperation session."""
    fps: int
    start_time: float = field(default_factory=time.time)
    stop_requested: bool = False
    stopped: bool = False
    duration_s: float = 0.0
    task: Optional[asyncio.Task] = field(default=None, repr=False)

    async def wait_for_stop(self, timeout: float = 30.0):
        """Wait until the session is fully stopped."""
        try:
            await asyncio.wait_for(self._wait_stopped(), timeout=timeout)
        except asyncio.TimeoutError:
            LOGGER.warning("Timeout waiting for session to stop, cancelling task")
            if self.task and not self.task.done():
                self.task.cancel()
            self.stopped = True
            self.duration_s = time.time() - self.start_time

    async def _wait_stopped(self):
        """Internal wait loop."""
        while not self.stopped:
            await asyncio.sleep(0.1)

class MyLeRobotService(Generic, EasyResource):
    """
    MyLeRobotService if a generic service for robot learning
    and teleoperation using the LeRobot framework.

    This service provides:
    - Episode recording (equivalent to lerobot-record)
    - Episode replay (equivalent to lerobot-replay)
    - Teleoperation control (equivalent to lerobot-teleoperate)
    - Policy loading and inference (equivalent to lerobot-eval)
    """
    MODEL: ClassVar[Model] = "jalen:lerobot:lerobot"

    def __init__(self, name: str):
        super().__init__(name)
        self.config: Optional[ComponentConfig] = None
        self.dataset: Optional[LeRobotDataset] = None
        self.dataset_dir: str = "./datasets"
        self.policy_dir: str = "./policies"
        self.policy: PreTrainedPolicy = None

        self.robot_config: Optional[ViamRobotConfig] = None
        self.teleop_config: Optional[ViamTeleoperatorConfig] = None

        # Session tracking
        self.recording_sessions: Dict[str, Dict[str, Any]] = {}
        self.teleop_sessions: Dict[str, TeleopSession] = {}
        self.loaded_policies: Dict[str, PreTrainedPolicy] = {}

    @classmethod
    def new(
        cls,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """Factory method to create a new instance of the service."""
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    @classmethod
    def validate_config(
        cls,
        config: ComponentConfig
    ) -> Tuple[Sequence[str], Sequence[str]]:
        """
        Validate the configuration and return dependencies.

        Returns:
            Tuple of (required_deps, optional_deps)
        """
        optional_deps = []
        attrs = config.attributes.fields

        # Collect optional arm dependencies
        if "arm" in attrs:
            optional_deps.append(attrs["arm"].string_value)

        # Collect optional camera dependencies
        if "cameras" in attrs:
            for camera in attrs["cameras"].list_value.values:
                optional_deps.append(camera.string_value)

        if "teleop" in attrs:
            optional_deps.append(attrs["teleop"].string_value)

        return [], optional_deps

    def reconfigure(
            self,
            config: ComponentConfig,
            dependencies: Mapping[ResourceName, ResourceBase]
        ) -> None:
        """Reconfigure the service with new configuration."""
        self.config = config
        attrs = config.attributes.fields

        self.dataset_dir = attrs["dataset_dir"].string_value if "dataset_dir" in attrs \
            else "./datasets"
        self.policy_dir = attrs["policy_dir"].string_value if "policy_dir" in attrs \
            else "./policies"

        arm_name = attrs["arm"].string_value if "arm" in attrs else ""
        teleop_name = attrs["teleop"].string_value if "teleop" in attrs else ""
        camera_names = attrs.get("cameras", {}).list_value.values

        arm = dependencies[Arm.get_resource_name(arm_name)]
        arm = cast(Arm, arm)

        joint_positions = asyncio.run(arm.get_joint_positions())
        num_joints = len(joint_positions.values)

        self.robot_config = ViamRobotConfig(
            robot_device_name=arm_name,
            num_joints=num_joints
        )

        self.teleop_config = ViamTeleoperatorConfig(
            teleop_device_name=teleop_name,
            num_joints=num_joints
        )

        for cam_name in camera_names:
            camera_name = cam_name.string_value
            camera = dependencies[Camera.get_resource_name(camera_name)]
            camera = cast(Camera, camera)

            props = asyncio.run(camera.get_properties())

            camera_config = ViamCameraConfig(
                camera_device_name=camera_name,
                width=props.intrinsic_parameters.width_px,
                height=props.intrinsic_parameters.height_px,
                fps=props.frame_rate
            )
            self.robot_config.cameras[camera_name] = camera_config

        LOGGER.info(
            "LeRobot service configured: dataset_dir=%s, policy_dir=%s",
            self.dataset_dir, self.policy_dir
        )

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        """Handle arbitrary commands for extended functionality."""
        cmd = command.get("command", "")

        if cmd == "start_teleoperation":
            fps = int(command.get("fps", 30))
            teleop_time_s = int(command.get("teleop_time_s", 60))
            display_data = bool(command.get("display_data", False))
            display_ip = command.get("display_ip", None)
            display_port = int(command.get("display_port", 0))
            display_compressed_images = bool(command.get("display_compressed_images", False))

            session_id = str(uuid.uuid4())
            session = TeleopSession(
                fps=fps
            )
            self.teleop_sessions[session_id] = session

            cfg = TeleoperateConfig(
                teleop = self.teleop_config,
                robot = self.robot_config,
                fps = fps,
                teleop_time_s = teleop_time_s,
                display_data = display_data,
                display_ip = display_ip,
                display_port = display_port,
                display_compressed_images = display_compressed_images,
            )

            # Start teleoperation loop in background task
            async def run_teleoperation():
                try:
                    await asyncio.to_thread(teleoperate, cfg)
                except asyncio.CancelledError:
                    LOGGER.info("Teleoperation session %s cancelled.", session_id)
                except Exception as e:
                    LOGGER.error("Teleoperation session %s error: %s", session_id, e)
                finally:
                    session.duration_s = time.time() - session.start_time
                    session.stopped = True

            session.task = asyncio.create_task(run_teleoperation())

            LOGGER.info("Started teleoperation session %s.", session_id)

            return {"session_id": session_id, "active_sessions": list(self.teleop_sessions.keys())}

        elif cmd == "stop_teleoperation":
            session_id = command.get("session_id", "")
            session = self.teleop_sessions.get(session_id)
            if not session:
                raise ValueError(f"Unknown session: {session_id}")

            session.stop_requested = True

            # Cancel the background task if still running
            if session.task and not session.task.done():
                session.task.cancel()

            await session.wait_for_stop()

            duration_s = session.duration_s
            del self.teleop_sessions[session_id]

            LOGGER.info("Stopped teleoperation session %s, duration: %.2fs", session_id, duration_s)
            return {"duration_s": duration_s, "active_sessions": list(self.teleop_sessions.keys())}

        LOGGER.info("Received do_command: %s", command)
        return dict(command)

    async def close(self):
        """Clean up resources when the service is stopped."""
        LOGGER.info("Closing LeRobot service %s", self.name)

        # Cancel any active teleoperation sessions
        for session_id, session in list(self.teleop_sessions.items()):
            if session.task and not session.task.done():
                session.task.cancel()

        # Clear state
        self.recording_sessions.clear()
        self.teleop_sessions.clear()
        self.loaded_policies.clear()
