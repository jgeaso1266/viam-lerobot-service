"""
Viam LeRobot Service Module

This module implements the LeRobot service interface for robot learning
and teleoperation using the LeRobot framework.
"""

from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple
import os

from typing_extensions import Self

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.input import Controller
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

from lerobot_robot_viam_robot.config_viam_robot import ViamRobotConfig
from lerobot_robot_viam_robot.viam_robot import ViamRobotWrapper
from lerobot_teleoperator_viam_teleoperator.config_viam_teleoperator import ViamTeleoperatorConfig
from lerobot_teleoperator_viam_teleoperator.viam_teleoperator import ViamTeleopWrapper

from src.teleoperate import TeleopSession, start_teleoperation, stop_teleoperation
from src.record import start_recording, stop_recording
from src.replay import replay_episode
from src.policy import load_policy, run_policy_episode

LOGGER = getLogger(__name__)

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
        self.robot_wrapper: Optional[ViamRobotWrapper] = None
        self.teleop_wrapper: Optional[ViamTeleopWrapper] = None
        self.policy: PreTrainedPolicy = None

        self.arm: Optional[Arm] = None
        self.cameras: List[Camera] = []
        self.teleop_device: Optional[Arm | Controller] = None

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

        if "dataset_dir" in attrs:
            self.dataset_dir = attrs["dataset_dir"].string_value
        else:
            self.dataset_dir = "./datasets"

        if "policy_dir" in attrs:
            self.policy_dir = attrs["policy_dir"].string_value
        else:
            self.policy_dir = "./policies"

        if "arm" in attrs:
            arm_name = attrs["arm"].string_value
            arm_resource_name = Arm.get_resource_name(arm_name)
            if arm_resource_name in dependencies:
                self.arm = dependencies[arm_resource_name]
            else:
                LOGGER.warning("Arm %s not found in dependencies", arm_name)

        if "teleop" in attrs:
            teleop_name = attrs["teleop"].string_value
            teleop_resource_name = Arm.get_resource_name(teleop_name)
            if teleop_resource_name in dependencies:
                self.teleop_device = dependencies[teleop_resource_name]
            else:
                LOGGER.warning("Teleop device %s not found in dependencies", teleop_name)

        if "cameras" in attrs:
            camera_names = [
                cam.string_value for cam in attrs["cameras"].list_value.values
            ]
            for cam_name in camera_names:
                cam_resource_name = ResourceName.from_string(cam_name)
                if cam_resource_name in dependencies:
                    self.cameras.append(dependencies[cam_resource_name])
                else:
                    LOGGER.warning("Camera %s not found in dependencies", cam_name)

        robot_config = ViamRobotConfig(
            arm=arm_name if "arm" in attrs else "",
            cameras=camera_names if "cameras" in attrs else [],
            logger=LOGGER,
        )

        teleop_config = ViamTeleoperatorConfig(
            teleop_device=teleop_name if "teleop" in attrs else "",
            fps=30,
            logger=LOGGER,
        )

        # Create robot wrapper with dependencies
        self.robot_wrapper = ViamRobotWrapper(robot_config, dependencies)
        self.teleop_wrapper = ViamTeleopWrapper(teleop_config, dependencies)

        LOGGER.info(
            "LeRobot service configured: dataset_dir=%s, policy_dir=%s",
            self.dataset_dir, self.policy_dir
        )

    start_teleoperation = start_teleoperation
    stop_teleoperation = stop_teleoperation

    start_recording = start_recording
    stop_recording = stop_recording

    replay_episode = replay_episode

    load_policy = load_policy
    run_policy_episode = run_policy_episode

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        """Handle arbitrary commands for extended functionality."""
        cmd = command.get("command", "")

        if cmd == "list_datasets":
            datasets = []
            if os.path.exists(self.dataset_dir):
                datasets = os.listdir(self.dataset_dir)
            return {"datasets": datasets}

        elif cmd == "push_to_hub":
            dataset_name = command.get("dataset_name", "")
            repo_id = command.get("repo_id", dataset_name)
            # Push dataset to HuggingFace Hub
            dataset = LeRobotDataset(repo_id=dataset_name, root=self.dataset_dir)
            dataset.push_to_hub(repo_id=repo_id)
            return {"status": "success", "repo_id": repo_id}

        elif cmd == "get_dataset_info":
            dataset_name = command.get("dataset_name", "")
            dataset = LeRobotDataset(repo_id=dataset_name, root=self.dataset_dir)
            return {
                "repo_id": dataset.repo_id,
                "num_episodes": dataset.num_episodes,
                "num_frames": len(dataset),
                "fps": dataset.fps,
            }

        elif cmd == "unload_policy":
            policy_id = command.get("policy_id", "")
            if policy_id in self.loaded_policies:
                del self.loaded_policies[policy_id]
                return {"status": "unloaded", "policy_id": policy_id}
            return {"status": "not_found", "policy_id": policy_id}

        elif cmd == "start_teleoperation":
            teleop_device_type = command.get("teleop_device_type", "arm")
            fps = int(command.get("fps", 30))
            session_id = await self.start_teleoperation(teleop_device_type, fps)
            return {"session_id": session_id, "active_sessions": list(self.teleop_sessions.keys())}

        elif cmd == "stop_teleoperation":
            session_id = command.get("session_id", "")
            duration_s = await self.stop_teleoperation(session_id)
            return {"duration_s": duration_s, "active_sessions": list(self.teleop_sessions.keys())}

        elif cmd == "start_recording":
            dataset_name = command.get("dataset_name", "")
            session_id = await self.start_recording(dataset_name)
            return {
                "session_id": session_id, 
                "active_sessions": list(self.recording_sessions.keys())
            }

        elif cmd == "stop_recording":
            session_id = command.get("session_id", "")
            duration_s = await self.stop_recording(session_id)
            return {
                "duration_s": duration_s,
                "active_sessions": list(self.recording_sessions.keys())
            }

        elif cmd == "replay_episode":
            dataset_name = command.get("dataset_name", "")
            episode_index = int(command.get("episode_index", 0))
            fps = int(command.get("fps", 30))
            result = await self.replay_episode(
                dataset_name,
                episode_index,
                fps=fps,
            )
            return result

        elif cmd == "load_policy":
            policy_repo_id = command.get("policy_repo_id", "")
            result = await self.load_policy(policy_repo_id)
            return result

        elif cmd == "run_policy_episode":
            policy_id = command.get("policy_id", "")
            task = command.get("task", "")
            max_steps = int(command.get("max_steps", 1000))
            fps = int(command.get("fps", 30))
            record_to_dataset = bool(command.get("record_to_dataset", False))
            dataset_name = command.get("dataset_name", "")
            episode_index = int(command.get("episode_index", 0))
            result = await self.run_policy_episode(
                policy_id,
                task,
                max_steps=max_steps,
                fps=fps,
                record_to_dataset=record_to_dataset,
                dataset_name=dataset_name,
                episode_index=episode_index,
            )
            return result

        LOGGER.info("Received do_command: %s", command)
        return dict(command)

    async def close(self):
        """Clean up resources when the service is stopped."""
        LOGGER.info("Closing LeRobot service %s", self.name)

        # Stop any active teleoperation sessions
        for session_id in list(self.teleop_sessions.keys()):
            try:
                await self.stop_teleoperation(session_id)
            except Exception as e:
                LOGGER.warning("Error stopping teleoperation %s: %s", session_id, e)

        # Disconnect robot
        if self.robot_wrapper and self.robot_wrapper.is_connected:
            await self.robot_wrapper.disconnect()

        # Clear state
        self.recording_sessions.clear()
        self.teleop_sessions.clear()
        self.loaded_policies.clear()
