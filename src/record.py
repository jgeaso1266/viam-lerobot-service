"""Recording methods"""
from logging import getLogger
from typing import Mapping, Optional, Any, Dict

import asyncio
import time
import uuid

from viam.components.arm import Arm
from viam.components.input import Controller
from viam.utils import ValueTypes

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy


LOGGER = getLogger(__name__)

class RecordingSource:
    """Enumeration of recording sources."""
    UNSPECIFIED = 0
    TELEOPERATION = 1
    POLICY = 2

async def start_recording(
    self,
    dataset_name: str,
    *,
    extra: Optional[Mapping[str, ValueTypes]] = None,
) -> str:
    """Begin a new recording session for the specified dataset."""
    session_id = str(uuid.uuid4())
    self.recording_sessions[session_id] = {
        "dataset_name": dataset_name,
        "start_time": time.time(),
        "extra": extra or {},
    }
    LOGGER.info("Started recording session %s for dataset %s", session_id, dataset_name)
    return session_id

async def stop_recording(
    self,
    session_id: str,
    *,
    extra: Optional[Mapping[str, ValueTypes]] = None,
) -> float:
    """End the current recording session and save the data."""
    if session_id not in self.recording_sessions:
        raise ValueError(f"Recording session {session_id} not found")

    session = self.recording_sessions.pop(session_id)
    duration = time.time() - session["start_time"]

    # Save any pending episode data
    if self.dataset is not None and self.dataset.episode_buffer is not None:
        self.dataset.save_episode()

    LOGGER.info("Stopped recording session %s, duration: %.2fs", session_id, duration)
    return duration

async def record_episode(
    self,
    dataset_name: str,
    episode_index: int,
    warmup_time_s: int = 0,
    episode_time_s: int = 0,
    reset_time_s: int = 0,
    source: int = RecordingSource.TELEOPERATION,
    fps: int = 30,
    task: str = "",
    *,
    extra: Optional[Mapping[str, ValueTypes]] = None,
) -> dict[str, Any]:
    """
    Records a single episode via teleoperation or policy.
    Equivalent to one iteration of lerobot-record.
    """
    # Lazy-load dataset
    if self.dataset is None or self.dataset.repo_id != dataset_name:
        self.dataset = LeRobotDataset(
            repo_id=dataset_name,
            root=self.dataset_dir,
        )

    # Connect robot if needed
    if not self.robot_wrapper.is_connected:
        LOGGER.info("Connecting to robot...")
        await self.robot_wrapper.connect()

    # Warmup period
    if warmup_time_s > 0:
        LOGGER.info("Warmup: %ds (move robot to starting position)", warmup_time_s)
        await asyncio.sleep(warmup_time_s)

    # Recording phase
    LOGGER.info("Recording episode %d for %ds...", episode_index, episode_time_s)
    num_frames = await self._record_loop(
        episode_index=episode_index,
        duration_s=episode_time_s,
        fps=fps,
        source=source,
        task=task,
    )

    # Consolidate episode (write Parquet, encode MP4s)
    LOGGER.info("Consolidating episode %d...", episode_index)
    self.dataset.save_episode()

    # Reset period
    if reset_time_s > 0:
        LOGGER.info("Reset: %ds (return robot to home)", reset_time_s)
        await asyncio.sleep(reset_time_s)

    episode_path = f"{self.dataset_dir}/{dataset_name}/episode_{episode_index:06d}"

    return {
        "num_frames": num_frames,
        "actual_duration_s": num_frames / fps,
        "episode_path": episode_path,
    }

async def _record_loop(
    self,
    episode_index: int,
    duration_s: int,
    fps: int,
    source: int,
    task: str,
) -> int:
    """Inner recording loop that captures frames at specified FPS."""
    dt = 1.0 / fps
    start_time = time.time()
    frame_index = 0

    # Get teleoperation or policy source
    if source == RecordingSource.TELEOPERATION:
        action_source = self.teleop_device
    else:
        action_source = self.policy

    while time.time() - start_time < duration_s:
        iter_start = time.time()

        # Get observation from robot
        observation = await self.robot_wrapper.get_observation()

        # Get action from teleop or policy
        if source == RecordingSource.TELEOPERATION \
            and action_source is not None \
            and isinstance(action_source, (Arm, Controller)):
            action = await get_teleop_action(action_source)
        elif source == RecordingSource.POLICY \
            and action_source is not None \
            and isinstance(action_source, PreTrainedPolicy):
            action = action_source.select_action(observation)
        else:
            # No action source, just record observations
            action = {}

        # Build frame
        frame = {
            "observation.state": observation.get("state", []),
            "action": action if isinstance(action, list) 
                        else list(action.values()) if action else [],
            "episode_index": episode_index,
            "frame_index": frame_index,
            "timestamp": time.time(),
        }

        # Add camera images
        for cam_name, cam_image in observation.get("images", {}).items():
            frame[f"observation.images.{cam_name}"] = cam_image

        # Add to dataset
        frame["task"] = task
        self.dataset.add_frame(frame)

        # Execute action on robot (if available)
        if action:
            await self.robot_wrapper.send_action(action)

        # Maintain timing
        elapsed = time.time() - iter_start
        if elapsed < dt:
            await asyncio.sleep(dt - elapsed)

        frame_index += 1

    return frame_index

async def get_teleop_action(teleop_device: Arm | Controller) -> Dict[str, float]:
    """
    Get action from teleoperation device.

    Returns:
        Dict mapping joint/action names to values:
        - For Arm: absolute joint positions (e.g., {"joint_0.pos": 0.5, ...})
        - For Controller: delta values (e.g., {"delta_x": 0.01, "delta_y": 0.0, ...})
    """
    if isinstance(teleop_device, Arm):
        # For arm teleop: return absolute joint positions to mirror
        joint_positions = await teleop_device.get_joint_positions()
        return {f"joint_{i}.pos": pos for i, pos in enumerate(joint_positions.values)}

    elif isinstance(teleop_device, Controller):
        # For controller teleop: return delta values for end-effector movement
        events = await teleop_device.get_events()

        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1  # STAY = 1

        for event in events:
            if event.control == "AbsoluteX":
                delta_x = event.value * 0.01
            elif event.control == "AbsoluteY":
                delta_y = event.value * 0.01
            elif event.control == "AbsoluteRZ":
                delta_z = event.value * 0.01
            elif event.control == "ButtonSouth" and event.value > 0:
                gripper_action = 0  # CLOSE
            elif event.control == "ButtonWest" and event.value > 0:
                gripper_action = 2  # OPEN

        return {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "gripper": float(gripper_action),
        }

    return {}
