"""Policy methods"""
from logging import getLogger
from typing import Mapping, Optional, Any
import asyncio
import time
import uuid

import torch

from viam.utils import ValueTypes

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy

LOGGER = getLogger(__name__)

async def load_policy(
    self,
    policy_repo_id: str,
    *,
    extra: Optional[Mapping[str, ValueTypes]] = None,
) -> dict[str, Any]:
    """
    Load a policy from HuggingFace Hub or local path.
    Equivalent to --pretrained-policy-path in lerobot-eval.
    """
    LOGGER.info("Loading policy from %s...", policy_repo_id)

    policy = PreTrainedPolicy.from_pretrained(policy_repo_id)
    policy_id = str(uuid.uuid4())
    self.loaded_policies[policy_id] = policy

    # Detect policy type from class name
    policy_type = type(policy).__name__.replace("Policy", "").lower()

    LOGGER.info("Loaded policy %s (%s) from %s", policy_id, policy_type, policy_repo_id)

    return {
        "policy_id": policy_id,
        "policy_type": policy_type,
    }

async def run_policy_episode(
    self,
    policy_id: str,
    task: str,
    max_steps: int = 1000,
    fps: int = 30,
    record_to_dataset: bool = False,
    dataset_name: str = "",
    episode_index: int = 0,
    *,
    extra: Optional[Mapping[str, ValueTypes]] = None,
) -> dict[str, Any]:
    """
    Run one episode using the loaded policy.
    Equivalent to one episode in lerobot-eval.
    """
    policy = self.loaded_policies.get(policy_id)
    if not policy:
        raise ValueError(f"Unknown policy: {policy_id}")

    # Optional: setup dataset for recording
    dataset = None
    if record_to_dataset and dataset_name:
        dataset = LeRobotDataset(
            repo_id=dataset_name,
            root=self.dataset_dir,
        )

    # Connect robot
    if not self.robot_wrapper.is_connected:
        await self.robot_wrapper.connect()

    # Policy execution loop
    dt = 1.0 / fps
    start_time = time.time()
    step = 0

    LOGGER.info("Running policy episode (max %d steps)...", max_steps)

    while step < max_steps:
        iter_start = time.time()

        # Get observation
        observation = await self.robot_wrapper.get_observation()

        # Convert observation to policy format
        obs_tensor = {
            "observation.state": torch
                    .tensor(observation.get("state", []), dtype=torch.float32)
                    .unsqueeze(0),
        }
        for cam_name, cam_image in observation.get("images", {}).items():
            # Convert image to tensor format expected by policy
            obs_tensor[f"observation.images.{cam_name}"] = cam_image

        # Policy inference
        with torch.no_grad():
            action = policy.select_action(obs_tensor)

        # Convert action tensor to dict
        action_dict = {f"joint_{i}": float(v) for i, v in enumerate(action.squeeze().tolist())}

        # Record if requested
        if dataset:
            frame = {
                "observation.state": observation.get("state", []),
                "action": list(action_dict.values()),
                "episode_index": episode_index,
                "frame_index": step,
                "timestamp": time.time(),
            }
            for cam_name, cam_image in observation.get("images", {}).items():
                frame[f"observation.images.{cam_name}"] = cam_image
            frame["task"] = task
            dataset.add_frame(frame)

        # Execute action
        await self.robot_wrapper.send_action(action_dict)

        # Maintain timing
        elapsed = time.time() - iter_start
        if elapsed < dt:
            await asyncio.sleep(dt - elapsed)

        step += 1

    duration_s = time.time() - start_time

    # Consolidate dataset if recording
    episode_path = ""
    if dataset:
        dataset.save_episode()
        episode_path = f"{self.dataset_dir}/{dataset_name}/episode_{episode_index:06d}"

    LOGGER.info("Policy episode completed: %d steps in %.2fs", step, duration_s)

    return {
        "num_steps": step,
        "duration_s": duration_s,
        "episode_path": episode_path,
    }
