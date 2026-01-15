"""Replay methods"""
from logging import getLogger
from typing import Mapping, Optional, Any
import asyncio
import time

from viam.utils import ValueTypes

from lerobot.datasets.lerobot_dataset import LeRobotDataset

LOGGER = getLogger(__name__)

async def replay_episode(
    self,
    dataset_name: str,
    episode_index: int,
    fps: int = 30,
    *,
    extra: Optional[Mapping[str, ValueTypes]] = None,
) -> dict[str, Any]:
    """
    Replays a recorded episode by executing the recorded actions.
    Equivalent to lerobot-replay.
    """
    # Load dataset
    dataset = LeRobotDataset(
        repo_id=dataset_name,
        root=self.dataset_dir,
    )

    # Get episode data
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_index]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_index]
    num_frames = to_idx - from_idx

    LOGGER.info("Replaying episode %d (%d frames) at %d FPS", episode_index, num_frames, fps)

    # Connect robot
    if not self.robot_wrapper.is_connected:
        await self.robot_wrapper.connect()

    # Replay loop
    dt = 1.0 / fps
    start_time = time.time()

    for frame_idx in range(num_frames):
        iter_start = time.time()

        # Get recorded action from dataset
        data_idx = from_idx + frame_idx
        action_tensor = dataset[data_idx]["action"]
        action = {f"joint_{i}": float(v) for i, v in enumerate(action_tensor)}

        # Execute action on robot
        await self.robot_wrapper.send_action(action)

        # Maintain timing
        elapsed = time.time() - iter_start
        if elapsed < dt:
            await asyncio.sleep(dt - elapsed)

    actual_duration = time.time() - start_time

    return {
        "num_frames_replayed": num_frames,
        "duration_s": actual_duration,
    }
