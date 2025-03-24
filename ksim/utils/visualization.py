"""Utilities for visualization and rendering."""

__all__ = [
    "save_video_with_rewards",
    "save_trajectory_visualization",
]

import logging
import os
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from flax.core import FrozenDict
from jaxtyping import Array
from matplotlib import animation
from PIL import Image
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


def save_video_with_rewards(
    frames: Sequence[np.ndarray],
    reward_components: FrozenDict[str, Array],
    fps: float,
    output_path: Path,
) -> None:
    """Save video with rewards."""
    num_frames = len(frames)
    reward_keys = list(reward_components.keys())

    # Interpolate rewards to match the number of frames
    reward_data_raw = np.hstack([reward_components[key] for key in reward_keys])
    original_timesteps = np.linspace(0, num_frames, reward_data_raw.shape[0])
    interpolated_timesteps = np.arange(num_frames)
    reward_data = np.vstack(
        [
            interp1d(original_timesteps, reward_data_raw[:, i], kind="linear", fill_value="extrapolate")(
                interpolated_timesteps
            )
            for i in range(reward_data_raw.shape[1])
        ]
    ).T

    fig, (ax_reward, ax_video) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 2]}, figsize=(8, 8))

    def _plot_rewards_with_frames(frame_idx: int) -> list[plt.Artist]:
        ax_reward.clear()
        ax_video.clear()

        for i, key in enumerate(reward_keys):
            ax_reward.plot(range(frame_idx + 1), reward_data[: frame_idx + 1, i], label=key)

        ax_reward.legend(loc="center", bbox_to_anchor=(1, 0.5))
        ax_reward.set_xlim(0, num_frames)
        ax_reward.set_ylabel("Reward Components")

        ax_video.imshow(frames[frame_idx])
        ax_video.axis("off")

        return [ax_reward, ax_video]

    ani = animation.FuncAnimation(fig, _plot_rewards_with_frames, frames=num_frames, interval=1000 / fps)
    ani.save(output_path, writer="ffmpeg", fps=int(fps))
    plt.close(fig)


def save_trajectory_visualization(
    frames: Sequence[np.ndarray],
    output_dir: Path,
    fps: float,
    rewards: FrozenDict[str, Array],
    *,
    save_frames: bool = True,
    save_video: bool = True,
) -> None:
    """Save trajectory visualization as individual frames and/or video."""
    frames_dir = output_dir / "frames"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    logger.info("Saving %d frames to %s", len(frames), frames_dir)

    if save_frames:
        start_time = time.time()
        for i, frame in enumerate(frames):
            frame_path = frames_dir / f"frame_{i:06d}.png"
            Image.fromarray(frame).save(frame_path)
        end_time = time.time()
        logger.info("Time taken for saving frames: %.2f seconds", end_time - start_time)

    if save_video:
        start_time = time.time()
        save_video_with_rewards(frames, rewards, fps, output_dir / "trajectory.mp4")
        end_time = time.time()
        logger.info("Time taken for video creation: %.2f seconds", end_time - start_time)
