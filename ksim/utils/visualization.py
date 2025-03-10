"""Utilities for visualization and rendering."""

import logging
import os
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from jaxtyping import Array, PRNGKeyArray, PyTree
from PIL import Image

from ksim.env.base_env import BaseEnv
from ksim.model.formulations import ActorCriticAgent

logger = logging.getLogger(__name__)

from pathlib import Path
from typing import Sequence

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def save_video_with_rewards(
    frames: Sequence[np.ndarray], reward_components: dict[str, Array], fps: float, output_path: Path
):
    num_frames = len(frames)
    reward_keys = list(reward_components.keys())

    # Interpolate rewards to match the number of frames
    reward_data_raw = np.hstack([reward_components[key] for key in reward_keys])
    original_timesteps = np.linspace(0, num_frames, reward_data_raw.shape[0])
    interpolated_timesteps = np.arange(num_frames)
    reward_data = np.vstack(
        [
            interp1d(
                original_timesteps, reward_data_raw[:, i], kind="linear", fill_value="extrapolate"
            )(interpolated_timesteps)
            for i in range(reward_data_raw.shape[1])
        ]
    ).T

    fig, (ax_reward, ax_video) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [1, 2]}, figsize=(8, 8)
    )

    def _update(frame_idx):
        ax_reward.clear()
        ax_video.clear()

        # Plot interpolated rewards
        for i, key in enumerate(reward_keys):
            ax_reward.plot(range(frame_idx + 1), reward_data[: frame_idx + 1, i], label=key)
        # ax_reward.legend(loc="upper right")
        ax_reward.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax_reward.set_xlim(0, num_frames)
        ax_reward.set_ylabel("Reward Components")

        # Display video frame
        ax_video.imshow(frames[frame_idx])
        ax_video.axis("off")

    ani = animation.FuncAnimation(fig, _update, frames=num_frames, interval=1000 / fps)
    ani.save(output_path, writer="ffmpeg", fps=fps)
    plt.close(fig)


def save_trajectory_visualization(
    frames: Sequence[np.ndarray],
    output_dir: Path,
    fps: float,
    *,
    save_frames: bool = True,
    save_video: bool = True,
    rewards: dict[str, Array] | None = None,
) -> None:
    """Save trajectory visualization as individual frames and/or video.

    Args:
        frames: List of frames to save
        output_dir: Directory to save the visualization to
        fps: Frames per second for the video
        save_frames: Whether to save individual frames
        save_video: Whether to save video file
    """
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


def render_and_save_trajectory(
    env: BaseEnv,
    model: ActorCriticAgent,
    variables: PyTree,
    rng: PRNGKeyArray,
    output_dir: Path,
    *,
    num_steps: int,
    width: int = 640,
    height: int = 480,
    camera: int | None = None,
    save_frames: bool = True,
    save_video: bool = True,
) -> None:
    """Render a trajectory and save it as frames and/or video.

    Args:
        env: Environment to render
        model: Model to use for actions
        variables: Model variables
        rng: Random number generator key
        output_dir: Directory to save the visualization to
        num_steps: Number of steps to render
        width: Width of rendered frames
        height: Height of rendered frames
        camera: Camera ID to use for rendering
        save_frames: Whether to save individual frames
        save_video: Whether to save video file
    """
    frames, rewards = env.render_trajectory(
        model=model,
        variables=variables,
        rng=rng,
        num_steps=num_steps,
        width=width,
        height=height,
        camera=camera,
    )

    save_trajectory_visualization(
        frames=frames,
        output_dir=output_dir,
        fps=1 / env.config.ctrl_dt,
        save_frames=save_frames,
        save_video=save_video,
        rewards=rewards,
    )
