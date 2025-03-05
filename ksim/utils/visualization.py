"""Utilities for visualization and rendering."""

import logging
import os
import time
from pathlib import Path
from typing import Sequence

import mediapy as mp
import numpy as np
from jaxtyping import PRNGKeyArray, PyTree
from PIL import Image

from ksim.env.base_env import BaseEnv
from ksim.model.formulations import ActorCriticAgent

logger = logging.getLogger(__name__)


def save_trajectory_visualization(
    frames: Sequence[np.ndarray],
    output_dir: Path,
    fps: float,
    *,
    save_frames: bool = True,
    save_video: bool = True,
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
        mp.write_video(output_dir / "trajectory.mp4", frames, fps=fps)
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
    frames = env.render_trajectory(
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
    )
