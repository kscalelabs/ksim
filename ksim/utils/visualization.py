"""Utilities for visualization and rendering."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Sequence

import mediapy as mp
import mujoco
import numpy as np
from jaxtyping import PRNGKeyArray, PyTree
from mujoco import mjx
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


def render_frame_debug(
    default_mj_model: mujoco.MjModel, mjx_data: mjx.Data, config: Any
) -> np.ndarray:
    """Debug utility to render a single frame from mjx_data. For use in debugger only.

    NOTE: This function is to be used for debugging only.

    Example usage from debugger:
    In MjxEnv.unroll_trajectories.env_step, right before returning, you can place a breakpoint:
    >>> jax.debug.breakpoint()
    Now from within jdb, you can call
    >>> render_func(self.default_mj_model, new_mjx_data, self.config)
    and it will render the current step as an image in the debug/render_frames directory.

    NOTE: In order to do the above, you need to:
    1) import this function in mjx_env.py
    >>> from ksim.utils.visualization import render_frame_debug
    2) assign this function to a local variable in env_step
    >>> render_func = render_frame_debug
    3) Step into jdb by placing a breakpoint
    >>> jax.debug.breakpoint()
    4) Then you can call this function in the jdb console
    >>> render_func(self.default_mj_model, new_mjx_data, self.config)

    NOTE: If you try to call this function regulary (i.e. not in the debugger), it will cause a JIT
    compilation error.
    NOTE: I only tested this with num_envs=1
    """
    renderer = mujoco.Renderer(default_mj_model, height=480, width=640)
    # Create fresh MjData for rendering
    render_mj_data = mujoco.MjData(renderer.model)
    render_mj_data.qpos = mjx_data.qpos
    render_mj_data.qvel = mjx_data.qvel
    render_mj_data.mocap_pos = mjx_data.mocap_pos
    render_mj_data.mocap_quat = mjx_data.mocap_quat
    render_mj_data.xfrc_applied = mjx_data.xfrc_applied

    # Update physics state
    mujoco.mj_forward(renderer.model, render_mj_data)

    # Update scene and render
    scene_option = mujoco.MjvOption()
    renderer.update_scene(render_mj_data, camera=0, scene_option=scene_option)
    frame = renderer.render()

    # Save frame for inspection
    render_dir = Path(".").absolute() / "debug" / "render_frames"
    render_dir.mkdir(parents=True, exist_ok=True)
    sim_time = mjx_data.time.item()
    dt = config.dt
    step = int(sim_time / dt)
    time_str = f"{sim_time:.4f}".replace(".", "_")
    save_path = render_dir / f"frame_step_{step}_time_{time_str}s.png"
    latest_save_path = render_dir / "frame_latest.png"
    # Image.fromarray(frame).save(save_path)
    Image.fromarray(frame).save(latest_save_path)
    logger.debug("Saved debug frame to: %s", save_path)

    return None
