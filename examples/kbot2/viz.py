"""Interactive visualization of K-Bot rewards."""

import argparse
import logging

import jax.numpy as jnp

# Import a K-Bot task definition that contains the environment and model
from examples.kbot2.standing import KBotV2StandingConfig, KBotV2StandingTask
from examples.kbot2.walking import KBotV2WalkingConfig, KBotV2WalkingTask
from ksim.utils.interactive.base import Keyframe
from ksim.utils.interactive.mujoco_viz import (
    MujocoInteractiveVisualizer,
    MujocoInteractiveVisualizerConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# python -m examples.kbot2.viz
# python -m examples.kbot2.viz --physics-backend mujoco
# Use mjpython if using a Mac, see
# https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--debug", action="store_true")
    args.add_argument("--physics-backend", type=str, default="mjx")
    args.add_argument("--task", type=str, default="standing")
    args = args.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    match args.task:
        case "standing":
            task = KBotV2StandingTask(KBotV2StandingConfig(num_envs=1))
        case "walking":
            task = KBotV2WalkingTask(KBotV2WalkingConfig(num_envs=1))
        case _:
            raise ValueError(f"Invalid task: {args.task}. Must be one of: standing, walking.")

    config = MujocoInteractiveVisualizerConfig(
        physics_backend=args.physics_backend,
        suspended_pos=jnp.array([0, 0, 0.7, 1, 0, 0, 0]),
    )
    interactive_visualizer = MujocoInteractiveVisualizer(task, config=config)

    high_feet_phase_right = jnp.zeros(20)
    high_feet_phase_right = high_feet_phase_right.at[10].set(-0.5)  # hip pitch
    high_feet_phase_right = high_feet_phase_right.at[13].set(-0.75)  # knee roll
    high_feet_phase_left = jnp.zeros(20)
    high_feet_phase_left = high_feet_phase_left.at[15].set(0.5)  # hip pitch
    high_feet_phase_left = high_feet_phase_left.at[18].set(0.75)  # knee

    interactive_visualizer.add_keyframe(
        Keyframe(
            name="right_feet_high",
            joint_positions=high_feet_phase_right,
        )
    )

    interactive_visualizer.add_keyframe(
        Keyframe(
            name="left_feet_high",
            joint_positions=high_feet_phase_left,
        )
    )

    logger.info(
        "Starting visualization - plots will be saved to %s",
        interactive_visualizer.viz_config.fig_save_dir,
    )
    logger.info("Open this file in another window to see the live updates")
    interactive_visualizer.run()
