"""Interactive visualization of K-Bot rewards."""

import argparse
import logging

import colorlogging

from examples.kbot.standing import KBotStandingConfig, KBotStandingTask
from ksim.utils.interactive.mujoco_viz import (
    MujocoInteractiveVisualizer,
    MujocoInteractiveVisualizerConfig,
)

logger = logging.getLogger(__name__)

colorlogging.configure()


# python -m examples.kbot.viz
# python -m examples.kbot.viz --physics-backend mujoco
# Use mjpython if using a Mac, see
# https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--debug", action="store_true")
    args.add_argument("--physics-backend", type=str, default="mjx")
    args = args.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    task = KBotStandingTask.get_task(
        KBotStandingConfig(
            num_envs=1,
            min_action_latency=0.0,
            max_action_latency=0.0,
        )
    )
    config = MujocoInteractiveVisualizerConfig(physics_backend=args.physics_backend)
    interactive_visualizer = MujocoInteractiveVisualizer(task, config=config)
    logger.info(
        "Starting visualization - plots will be saved to %s",
        interactive_visualizer.viz_config.fig_save_dir,
    )
    logger.info("Open this file in another window to see the live updates")
    interactive_visualizer.run()
