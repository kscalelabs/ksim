"""Interactive visualization of K-Bot rewards."""

import argparse
import logging

# Import a K-Bot task definition that contains the environment and model
from examples.kbot.standing import KBotStandingConfig, KBotStandingTask
from ksim.utils.interactive.mujoco import (
    MujocoInteractiveVisualizer,
    MujocoInteractiveVisualizerConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# python -m examples.kbot.viz_standing
# python -m examples.kbot.viz_standing --physics-backend mujoco
# Use mjpython if using a Mac, see
# https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--debug", action="store_true")
    args.add_argument("--physics-backend", type=str, default="mjx")
    args = args.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    task = KBotStandingTask(KBotStandingConfig(num_envs=1))
    config = MujocoInteractiveVisualizerConfig(physics_backend=args.physics_backend)
    interactive_visualizer = MujocoInteractiveVisualizer(task, config=config)
    logger.info(
        "Starting visualization - plots will be saved to %s",
        interactive_visualizer.viz_config.fig_save_dir,
    )
    logger.info("Open this file in another window to see the live updates")
    interactive_visualizer.run()
