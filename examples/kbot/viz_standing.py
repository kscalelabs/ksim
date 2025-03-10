"""Interactive visualization of K-Bot rewards."""

import argparse
import logging

# Import a K-Bot task definition that contains the environment and model
from examples.kbot.standing import KBotStandingConfig, KBotStandingTask
from ksim.utils.reward_visualization.mujoco import MujocoRewardVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--debug", action="store_true")
    args = args.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    task = KBotStandingTask(KBotStandingConfig(num_envs=1))
    reward_visualizer = MujocoRewardVisualizer(task)
    logger.info(
        "Starting visualization - plots will be saved to %s",
        reward_visualizer.viz_config.fig_save_dir,
    )
    logger.info("Open this file in another window to see the live updates")
    reward_visualizer.run()
