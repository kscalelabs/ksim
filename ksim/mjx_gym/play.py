"""Defines the CLI for running PPO training with specified config file."""

import argparse
import logging
from typing import Any

import mediapy as media
import numpy as np
import wandb
import yaml
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from ksim.mjx_gym.envs import get_env
from ksim.mjx_gym.envs.default_humanoid_env.default_humanoid import (
    DEFAULT_REWARD_PARAMS,
)
from ksim.mjx_gym.utils.rollouts import render_mjx_rollout, render_mujoco_rollout

logger = logging.getLogger(__name__)


def train(config: dict[str, Any], n_steps: int, render_every: int) -> None:
    wandb.init(
        project=config.get("project_name", "robotic_locomotion_training") + "_test",
        name=config.get("experiment_name", "ppo-training") + "_test",
    )

    # Load environment
    env = get_env(
        name=config.get("env_name", "default_humanoid"),
        reward_params=config.get("reward_params", DEFAULT_REWARD_PARAMS),
        terminate_when_unhealthy=config.get("terminate_when_unhealthy", True),
        reset_noise_scale=config.get("reset_noise_scale", 1e-2),
        exclude_current_positions_from_observation=config.get("exclude_current_positions_from_observation", True),
        log_reward_breakdown=config.get("log_reward_breakdown", True),
    )

    logger.info(
        "Loaded environment %s with env.observation_size: %s and env.action_size: %s",
        config.get("env_name", ""),
        env.observation_size,
        env.action_size,
    )

    # Loading params
    if args.params_path is not None:
        model_path = args.params_path
    else:
        model_path = "weights/" + config.get("project_name", "model") + ".pkl"
    params = model.load_params(model_path)

    def normalize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x

    if config.get("normalize_observations", False):
        normalize = (
            running_statistics.normalize
        )  # NOTE: very important to keep training & test normalization consistent
    policy_network = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size, preprocess_observations_fn=normalize
    )
    inference_fn = ppo_networks.make_inference_fn(policy_network)(params)
    print(f"Loaded params from {model_path}")

    # rolling out a trajectory
    if args.use_mujoco:
        images_thwc = render_mujoco_rollout(env, inference_fn, n_steps, render_every)
    else:
        images_thwc = render_mjx_rollout(env, inference_fn, n_steps, render_every)
    print(f"Rolled out {len(images_thwc)} steps")

    # render the trajectory
    images_tchw = np.transpose(images_thwc, (0, 3, 1, 2))

    fps = int(1 / env.dt)
    print(f"Writing video to video.mp4 with fps={fps}")
    media.write_video("video.mp4", images_thwc, fps=fps)
    video = wandb.Video(images_tchw, fps=fps, format="mp4")
    wandb.log({"video": video})


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run PPO training with specified config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
    parser.add_argument("--use_mujoco", type=bool, default=False, help="Use mujoco instead of mjx for rendering")
    parser.add_argument("--params_path", type=str, default=None, help="Path to the params file")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of steps to rollout")
    parser.add_argument("--render_every", type=int, default=2, help="Render every nth step")
    args = parser.parse_args()

    # Load config file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    train(config, args.n_steps, args.render_every)
