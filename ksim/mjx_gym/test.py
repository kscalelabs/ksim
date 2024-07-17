"""Test random actions in a specified environment."""

import argparse
import logging
import os
from typing import Any

import jax as j
import mediapy as media
import yaml

from ksim.mjx_gym.envs import get_env
from ksim.mjx_gym.envs.default_humanoid_env.default_humanoid import (
    DEFAULT_REWARD_PARAMS,
)
from ksim.mjx_gym.utils.rollouts import render_random_rollout

logger = logging.getLogger(__name__)

os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"


def play(config: dict[str, Any], n_steps: int, render_every: int, width: int, height: int) -> None:
    # Load environment
    env = get_env(
        name=config.get("env_name", "default_humanoid"),
        reward_params=config.get("reward_params", DEFAULT_REWARD_PARAMS),
        terminate_when_unhealthy=config.get("terminate_when_unhealthy", True),
        reset_noise_scale=config.get("reset_noise_scale", 1e-2),
        exclude_current_positions_from_observation=config.get("exclude_current_positions_from_observation", True),
        log_reward_breakdown=config.get("log_reward_breakdown", True),
    )
    rng = j.random.PRNGKey(0)
    env.reset(rng)
    print(f"Env loaded: {config.get('env_name', 'could not find environment')}")
    images_thwc = render_random_rollout(env, 200, render_every, width, height)
    print(f"Rolled out {len(images_thwc) * render_every} steps")

    fps = int(1 / env.dt)
    print(f"Writing video to video.mp4 with fps={fps}")
    media.write_video("video.mp4", images_thwc, fps=fps)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run PPO training with specified config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
    parser.add_argument("--use_mujoco", action="store_true", help="Use mujoco instead of mjx for rendering")
    parser.add_argument("--params_path", type=str, default=None, help="Path to the params file")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of steps to rollout")
    parser.add_argument("--render_every", type=int, default=2, help="Render every nth step")
    parser.add_argument("--width", type=int, default=320, help="width in pixels")
    parser.add_argument("--height", type=int, default=240, help="height in pixels")
    args = parser.parse_args()

    # Load config file
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    play(config, args.n_steps, args.render_every, args.width, args.height)
