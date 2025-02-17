"""Training script for walking using Brax PPO."""

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import xax
from brax.training.agents.ppo import train as ppo
from omegaconf import OmegaConf

from ksim.commands import AngularVelocityCommand, LinearVelocityCommand
from ksim.env.brax import KScaleEnv, KScaleEnvConfig
from ksim.observation import (
    BaseAngularVelocityObservation,
    BaseLinearVelocityObservation,
    BaseOrientationObservation,
    BasePositionObservation,
    JointPositionObservation,
    JointVelocityObservation,
)
from ksim.resets import XYPositionResetBuilder
from ksim.rewards import LinearVelocityZPenalty
from ksim.terminations import IllegalContactTerminationBuilder

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class Config(KScaleEnvConfig):
    """Configuration for PPO training."""

    model_name: str = xax.field(value="kbot-v1")

    output_dir: str = xax.field("walking_brax", help="The directory to save the training results.")
    num_timesteps: int = xax.field(50_000_000, help="The number of timesteps to train for.")
    num_evals: int = xax.field(10, help="The number of evaluation episodes to run.")
    reward_scaling: float = xax.field(10.0, help="The scaling factor for the reward.")
    episode_length: int = xax.field(1000, help="The length of each episode.")
    normalize_observations: bool = xax.field(True, help="Whether to normalize the observations.")
    action_repeat: int = xax.field(1, help="The number of times to repeat the action.")
    unroll_length: int = xax.field(5, help="The number of steps to unroll the model for.")
    num_minibatches: int = xax.field(32, help="The number of minibatches to use.")
    num_updates_per_batch: int = xax.field(4, help="The number of updates per batch.")
    discounting: float = xax.field(0.97, help="The discount factor.")
    learning_rate: float = xax.field(3e-4, help="The learning rate.")
    entropy_cost: float = xax.field(1e-2, help="The entropy cost.")
    num_envs: int = xax.field(4096, help="The number of environments to run in parallel.")
    batch_size: int = xax.field(2048, help="The batch size.")
    seed: int = xax.field(1, help="The seed for the training.")
    min_reward: float = xax.field(0.0, help="The minimum reward.")
    max_reward: float = xax.field(8000.0, help="The maximum reward.")


def main() -> None:
    # Load config
    config = OmegaConf.structured(Config)

    # Allow command line overrides
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, cli_config)

    # Create environment
    env = KScaleEnv(
        config,
        terminations=[
            IllegalContactTerminationBuilder(
                body_names=[
                    "shoulder",
                    "shoulder_2",
                    "hand_shell",
                    "hand_shell_2",
                    "leg0_shell",
                    "leg0_shell_2",
                ],
            ),
        ],
        resets=[
            XYPositionResetBuilder(),
        ],
        rewards=[
            LinearVelocityZPenalty(scale=-1.0),
        ],
        observations=[
            BasePositionObservation(noise=0.01),
            BaseOrientationObservation(noise=0.01),
            BaseLinearVelocityObservation(noise=0.01),
            BaseAngularVelocityObservation(noise=0.01),
            JointPositionObservation(noise=0.01),
            JointVelocityObservation(noise=0.01),
        ],
        commands=[
            LinearVelocityCommand(
                x_scale=1.0,
                y_scale=0.0,
                switch_prob=0.02,
                zero_prob=0.3,
            ),
            AngularVelocityCommand(
                scale=1.0,
                switch_prob=0.02,
                zero_prob=0.8,
            ),
        ],
    )

    # Setup training progress tracking
    xdata, ydata = [], []

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
        xdata.append(num_steps)
        ydata.append(metrics["eval/episode_reward"])

        plt.clf()
        plt.xlim([0, config.num_timesteps])
        plt.ylim([config.min_reward, config.max_reward])
        plt.xlabel("# environment steps")
        plt.ylabel("reward per episode")
        plt.plot(xdata, ydata)
        plt.savefig(Path(config.output_dir) / "training_progress.png")

    # Configure PPO training
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=config.num_timesteps,
        num_evals=config.num_evals,
        reward_scaling=config.reward_scaling,
        episode_length=config.episode_length,
        normalize_observations=config.normalize_observations,
        action_repeat=config.action_repeat,
        unroll_length=config.unroll_length,
        num_minibatches=config.num_minibatches,
        num_updates_per_batch=config.num_updates_per_batch,
        discounting=config.discounting,
        learning_rate=config.learning_rate,
        entropy_cost=config.entropy_cost,
        num_envs=config.num_envs,
        batch_size=config.batch_size,
        seed=config.seed,
    )

    # Train the agent
    logger.info("Starting training...")
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        progress_fn=progress,
    )
    logger.info("Training complete")

    # Save the trained policy
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = output_dir / "policy.pkl"
    with open(params_path, "wb") as f:
        jax.tree.map(lambda x: jnp.save(f, x), params)
    logger.info("Saved policy to %s", params_path)


if __name__ == "__main__":
    main()
