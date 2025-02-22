import bdb
import functools
import signal
import sys
import textwrap
import traceback
from dataclasses import dataclass
from threading import Thread
from typing import Dict, Literal, NamedTuple

import equinox as eqx
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import xax
from brax.base import System
from brax.envs.base import State as BraxState
from dpshdl.dataset import Dataset
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.base_env import BaseEnv
from ksim.env.cartpole_env import CartPoleEnv
from ksim.model.formulations import ActorCriticModel
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOBatch, PPOConfig, PPOTask
from ksim.task.rl import RLTask


@dataclass
class CartPoleConfig(PPOConfig):
    """Configuration for CartPole training."""

    # Env parameters.
    sutton_barto_reward: bool = xax.field(value=False, help="Use Sutton and Barto reward function.")
    batch_size: int = xax.field(value=16, help="Batch size.")

    # ML model parameters.
    actor_hidden_dims: int = xax.field(value=128, help="Hidden dimensions for the actor.")
    actor_num_layers: int = xax.field(value=2, help="Number of layers for the actor.")
    critic_hidden_dims: int = xax.field(value=128, help="Hidden dimensions for the critic.")
    critic_num_layers: int = xax.field(value=2, help="Number of layers for the critic.")

    observation_size: int = 4
    action_size: int = 1


class CartPoleTask(PPOTask[CartPoleConfig]):
    """Task for CartPole training."""

    def get_environment(self) -> CartPoleEnv:
        """Get the environment.

        Returns:
            The environment.
        """
        return CartPoleEnv()

    def get_model(self, key: PRNGKeyArray) -> ActorCriticModel:
        """Get the model.
        Args:
            key: The random key.

        Returns:
            The model.
        """
        return ActorCriticModel(
            actor_module=MLP(
                num_hidden_layers=self.config.actor_num_layers,
                hidden_features=self.config.actor_hidden_dims,
                out_features=2,  # two discrete actions for CartPole
            ),
            critic_module=MLP(
                num_hidden_layers=self.config.critic_num_layers,
                hidden_features=self.config.critic_hidden_dims,
                out_features=1,
            ),
        )


if __name__ == "__main__":
    # python -m examples.cartpole.cartpole_task train
    CartPoleTask.launch(
        CartPoleConfig(
            num_envs=1,
            batch_size=16,
            max_trajectory_seconds=10.0,
            valid_every_n_steps=5,
        ),
    )
