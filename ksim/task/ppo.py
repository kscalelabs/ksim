"""Defines a standard task interface for training a policy."""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
import xax

from ksim.task.rl import RLConfig, RLTask


@dataclass(kw_only=True)
class PPOConfig(RLConfig):
    clip_param: float = xax.field(value=0.2, help="Clipping parameter for PPO.")
    gamma: float = xax.field(value=0.998, help="Discount factor for PPO.")
    lam: float = xax.field(value=0.95, help="Lambda parameter for PPO.")
    value_loss_coef: float = xax.field(value=1.0, help="Value loss coefficient for PPO.")
    entropy_coef: float = xax.field(value=0.0, help="Entropy coefficient for PPO.")
    learning_rate: float = xax.field(value=1e-3, help="Learning rate for PPO.")
    max_grad_norm: float = xax.field(value=1.0, help="Maximum gradient norm for PPO.")
    use_clipped_value_loss: bool = xax.field(value=True, help="Whether to use clipped value loss for PPO.")
    schedule: str = xax.field(value="fixed", help="Schedule for PPO.")
    desired_kl: float = xax.field(value=0.01, help="Desired KL divergence for PPO.")


Config = TypeVar("Config", bound=PPOConfig)


class PPOTask(RLTask[Config], Generic[Config], ABC):

    def compute_loss(self, model: eqx.Module, batch: Any, output: Any, state: xax.State) -> jnp.ndarray:
        raise NotImplementedError
