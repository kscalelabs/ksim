"""High Level Formulations of RL Models."""

from abc import ABC, abstractmethod
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray


class ActionModel(nn.Module, ABC):
    """Action model."""

    @abstractmethod
    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        """Forward pass of the model."""
        ...

    @abstractmethod
    def calc_log_prob(self, prediction: Array, action: Array) -> Array:
        """Calculate the log probability of the action."""
        ...

    @abstractmethod
    def sample_and_log_prob(
        self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array], rng: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        """Sample and calculate the log probability of the action."""
        ...


class GaussianActionModel(ActionModel, ABC):
    """Gaussian action model."""

    init_log_std: float
    num_outputs: int

    def setup(self) -> None:
        self.log_std = self.param(
            "log_std", nn.initializers.constant(self.init_log_std), (self.num_outputs,)
        )

    def calc_log_prob(self, prediction: Array, action: Array) -> Array:
        mean = prediction
        std = jnp.exp(self.log_std)

        log_prob = (
            -0.5 * jnp.square((action - mean) / std) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
        )  # (...batch_dims..., action_dim)
        # if we assume indep gaussians, can just sum over action dim in log space for log prob
        return jnp.sum(log_prob, axis=-1)

    def sample_and_log_prob(
        self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array], rng: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        mean = self(obs, cmd)
        std = jnp.exp(self.log_std)

        noise = jax.random.normal(rng, mean.shape)
        action = mean + noise * std
        log_prob = self.calc_log_prob(mean, action)

        return action, log_prob


class CategoricalActionModel(ActionModel, ABC):
    """Categorical action model."""

    sampling_temperature: float


class ActorCriticModel(nn.Module):
    """Actor-Critic model."""

    actor_module: ActionModel
    critic_module: nn.Module

    @nn.compact
    def __call__(
        self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]
    ) -> Tuple[Array, Array]:
        """Forward pass of the model."""
        return self.actor(obs, cmd), self.critic(obs, cmd)

    def actor(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        """Actor forward pass."""
        return self.actor_module(obs, cmd)

    def critic(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        """Critic forward pass."""
        return self.critic_module(obs, cmd)

    def actor_calc_log_prob(self, prediction: Array, action: Array) -> Array:
        """Calculate the log probability of the action."""
        return self.actor_module.calc_log_prob(prediction, action)

    def actor_sample_and_log_prob(
        self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array], rng: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        """Sample and calculate the log probability of the action."""
        return self.actor_module.sample_and_log_prob(obs, cmd, rng)
