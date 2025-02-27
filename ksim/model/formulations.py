"""High Level Formulations of RL Models."""

from abc import ABC, abstractmethod
from typing import Tuple

import flax.linen as nn
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

    @abstractmethod
    def get_variance(self) -> Array:
        """Get the variance of the action model."""
        ...


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
