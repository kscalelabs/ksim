"""High Level Formulations of RL Models."""

from abc import ABC, abstractmethod
from typing import Tuple

import flax.linen as nn
from jaxtyping import Array, PRNGKeyArray

from ksim.env.types import EnvState


class ActionModel(nn.Module, ABC):
    """Action model."""

    @abstractmethod
    def __call__(self, state: EnvState) -> Array:
        """Forward pass of the model."""
        ...

    @abstractmethod
    def calc_log_prob(self, prediction: Array, action: Array) -> Array:
        """Calculate the log probability of the action."""
        ...

    @abstractmethod
    def sample_and_log_prob(self, state: EnvState, rng: PRNGKeyArray) -> Tuple[Array, Array]:
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
    def __call__(self, state: EnvState) -> Tuple[Array, Array]:
        """Forward pass of the model."""
        return self.actor(state), self.critic(state)

    def actor(self, state: EnvState) -> Array:
        """Actor forward pass."""
        return self.actor_module(state)

    def critic(self, state: EnvState) -> Array:
        """Critic forward pass."""
        return self.critic_module(state)

    def actor_calc_log_prob(self, prediction: Array, action: Array) -> Array:
        """Calculate the log probability of the action."""
        return self.actor_module.calc_log_prob(prediction, action)

    def actor_sample_and_log_prob(self, state: EnvState, rng: PRNGKeyArray) -> Tuple[Array, Array]:
        """Sample and calculate the log probability of the action."""
        return self.actor_module.sample_and_log_prob(state, rng)


class GaussianActorCriticModel(ActorCriticModel):
    """Gaussian actor-critic model."""

    actor_module: GaussianActionModel
    critic_module: nn.Module


class CategoricalActorCriticModel(ActorCriticModel):
    """Categorical actor-critic model."""

    actor_module: CategoricalActionModel
    critic_module: nn.Module
