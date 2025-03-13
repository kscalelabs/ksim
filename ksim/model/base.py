"""Minimal framework-agnostic base classes for actors and critics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
from jaxtyping import Array, Shaped

from ksim.model.distributions import ActionDistribution
from ksim.model.types import ModelInput


class ActorModel(ABC):
    """Base class for all actor-critic agents."""

    @abstractmethod
    def forward(self, input: ModelInput) -> Array:
        """Apply the actor-critic to the given input."""


class CriticModel(ABC):
    """Base class for all critic models."""

    @abstractmethod
    def forward(self, input: ModelInput) -> Shaped[Array, "... 1"]:
        """Apply the critic to the given input, must return a scalar value."""


@jax.tree_util.register_dataclass
@dataclass
class Agent(ABC):
    """Base class for all agents."""

    actor_model: ActorModel
    action_distribution: ActionDistribution


class ActorCriticAgent(Agent):
    """An agent that has both an actor and a critic."""

    actor_model: ActorModel
    critic_model: CriticModel
    action_distribution: ActionDistribution
