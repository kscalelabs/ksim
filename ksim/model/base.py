"""Minimal framework-agnostic base classes for actors and critics.

Philosophy: you may use any framework you want so long as the forward pass
expects a `ModelInput` and returns an `Array`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
from jaxtyping import Array

from ksim.model.distributions import ActionDistribution
from ksim.model.types import ModelInput, ModelRecurrence


class KSimModule(ABC):
    """Base class for all actor-critic agents."""

    @abstractmethod
    def forward(self, input: ModelInput, recurrence: ModelRecurrence | None) -> tuple[Array, ModelRecurrence | None]:
        """Apply the actor-critic to the given input. Can be recurrent."""

    @abstractmethod
    def initial_recurrence(self) -> ModelRecurrence:
        """Initial recurrence state for the model."""


@jax.tree_util.register_dataclass
@dataclass
class Agent(ABC):
    """Base class for all agents."""

    actor_model: KSimModule
    action_distribution: ActionDistribution


@jax.tree_util.register_dataclass
@dataclass
class ActorCriticAgent(Agent):
    """An agent that has both an actor and a critic."""

    critic_model: KSimModule
    actor_model: KSimModule
    action_distribution: ActionDistribution
