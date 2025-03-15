"""Minimal framework-agnostic base classes for actors and critics.

Philosophy: you may use any framework you want so long as the forward pass
expects a `ModelInput` and returns an `Array`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
from flax.core import FrozenDict
from jaxtyping import Array

from ksim.model.distributions import ActionDistribution
from ksim.model.types import ModelCarry


class KSimModule(ABC):
    """Base class for all actor-critic agents."""

    @abstractmethod
    def forward(
        self, obs: FrozenDict[str, Array], command: FrozenDict[str, Array], carry: ModelCarry | None
    ) -> tuple[Array, ModelCarry | None]:
        """Apply the actor-critic to the given input. Can be recurrent."""

    ######################
    # Easily Overridable #
    ######################

    def initial_carry(self) -> ModelCarry | None:
        """Initial carry state for the model.

        NOTE: you may use this to initialize recurrence, action history, etc.
        It gives you total freedom over the temporal dimension so long
        as the output is a PyTree.
        """
        return None

    def forward_accross_episode(self, obs: Array, command: Array) -> Array:
        """Forward pass across the episode (time, ...). No env dimension.

        By default, we vmap the forward pass for efficiency. If you implement
        recurrence, you should override this with an appropriate scan.
        """
        vmapped_forward = jax.vmap(self.forward, in_axes=(0, 0, None))
        prediction, _ = vmapped_forward(obs, command, None)
        return prediction


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
