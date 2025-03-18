"""Minimal framework-agnostic base classes for actors and critics.

Philosophy: you may use any framework you want so long as the forward pass
expects a `ModelInput` and returns an `Array`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
from flax.core import FrozenDict
from jaxtyping import Array, PyTree
from xax.nn.distributions import ActionDistribution

ModelCarry = PyTree

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

    # TODO: move this to RLTask and make it overrideable there...
    def initial_carry(self) -> ModelCarry | None:
        """Initial carry state for the model.

        NOTE: you may use this to initialize recurrence, action history, etc.
        It gives you total freedom over the temporal dimension so long
        as the output is a PyTree.
        """
        return None

    def batched_forward_across_time(self, obs: FrozenDict[str, Array], command: FrozenDict[str, Array]) -> Array:
        """Forward pass across the episode (time, ...). No env dimension.

        By default, we vmap the forward pass for efficiency. If you implement
        recurrence, you should override this with an appropriate scan.
        """
        vmapped_forward = jax.vmap(self.forward, in_axes=(0, 0, None))
        prediction, _ = vmapped_forward(obs, command, None)
        return prediction

class DeployableModel(KSimModule):
    """A model that can be deployed."""

    @abstractmethod
    def make_export_model(self) -> tuple[Callable, int]:
        """Makes a callable inference function that directly takes a flattened input vector and returns an action."""

@jax.tree_util.register_dataclass
@dataclass
class Agent(ABC):
    """Base class for all agents."""

    actor_model: DeployableModel
    action_distribution: ActionDistribution


@jax.tree_util.register_dataclass
@dataclass
class ActorCriticAgent(Agent):
    """An agent that has both an actor and a critic."""

    critic_model: KSimModule
    actor_model: DeployableModel
    action_distribution: ActionDistribution = eqx.static_field()
