"""Base Types for Environments."""

from dataclasses import dataclass

import jax
from flax.core import FrozenDict
from jaxtyping import Array
from mujoco import mjx


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class EnvState:
    """Base environment state class."""

    obs: FrozenDict[str, Array]  # Observations derived from the state.
    command: FrozenDict[str, Array]  # Command provided to the actor and critic.
    action: Array  # The action that the actor should take
    reward: Array  # The total reward.
    done: Array  # Whether the episode has terminated for any reason.
    timestep: Array  # The current timestep.

    termination_components: FrozenDict[str, Array]  # The specific reasons the episode terminated.
    reward_components: FrozenDict[str, Array]  # The individual reward components, scaled.


PhysicsData = mjx.Data | None
PhysicsModel = mjx.Model | None
