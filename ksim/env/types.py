"""Base Types for Environments."""

from dataclasses import dataclass
from typing import Literal

import jax
from flax.core import FrozenDict
from jaxtyping import Array
from mujoco import mjx


@jax.tree_util.register_dataclass
@dataclass(frozen=True, kw_only=True)
class EnvState:
    """Base environment state class."""

    # Data attributes
    obs: FrozenDict[str, Array]
    command: FrozenDict[str, Array]
    action: Array
    reward: Array  # R(prev_state, action_at_prev_state, current_state)
    done: Array
    timestep: Array

    termination_components: FrozenDict[str, Array]  # Termination terms at same timestep as done
    reward_components: FrozenDict[str, Array]  # Reward terms at same timestep as reward


KScaleActionModelType = Literal["random", "zero", "midpoint"]
PhysicsData = mjx.Data | None
PhysicsModel = mjx.Model | None
