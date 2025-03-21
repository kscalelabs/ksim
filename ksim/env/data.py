"""Base Types for Environments."""

from dataclasses import dataclass
from typing import TypeAlias

import jax
import mujoco
from flax.core import FrozenDict
from jaxtyping import Array, PyTree
from mujoco import mjx

PhysicsData: TypeAlias = mjx.Data | mujoco.MjData
PhysicsModel: TypeAlias = mjx.Model | mujoco.MjModel


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhysicsState:
    """Everything you need for the engine to take an action and step physics."""

    most_recent_action: Array
    data: PhysicsData


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Trajectory:
    qpos: Array
    qvel: Array
    obs: FrozenDict[str, Array]
    command: FrozenDict[str, Array]
    action: Array
    done: Array
    timestep: Array
    termination_components: FrozenDict[str, Array]
    aux_outputs: PyTree | None


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Rewards:
    total: Array
    components: FrozenDict[str, Array]
