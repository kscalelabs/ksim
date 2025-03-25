"""Base Types for Environments."""

__all__ = [
    "PhysicsData",
    "PhysicsModel",
    "PhysicsState",
    "Trajectory",
    "Rewards",
    "Histogram",
    "Metrics",
]

from dataclasses import dataclass
from typing import Mapping, TypeAlias

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
    event_info: FrozenDict[str, PyTree]


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
    aux_outputs: PyTree


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Rewards:
    total: Array
    components: FrozenDict[str, Array]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Histogram:
    counts: Array
    limits: Array
    min: Array
    max: Array
    sum: Array
    sum_squares: Array
    mean: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Metrics:
    train: Mapping[str, Array | Histogram]
    reward: Mapping[str, Array | Histogram]
    termination: Mapping[str, Array | Histogram]
