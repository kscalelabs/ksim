"""Base Types for Environments."""

__all__ = [
    "PhysicsData",
    "PhysicsModel",
    "PhysicsState",
    "Trajectory",
    "Rewards",
    "RolloutVariables",
    "Histogram",
    "Metrics",
]

from dataclasses import dataclass
from typing import Mapping, TypeAlias

import jax
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from mujoco import mjx

PhysicsData: TypeAlias = mjx.Data | mujoco.MjData
PhysicsModel: TypeAlias = mjx.Model | mujoco.MjModel


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhysicsState:
    """Everything you need for the engine to take an action and step physics."""

    most_recent_action: Array
    data: PhysicsData
    event_states: xax.FrozenDict[str, PyTree]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Trajectory:
    qpos: Array
    qvel: Array
    obs: xax.FrozenDict[str, Array]
    command: xax.FrozenDict[str, Array]
    action: Array
    done: Array
    timestep: Array
    termination_components: xax.FrozenDict[str, Array]
    aux_outputs: PyTree


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Rewards:
    total: Array
    components: xax.FrozenDict[str, Array]


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


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RolloutVariables:
    carry: PyTree
    commands: xax.FrozenDict[str, Array]
    physics_state: PhysicsState
    rng: PRNGKeyArray
