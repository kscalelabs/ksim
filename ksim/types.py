"""Base Types for Environments."""

__all__ = [
    "PhysicsData",
    "PhysicsModel",
    "PhysicsState",
    "Trajectory",
    "RewardState",
    "Action",
    "Histogram",
    "Metrics",
    "LoggedTrajectory",
]

from dataclasses import dataclass
from typing import Mapping, TypeAlias

import jax
import jax.numpy as jnp
import mujoco
import xax
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
    event_states: xax.FrozenDict[str, PyTree]
    actuator_state: PyTree
    action_latency: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Trajectory:
    """Stackable structure of transitions.

    Note that `qpos`, `qvel`, `xpos`, `xquat` and `timestep` are the values
    from *after* the action has been taken, while `obs` and `command` are the
    values from *before* the action has been taken.
    """

    qpos: Array
    qvel: Array
    xpos: Array
    xquat: Array
    obs: xax.FrozenDict[str, PyTree]
    command: xax.FrozenDict[str, PyTree]
    event_state: xax.FrozenDict[str, Array]
    action: Array
    done: Array
    success: Array
    timestep: Array
    termination_components: xax.FrozenDict[str, Array]
    aux_outputs: xax.FrozenDict[str, PyTree] | None

    def episode_length(self) -> Array:
        done_mask = self.done.at[..., -1].set(True)
        termination_sum = jnp.sum(jnp.where(done_mask, self.timestep, 0.0), axis=-1)
        episode_length = termination_sum / done_mask.sum(axis=-1)
        return episode_length


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RewardState:
    total: Array
    components: xax.FrozenDict[str, Array]
    carry: xax.FrozenDict[str, PyTree]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Action:
    action: Array
    carry: PyTree | None = None
    aux_outputs: xax.FrozenDict[str, PyTree] | None = None


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
    curriculum_level: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LoggedTrajectory:
    trajectory: Trajectory
    rewards: RewardState
    metrics: xax.FrozenDict[str, Array]
