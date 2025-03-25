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
from typing import Collection, Mapping, TypeAlias

import jax
import mujoco
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree
from mujoco import mjx

from ksim.commands import Command
from ksim.observation import Observation
from ksim.randomization import Randomization
from ksim.rewards import Reward
from ksim.terminations import Termination

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
class RolloutConstants:
    obs_generators: Collection[Observation]
    command_generators: Collection[Command]
    reward_generators: Collection[Reward]
    termination_generators: Collection[Termination]
    randomization_generators: Collection[Randomization]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RolloutVariables:
    carry: PyTree
    commands: FrozenDict[str, Array]
    physics_state: PhysicsState
    rng: PRNGKeyArray



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
