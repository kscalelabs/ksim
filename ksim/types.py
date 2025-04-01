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
from typing import Mapping, Self, TypeAlias

import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from mujoco import mjx

PhysicsData: TypeAlias = mjx.Data | mujoco.MjData
PhysicsModel: TypeAlias = mjx.Model | mujoco.MjModel


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CurriculumState:
    """Persistent curriculum state."""

    curriculum_steps: xax.FrozenDict[str, int]

    @classmethod
    def initialize(cls, event_names: list[str], num_envs: int) -> Self:
        """Initialize curriculum state with zeros for each event."""
        return cls(curriculum_steps=xax.FrozenDict({name: 0 for name in event_names}))

    def update(self, event_name: str, should_step: Array) -> Self:
        """Update curriculum step for a specific event."""
        current_step = self.curriculum_steps[event_name]
        new_step = jnp.where(should_step, current_step + 1, current_step)

        # Create new curriculum steps dictionary with updated values
        new_curriculum_steps = dict(self.curriculum_steps)
        new_curriculum_steps[event_name] = new_step.item()

        return CurriculumState(curriculum_steps=xax.FrozenDict(new_curriculum_steps))


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
    xpos: Array
    obs: xax.FrozenDict[str, PyTree]
    command: xax.FrozenDict[str, PyTree]
    event_state: xax.FrozenDict[str, Array]
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
    curriculum: Mapping[str, int]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RolloutVariables:
    carry: PyTree
    commands: xax.FrozenDict[str, Array]
    physics_state: PhysicsState
    rng: PRNGKeyArray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SingleTrajectory:
    trajectory: Trajectory
    rewards: Rewards
    metrics: xax.FrozenDict[str, Array]
