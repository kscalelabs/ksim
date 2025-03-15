"""Base Types for Environments."""

from dataclasses import dataclass

import jax
import mujoco
from flax.core import FrozenDict
from jaxtyping import Array
from mujoco import mjx

PhysicsData = mjx.Data | mujoco.MjData
PhysicsModel = mjx.Model | mujoco.MjModel


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhysicsState:
    """Everything you need for the engine to take an action and step physics."""

    most_recent_action: Array  # since ctrl_dt > dt, we need this to simulate actuator logic
    model: PhysicsModel  # really important to think of this as a pointer of pointers... can be shared
    data: PhysicsData


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Transition:
    obs: FrozenDict[str, Array]  # <- state
    command: FrozenDict[str, Array]  # <- prev command
    action: Array  # <- obs, command
    reward: Array  # <- state, action, next state
    done: Array  # <- state, action, next state
    timestep: Array  # <- state

    termination_components: FrozenDict[str, Array]  # The specific reasons the episode terminated.
    reward_components: FrozenDict[str, Array]  # The individual reward components, scaled.
