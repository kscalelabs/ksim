"""Base Types for Environments."""

from typing import NamedTuple

import mujoco
from flax.core import FrozenDict
from jaxtyping import Array
from mujoco import mjx


class Transition(NamedTuple):
    """Base environment state class."""

    obs: FrozenDict[str, Array]  # <- state
    command: FrozenDict[str, Array]  # <- prev command
    action: Array  # <- obs, command
    reward: Array  # <- state, action, next state
    done: Array  # <- state, action, next state
    timestep: Array  # <- state

    termination_components: FrozenDict[str, Array]  # The specific reasons the episode terminated.
    reward_components: FrozenDict[str, Array]  # The individual reward components, scaled.


PhysicsData = mjx.Data | mujoco.MjData
PhysicsModel = mjx.Model | mujoco.MjModel


class PhysicsState(NamedTuple):
    """Everything you need for the engine to take an action and step physics."""

    most_recent_action: Array  # since ctrl_dt > dt, we need this to simulate actuator logic
    model: PhysicsModel  # really important to think of this as a pointer of pointers... can be shared
    data: PhysicsData
