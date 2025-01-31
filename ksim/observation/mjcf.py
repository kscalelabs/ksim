"""Defines some useful resets for MJCF environments."""

import jax
import jax.numpy as jnp
from mujoco.mjx._src import math

from ksim.observation.base import Observation
from ksim.state.base import State


class BasePositionObservation(Observation[State]):
    def __call__(self, state: State) -> jnp.ndarray:
        return state.data.qpos[0:2]


class BaseOrientationObservation(Observation[State]):
    def __call__(self, state: State) -> jnp.ndarray:
        return state.data.qvel[0:2]


class JointPositionObservation(Observation[State]):
    def __call__(self, state: State) -> jnp.ndarray:
        return state.data.qpos[2:]


class JointVelocityObservation(Observation[State]):
    def __call__(self, state: State) -> jnp.ndarray:
        return state.data.qvel[2:]
