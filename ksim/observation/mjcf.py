"""Defines some useful resets for MJCF environments."""

import jax.numpy as jnp
from brax.mjx.base import State as MjxState

from ksim.observation.base import Observation


class BasePositionObservation(Observation):
    def __call__(self, state: MjxState) -> jnp.ndarray:
        return state.qpos[0:3]  # x, y, z


class BaseOrientationObservation(Observation):
    def __call__(self, state: MjxState) -> jnp.ndarray:
        return state.qpos[3:7]  # qw, qx, qy, qz


class BaseLinearVelocityObservation(Observation):
    def __call__(self, state: MjxState) -> jnp.ndarray:
        return state.qvel[0:3]  # x, y, z


class BaseAngularVelocityObservation(Observation):
    def __call__(self, state: MjxState) -> jnp.ndarray:
        return state.qvel[3:6]  # wx, wy, wz


class JointPositionObservation(Observation):
    def __call__(self, state: MjxState) -> jnp.ndarray:
        return state.qpos[7:]


class JointVelocityObservation(Observation):
    def __call__(self, state: MjxState) -> jnp.ndarray:
        return state.qvel[6:]
