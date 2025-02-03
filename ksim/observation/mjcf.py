"""Defines some useful resets for MJCF environments."""

from typing import Generic, TypeVar

import jax.numpy as jnp
from brax.mjx.base import State

from ksim.observation.base import Observation

Tstate = TypeVar("Tstate", bound=State)


class BasePositionObservation(Observation[Tstate], Generic[Tstate]):
    def __call__(self, state: Tstate) -> jnp.ndarray:
        return state.qpos[0:3]  # x, y, z


class BaseOrientationObservation(Observation[Tstate], Generic[Tstate]):
    def __call__(self, state: Tstate) -> jnp.ndarray:
        return state.qpos[3:7]  # qw, qx, qy, qz


class BaseLinearVelocityObservation(Observation[Tstate], Generic[Tstate]):
    def __call__(self, state: Tstate) -> jnp.ndarray:
        return state.qvel[0:3]  # x, y, z


class BaseAngularVelocityObservation(Observation[Tstate], Generic[Tstate]):
    def __call__(self, state: Tstate) -> jnp.ndarray:
        return state.qvel[3:6]  # wx, wy, wz


class JointPositionObservation(Observation[Tstate], Generic[Tstate]):
    def __call__(self, state: Tstate) -> jnp.ndarray:
        return state.qpos[7:]


class JointVelocityObservation(Observation[Tstate], Generic[Tstate]):
    def __call__(self, state: Tstate) -> jnp.ndarray:
        return state.qvel[6:]
