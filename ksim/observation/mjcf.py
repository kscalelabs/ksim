"""Defines some useful reset functions for MJCF environments."""

import equinox as eqx
import jax.numpy as jnp
from brax.base import State

from ksim.observation.base import Observation


class BasePositionObservation(Observation):
    @eqx.filter_jit
    def __call__(self, state: State) -> jnp.ndarray:
        return state.q[0:3]  # (3,)


class BaseOrientationObservation(Observation):
    @eqx.filter_jit
    def __call__(self, state: State) -> jnp.ndarray:
        return state.q[3:7]  # (4,)


class BaseLinearVelocityObservation(Observation):
    @eqx.filter_jit
    def __call__(self, state: State) -> jnp.ndarray:
        return state.xd.vel[0]  # (3,)


class BaseAngularVelocityObservation(Observation):
    @eqx.filter_jit
    def __call__(self, state: State) -> jnp.ndarray:
        return state.xd.ang[0]  # (3,)


class JointPositionObservation(Observation):
    @eqx.filter_jit
    def __call__(self, state: State) -> jnp.ndarray:
        return state.q[7:]  # (N,)


class JointVelocityObservation(Observation):
    @eqx.filter_jit
    def __call__(self, state: State) -> jnp.ndarray:
        return state.qd  # (N,)
