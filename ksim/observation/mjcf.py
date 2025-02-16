"""Defines some useful reset functions for MJCF environments."""

import equinox as eqx
import jax.numpy as jnp
from brax.base import State

from ksim.observation.base import NoiseType, Observation, ObservationBuilder


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
        return state.qd[0:3]  # (3,)


class BaseAngularVelocityObservation(Observation):
    @eqx.filter_jit
    def __call__(self, state: State) -> jnp.ndarray:
        return state.qd[3:6]  # (3,)


class JointPositionObservation(Observation):
    @eqx.filter_jit
    def __call__(self, state: State) -> jnp.ndarray:
        return state.q[7:]  # (N,)


class JointVelocityObservation(Observation):
    @eqx.filter_jit
    def __call__(self, state: State) -> jnp.ndarray:
        return state.qd[6:]  # (N,)


class IMUOrientationObservation(Observation):
    imu_name: str

    def __init__(
        self,
        imu_name: str,
        noise: float = 0.0,
        noise_type: NoiseType = "gaussian",
    ) -> None:
        super().__init__(noise, noise_type)

        self.imu_name = imu_name

    @eqx.filter_jit
    def __call__(self, state: State) -> jnp.ndarray:
        raise NotImplementedError


class IMUOrientationObservationBuilder(ObservationBuilder[IMUOrientationObservation]):
    imu_name: str

    def __init__(self, imu_name: str) -> None:
        super().__init__()

        self.imu_name = imu_name

    def build(self) -> IMUOrientationObservation:
        return IMUOrientationObservation(self.imu_name)
