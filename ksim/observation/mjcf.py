"""Defines some useful reset functions for MJCF environments."""

import equinox as eqx
import jax.numpy as jnp
from brax.base import State

from ksim.observation.base import NoiseType, Observation, ObservationBuilder
from ksim.utils.data import BuilderData


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


class SensorObservation(Observation):
    sensor_id: int
    sensor_name: str | None

    def __init__(
        self,
        sensor_id: int,
        noise: float = 0.0,
        noise_type: NoiseType = "gaussian",
        sensor_name: str | None = None,
    ) -> None:
        super().__init__(noise, noise_type)

        self.sensor_id = sensor_id
        self.sensor_name = sensor_name

    @eqx.filter_jit
    def __call__(self, state: State) -> jnp.ndarray:
        breakpoint()
        raise NotImplementedError

    def get_name(self) -> str:
        base_name = super().get_name()
        return base_name if self.sensor_name is None else f"{self.sensor_name}_{base_name}"


class SensorObservationBuilder(ObservationBuilder[SensorObservation]):
    sensor_name: str
    noise: float
    noise_type: NoiseType

    def __init__(
        self,
        sensor_name: str,
        noise: float = 0.0,
        noise_type: NoiseType = "gaussian",
    ) -> None:
        super().__init__()

        self.sensor_name = sensor_name
        self.noise = noise
        self.noise_type = noise_type

    def __call__(self, data: BuilderData) -> SensorObservation:
        if self.sensor_name not in data.sensor_name_to_idx:
            options = "\n".join(sorted(data.sensor_name_to_idx.keys()))
            raise ValueError(f"IMU {self.sensor_name} not found in model. Available:\n{options}")
        return SensorObservation(
            data.sensor_name_to_idx[self.sensor_name],
            noise=self.noise,
            noise_type=self.noise_type,
            sensor_name=self.sensor_name,
        )
