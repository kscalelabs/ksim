"""Defines the base observation class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from brax.base import State
from brax.mjx.base import State as MjxState
from jaxtyping import PRNGKeyArray

from ksim.utils.data import BuilderData

NoiseType = Literal["gaussian", "uniform"]


class Observation(eqx.Module, ABC):
    noise: float = eqx.field(default=0.0, static=True)
    noise_type: NoiseType = eqx.field(default="gaussian", static=True)

    def __init__(
        self,
        *,
        noise: float = 0.0,
        noise_type: NoiseType = "gaussian",
    ) -> None:
        super().__init__()

        self.noise = noise
        self.noise_type = noise_type

    @abstractmethod
    def __call__(self, state: State) -> jnp.ndarray:
        """Gets the observation from the state."""

    def add_noise(self, observation: jnp.ndarray, rng: PRNGKeyArray) -> jnp.ndarray:
        match self.noise_type:
            case "gaussian":
                return observation + jax.random.normal(rng, observation.shape) * self.noise
            case "uniform":
                return observation + jax.random.uniform(rng, observation.shape, minval=-self.noise, maxval=self.noise)
            case _:
                raise ValueError(f"Invalid noise type: {self.noise_type}")

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def observation_name(self) -> str:
        return self.get_name()


T = TypeVar("T", bound=Observation)


class ObservationBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, data: BuilderData) -> T:
        """Builds an observation from a MuJoCo model.

        Args:
            data: The data to build the observation from.

        Returns:
            An observation that can be applied to a state.
        """


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
    sensor_adr: int = eqx.field(static=True)
    sensor_dim: int = eqx.field(static=True)
    sensor_name: str | None = eqx.field(default=None, static=True)

    def __init__(
        self,
        *,
        sensor_adr: int,
        sensor_dim: int,
        noise: float = 0.0,
        noise_type: NoiseType = "gaussian",
        sensor_name: str | None = None,
    ) -> None:
        super().__init__(noise, noise_type)

        self.sensor_adr = sensor_adr
        self.sensor_dim = sensor_dim
        self.sensor_name = sensor_name

    @eqx.filter_jit
    def __call__(self, state: State) -> jnp.ndarray:
        if isinstance(state, MjxState):
            return state.sensordata[self.sensor_adr : self.sensor_adr + self.sensor_dim].ravel()

        raise NotImplementedError(f"SensorObservation is not implemented for {type(state)}")

    def get_name(self) -> str:
        base_name = super().get_name()
        return base_name if self.sensor_name is None else f"{self.sensor_name}_{base_name}"


class SensorObservationBuilder(ObservationBuilder[SensorObservation]):
    def __init__(
        self,
        *,
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

        sensor_id = data.sensor_name_to_idx[self.sensor_name]
        sensor_adr = data.model.sensor_adr[sensor_id]
        sensor_dim = data.model.sensor_dim[sensor_id]

        return SensorObservation(
            sensor_adr=sensor_adr,
            sensor_dim=sensor_dim,
            noise=self.noise,
            noise_type=self.noise_type,
            sensor_name=self.sensor_name,
        )
