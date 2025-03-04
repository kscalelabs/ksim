"""Defines the base observation class."""

import functools
from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, TypeVar

import attrs
import jax
import mujoco.mjx as mjx
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.utils.data import BuilderData
from ksim.utils.jit import legit_jit

NoiseType = Literal["gaussian", "uniform"]


@attrs.define(frozen=True, kw_only=True)
class Observation(ABC):
    """Base class for observations."""

    noise: float = attrs.field(default=0.0)
    noise_type: NoiseType = attrs.field(default="gaussian")

    @abstractmethod
    def __call__(self, state: Any, rng: PRNGKeyArray) -> Array:
        """Gets the observation from the state."""

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        match self.noise_type:
            case "gaussian":
                return observation + jax.random.normal(rng, observation.shape) * self.noise
            case "uniform":
                return observation + jax.random.uniform(
                    rng, observation.shape, minval=-self.noise, maxval=self.noise
                )
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
        """Builds an observation from a MuJoCo model."""


####################
# MJX Observations #
####################

# NOTE: we make assumption that the freejoint is ALWAYS the first 7 joints


class MjxObservation(Observation, ABC):
    @abstractmethod
    def __call__(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        """Gets the observation from the state."""


@attrs.define(frozen=True)
class BasePositionObservation(MjxObservation):
    # TODO: this should be fixed to return relative position
    @legit_jit(static_argnames=["self"])
    def __call__(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        qpos = state.qpos[0:3]  # (3,)
        return self.add_noise(qpos, rng)


@attrs.define(frozen=True)
class BaseOrientationObservation(Observation):
    @legit_jit(static_argnames=["self"])
    def __call__(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        qpos = state.qpos[3:7]  # (4,)
        return self.add_noise(qpos, rng)


@attrs.define(frozen=True)
class BaseLinearVelocityObservation(Observation):
    @legit_jit(static_argnames=["self"])
    def __call__(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        qvel = state.qvel[0:3]  # (3,)
        return self.add_noise(qvel, rng)


@attrs.define(frozen=True)
class BaseAngularVelocityObservation(Observation):
    @legit_jit(static_argnames=["self"])
    def __call__(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        qvel = state.qvel[3:6]  # (3,)
        return self.add_noise(qvel, rng)


@attrs.define(frozen=True)
class JointPositionObservation(Observation):
    @legit_jit(static_argnames=["self"])
    def __call__(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        qpos = state.qpos[7:]  # (N,)
        return self.add_noise(qpos, rng)


@attrs.define(frozen=True)
class JointVelocityObservation(Observation):
    @legit_jit(static_argnames=["self"])
    def __call__(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        qvel = state.qvel[6:]  # (N,)
        return self.add_noise(qvel, rng)


@attrs.define(frozen=True, kw_only=True)
class SensorObservation(Observation):
    sensor_name: str = attrs.field()
    sensor_idx_range: tuple[int, int | None] | None = attrs.field(default=None)

    @legit_jit(static_argnames=["self"])
    def __call__(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        assert self.sensor_idx_range is not None
        sensor_data = state.sensordata[self.sensor_idx_range[0] : self.sensor_idx_range[1]].ravel()
        return self.add_noise(sensor_data, rng)

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
        if self.sensor_name not in data.mujoco_mappings.sensor_name_to_idx_range:
            options = "\n".join(sorted(data.mujoco_mappings.sensor_name_to_idx_range.keys()))
            raise ValueError(f"{self.sensor_name} not found in model. Available:\n{options}")

        sensor_idx_range = data.mujoco_mappings.sensor_name_to_idx_range[self.sensor_name]

        return SensorObservation(
            noise=self.noise,
            noise_type=self.noise_type,
            sensor_name=self.sensor_name,
            sensor_idx_range=sensor_idx_range,
        )
