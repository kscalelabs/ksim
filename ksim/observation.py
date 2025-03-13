"""Defines the base observation class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, get_args

import attrs
import jax
import xax
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx

from ksim.env.types import PhysicsData
from ksim.types import NoiseType, ObsType
from ksim.utils.data import BuilderData


@attrs.define(frozen=True, kw_only=True)
class Observation(ABC):
    """Base class for observations."""

    noise: float = attrs.field(default=0.0)
    noise_type: NoiseType = attrs.field(default="gaussian")
    obs_type: ObsType = attrs.field(default="proprio")

    def __attrs_post_init__(self) -> None:
        """Ensuring protected attributes are not present in the class name."""
        obs_types = get_args(ObsType)
        noise_types = get_args(NoiseType)
        name = self.__class__.__name__
        if "_" in name:
            raise ValueError("Class name cannot contain underscores")
        for obs_type in obs_types:
            if f"{obs_type}" in name.lower():
                raise ValueError(f"Class name cannot contain protected string: {obs_type}")
        for noise_type in noise_types:
            if f"{noise_type}" in name.lower():
                raise ValueError(f"Class name cannot contain protected string: {noise_type}")

    @abstractmethod
    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        """Gets the observation from the state."""

    def __call__(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        raw_observation = self.observe(state, rng)
        return self.add_noise(raw_observation, rng)

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
        """Get the name of the observation."""
        name = xax.camelcase_to_snakecase(self.__class__.__name__)
        name += f"_{self.obs_type}_{self.noise_type}"
        return name

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
    obs_type: ObsType = "proprio"

    @abstractmethod
    def observe(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        """Gets the observation from the state."""


@attrs.define(frozen=True)
class BasePositionObservation(MjxObservation):
    obs_type: ObsType = "proprio"

    def observe(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        qpos = state.qpos[0:3]  # (3,)
        return qpos


@attrs.define(frozen=True)
class BaseOrientationObservation(MjxObservation):
    obs_type: ObsType = "proprio"

    def observe(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        qpos = state.qpos[3:7]  # (4,)
        return qpos


@attrs.define(frozen=True)
class BaseLinearVelocityObservation(MjxObservation):
    obs_type: ObsType = "proprio"

    def observe(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        qvel = state.qvel[0:3]  # (3,)
        return qvel


@attrs.define(frozen=True)
class BaseAngularVelocityObservation(MjxObservation):
    obs_type: ObsType = "proprio"

    def observe(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        qvel = state.qvel[3:6]  # (3,)
        return qvel


@attrs.define(frozen=True)
class JointPositionObservation(MjxObservation):
    obs_type: ObsType = "proprio"

    def observe(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        qpos = state.qpos[7:]  # (N,)
        return qpos


@attrs.define(frozen=True)
class JointVelocityObservation(MjxObservation):
    obs_type: ObsType = "proprio"

    def observe(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        qvel = state.qvel[6:]  # (N,)
        return qvel


@attrs.define(frozen=True)
class LegacyPositionObservation(MjxObservation):
    """Legacy position observation that excludes x,y positions.

    In the legacy code, if exclude_current_positions_from_observation is True,
    it skips the first two elements (x,y) of qpos but includes z and all joint positions.
    """

    obs_type: ObsType = "proprio"

    exclude_xy: bool = attrs.field(default=True)

    def observe(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        position = state.qpos
        if self.exclude_xy:
            position = position[2:]  # Skip x,y but include z and all joint positions
        return position


@attrs.define(frozen=True)
class LegacyVelocityObservation(Observation):
    """Legacy velocity observation that includes all velocities.

    In the legacy code, all velocities (base + joint) are included without any exclusions.
    """

    def observe(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        return state.qvel


@attrs.define(frozen=True)
class CenterOfMassInertiaObservation(Observation):
    def observe(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        # Skip the first entry (world body) and flatten
        cinert = state.cinert[1:].ravel()  # Shape will be (nbody-1, 10)
        return cinert


@attrs.define(frozen=True)
class CenterOfMassVelocityObservation(Observation):
    def observe(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        # Skip the first entry (world body) and flatten
        cvel = state.cvel[1:].ravel()  # Shape will be (nbody-1, 6)
        return cvel


@attrs.define(frozen=True)
class ActuatorForceObservation(Observation):
    def observe(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        # Get actuator forces
        qfrc_actuator = state.qfrc_actuator  # Shape will be (nu,)
        return qfrc_actuator


@attrs.define(frozen=True, kw_only=True)
class SensorObservation(Observation):
    sensor_name: str = attrs.field()
    sensor_idx_range: tuple[int, int | None] | None = attrs.field(default=None)

    def observe(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        assert self.sensor_idx_range is not None
        sensor_data = state.sensordata[self.sensor_idx_range[0] : self.sensor_idx_range[1]].ravel()
        return sensor_data

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
