"""Defines the base observation class."""

import functools
from abc import ABC, abstractmethod
from typing import TypeVar, get_args

import attrs
import jax
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.env.data import PhysicsData, PhysicsModel
from ksim.types import NoiseType, ObsType
from ksim.utils.named_access import get_sensor_data_idxs_by_name


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
                    rng,
                    observation.shape,
                    minval=-self.noise,
                    maxval=self.noise,
                )
            case "uniform_multiply":
                return observation * jax.random.uniform(
                    rng,
                    observation.shape,
                    minval=1.0 - self.noise,
                    maxval=1.0 + self.noise,
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


class MjxObservation(Observation, ABC):
    obs_type: ObsType = "proprio"

    @abstractmethod
    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        """Gets the observation from the state."""


@attrs.define(frozen=True)
class BasePositionObservation(MjxObservation):
    obs_type: ObsType = "proprio"

    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        qpos = state.qpos[0:3]  # (3,)
        return qpos


@attrs.define(frozen=True)
class BaseOrientationObservation(MjxObservation):
    obs_type: ObsType = "proprio"

    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        qpos = state.qpos[3:7]  # (4,)
        return qpos


@attrs.define(frozen=True)
class BaseLinearVelocityObservation(MjxObservation):
    obs_type: ObsType = "proprio"

    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        qvel = state.qvel[0:3]  # (3,)
        return qvel


@attrs.define(frozen=True)
class BaseAngularVelocityObservation(MjxObservation):
    obs_type: ObsType = "proprio"

    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        qvel = state.qvel[3:6]  # (3,)
        return qvel


@attrs.define(frozen=True)
class JointPositionObservation(MjxObservation):
    obs_type: ObsType = "proprio"

    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        qpos = state.qpos[7:]  # (N,)
        return qpos


@attrs.define(frozen=True)
class JointVelocityObservation(MjxObservation):
    obs_type: ObsType = "proprio"

    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
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

    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        position = state.qpos
        if self.exclude_xy:
            position = position[2:]  # Skip x,y but include z and all joint positions
        return position


@attrs.define(frozen=True)
class LegacyVelocityObservation(Observation):
    """Legacy velocity observation that includes all velocities.

    In the legacy code, all velocities (base + joint) are included without any exclusions.
    """

    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        return state.qvel


@attrs.define(frozen=True)
class CenterOfMassInertiaObservation(Observation):
    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        # Skip the first entry (world body) and flatten
        cinert = state.cinert[1:].ravel()  # Shape will be (nbody-1, 10)
        return cinert


@attrs.define(frozen=True)
class CenterOfMassVelocityObservation(Observation):
    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        # Skip the first entry (world body) and flatten
        cvel = state.cvel[1:].ravel()  # Shape will be (nbody-1, 6)
        return cvel


@attrs.define(frozen=True)
class ActuatorForceObservation(Observation):
    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        # Get actuator forces
        qfrc_actuator = state.qfrc_actuator  # Shape will be (nu,)
        return qfrc_actuator


@attrs.define(frozen=True, kw_only=True)
class SensorObservation(Observation):
    sensor_name: str = attrs.field()
    sensor_idx_range: tuple[int, int | None] = attrs.field()

    @classmethod
    def create(
        cls,
        physics_model: PhysicsModel,
        sensor_name: str,
        noise: float = 0.0,
        noise_type: NoiseType = "gaussian",
    ) -> "SensorObservation":
        """Create a sensor observation from a physics model.

        Args:
            physics_model: MuJoCo physics model
            sensor_name: Name of sensor to observe
            noise: Amount of noise to add
            noise_type: Type of noise to add
        """
        sensor_name_to_idx_range = get_sensor_data_idxs_by_name(physics_model)
        if sensor_name not in sensor_name_to_idx_range:
            options = "\n".join(sorted(sensor_name_to_idx_range.keys()))
            raise ValueError(f"{sensor_name} not found in model. Available:\n{options}")

        return cls(
            noise=noise,
            noise_type=noise_type,
            sensor_name=sensor_name,
            sensor_idx_range=sensor_name_to_idx_range[sensor_name],
        )

    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        sensor_data = state.sensordata[self.sensor_idx_range[0] : self.sensor_idx_range[1]].ravel()
        return sensor_data

    def get_name(self) -> str:
        base_name = super().get_name()
        return f"{self.sensor_name}_{base_name}"
