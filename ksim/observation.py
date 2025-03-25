"""Defines the base observation class."""

__all__ = [
    "Observation",
    "BasePositionObservation",
    "BaseOrientationObservation",
    "BaseLinearVelocityObservation",
    "BaseAngularVelocityObservation",
    "JointPositionObservation",
    "JointVelocityObservation",
    "CenterOfMassInertiaObservation",
    "CenterOfMassVelocityObservation",
    "ActuatorForceObservation",
    "SensorObservation",
    "BaseLinearAccelerationObservation",
    "BaseAngularAccelerationObservation",
    "ActuatorAccelerationObservation",
]

import functools
from abc import ABC, abstractmethod
from typing import Literal

import attrs
import jax
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.types import PhysicsModel, RolloutVariables
from ksim.utils.mujoco import get_sensor_data_idxs_by_name

NoiseType = Literal["gaussian", "uniform"]


def add_noise(observation: Array, rng: PRNGKeyArray, noise_type: NoiseType, noise: float) -> Array:
    match noise_type:
        case "gaussian":
            return observation + jax.random.normal(rng, observation.shape) * noise
        case "uniform":
            return observation + (jax.random.uniform(rng, observation.shape) * 2 - 1) * noise
        case _:
            raise ValueError(f"Invalid noise type: {noise_type}")


@attrs.define(frozen=True, kw_only=True)
class Observation(ABC):
    """Base class for observations."""

    @abstractmethod
    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        """Gets the observation from the state and carry.

        Args:
            rollout_state: The rollout state of the engine
            rng: A PRNGKeyArray to use for the observation

        Returns:
            The observation
        """

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        """Override to add noise to the observation.

        Args:
            observation: The raw observation from the state
            rng: A PRNGKeyArray to use for the noise

        Returns:
            The observation with noise added
        """
        return observation

    def __call__(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        obs_rng, noise_rng = jax.random.split(rng)
        raw_observation = self.observe(rollout_state, obs_rng)
        return self.add_noise(raw_observation, noise_rng)

    def get_name(self) -> str:
        """Get the name of the observation."""
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def observation_name(self) -> str:
        return self.get_name()


@attrs.define(frozen=True)
class BasePositionObservation(Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        qpos = rollout_state.physics_state.data.qpos[0:3]  # (3,)
        return qpos

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True)
class BaseOrientationObservation(Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        qpos = rollout_state.physics_state.data.qpos[3:7]  # (4,)
        return qpos

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True)
class BaseLinearVelocityObservation(Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        qvel = rollout_state.physics_state.data.qvel[0:3]  # (3,)
        return qvel

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True)
class BaseAngularVelocityObservation(Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        qvel = rollout_state.physics_state.data.qvel[3:6]  # (3,)
        return qvel

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True)
class JointPositionObservation(Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        qpos = rollout_state.physics_state.data.qpos[7:]  # (N,)
        return qpos

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True)
class JointVelocityObservation(Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        qvel = rollout_state.physics_state.data.qvel[6:]  # (N,)
        return qvel

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True)
class CenterOfMassInertiaObservation(Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        # Skip the first entry (world body) and flatten
        cinert = rollout_state.physics_state.data.cinert[1:].ravel()  # Shape will be (nbody-1, 10)
        return cinert

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True)
class CenterOfMassVelocityObservation(Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        # Skip the first entry (world body) and flatten
        cvel = rollout_state.physics_state.data.cvel[1:].ravel()  # Shape will be (nbody-1, 6)
        return cvel

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True)
class ActuatorForceObservation(Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        return rollout_state.physics_state.data.actuator_force  # Shape will be (nu,)

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.noise


@attrs.define(frozen=True, kw_only=True)
class SensorObservation(Observation):
    noise: float = attrs.field(default=0.0)
    noise_type: NoiseType = attrs.field(default="gaussian")
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

    def get_name(self) -> str:
        return f"{self.sensor_name}_obs"

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        sensor_data = rollout_state.physics_state.data.sensordata[
            self.sensor_idx_range[0] : self.sensor_idx_range[1]
        ].ravel()
        return sensor_data

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        return add_noise(observation, rng, self.noise_type, self.noise)


@attrs.define(frozen=True)
class BaseLinearAccelerationObservation(Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        return rollout_state.physics_state.data.qacc[0:3]


@attrs.define(frozen=True)
class BaseAngularAccelerationObservation(Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        return rollout_state.physics_state.data.qacc[3:6]


@attrs.define(frozen=True)
class ActuatorAccelerationObservation(Observation):
    noise: float = attrs.field(default=0.0)

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        return rollout_state.physics_state.data.qacc[6:]
