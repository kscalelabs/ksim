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
    "FeetContactObservation",
    "FeetPositionObservation",
]

import functools
from abc import ABC, abstractmethod
from typing import Collection, Literal, Self

import attrs
import jax
import xax
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ksim.types import PhysicsModel, RolloutVariables
from ksim.utils.mujoco import geoms_colliding, get_geom_data_idx_by_name, get_sensor_data_idxs_by_name
from ksim.vis import Marker

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

    noise: float = attrs.field(default=0.0)
    noise_type: NoiseType = attrs.field(default="gaussian")

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
        return add_noise(observation, rng, self.noise_type, self.noise)

    def get_markers(self) -> Collection[Marker]:
        return []

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

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        qpos = rollout_state.physics_state.data.qpos[0:3]  # (3,)
        return qpos


@attrs.define(frozen=True)
class BaseOrientationObservation(Observation):

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        qpos = rollout_state.physics_state.data.qpos[3:7]  # (4,)
        return qpos


@attrs.define(frozen=True)
class BaseLinearVelocityObservation(Observation):

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        qvel = rollout_state.physics_state.data.qvel[0:3]  # (3,)
        return qvel


@attrs.define(frozen=True)
class BaseAngularVelocityObservation(Observation):

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        qvel = rollout_state.physics_state.data.qvel[3:6]  # (3,)
        return qvel


@attrs.define(frozen=True)
class JointPositionObservation(Observation):

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        qpos = rollout_state.physics_state.data.qpos[7:]  # (N,)
        return qpos


@attrs.define(frozen=True)
class JointVelocityObservation(Observation):

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        qvel = rollout_state.physics_state.data.qvel[6:]  # (N,)
        return qvel


@attrs.define(frozen=True)
class CenterOfMassInertiaObservation(Observation):

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        # Skip the first entry (world body) and flatten
        cinert = rollout_state.physics_state.data.cinert[1:].ravel()  # Shape will be (nbody-1, 10)
        return cinert


@attrs.define(frozen=True)
class CenterOfMassVelocityObservation(Observation):

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        # Skip the first entry (world body) and flatten
        cvel = rollout_state.physics_state.data.cvel[1:].ravel()  # Shape will be (nbody-1, 6)
        return cvel


@attrs.define(frozen=True)
class ActuatorForceObservation(Observation):

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        return rollout_state.physics_state.data.actuator_force  # Shape will be (nu,)


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
    ) -> Self:
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


@attrs.define(frozen=True)
class BaseLinearAccelerationObservation(Observation):

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        return rollout_state.physics_state.data.qacc[0:3]


@attrs.define(frozen=True)
class BaseAngularAccelerationObservation(Observation):

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        return rollout_state.physics_state.data.qacc[3:6]


@attrs.define(frozen=True)
class ActuatorAccelerationObservation(Observation):

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        return rollout_state.physics_state.data.qacc[6:]


@attrs.define(frozen=True)
class FeetContactObservation(Observation):
    foot_left: int = attrs.field()
    foot_right: int = attrs.field()
    floor_geom_id: int = attrs.field()

    @classmethod
    def create(
        cls,
        physics_model: PhysicsModel,
        foot_left: str,
        foot_right: str,
        floor_geom_id: str,
    ) -> Self:
        """Create a sensor observation from a physics model."""
        foot_left_idx = get_geom_data_idx_by_name(physics_model)[foot_left]
        foot_right_idx = get_geom_data_idx_by_name(physics_model)[foot_right]
        floor_geom_id = get_geom_data_idx_by_name(physics_model)[floor_geom_id]
        return cls(
            foot_left=foot_left_idx,
            foot_right=foot_right_idx,
            floor_geom_id=floor_geom_id,
        )

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        contact_1 = geoms_colliding(rollout_state.physics_state.data, self.foot_left, self.floor_geom_id)
        contact_2 = geoms_colliding(rollout_state.physics_state.data, self.foot_right, self.floor_geom_id)
        return jnp.array([contact_1, contact_2])


@attrs.define(frozen=True)
class FeetPositionObservation(Observation):
    foot_left: int = attrs.field()
    foot_right: int = attrs.field()

    @classmethod
    def create(cls, physics_model: PhysicsModel, foot_left: str, foot_right: str) -> Self:
        foot_left_idx = get_geom_data_idx_by_name(physics_model)[foot_left]
        foot_right_idx = get_geom_data_idx_by_name(physics_model)[foot_right]
        return cls(foot_left=foot_left_idx, foot_right=foot_right_idx)

    def observe(self, rollout_state: RolloutVariables, rng: PRNGKeyArray) -> Array:
        foot_left_pos = rollout_state.physics_state.data.geom_xpos[self.foot_left]
        foot_right_pos = rollout_state.physics_state.data.geom_xpos[self.foot_right]
        return jnp.concatenate([foot_left_pos, foot_right_pos], axis=-1)
