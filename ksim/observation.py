"""Defines the base observation class."""

__all__ = [
    "ObservationState",
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
    "ContactObservation",
    "FeetContactObservation",
    "FeetPositionObservation",
    "FeetOrientationObservation",
    "TimestepObservation",
]

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Literal, Self

import attrs
import jax
import xax
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ksim.types import PhysicsModel, PhysicsState
from ksim.utils.mujoco import (
    geoms_colliding,
    get_body_data_idx_from_name,
    get_geom_data_idx_from_name,
    get_sensor_data_idxs_by_name,
)
from ksim.vis import Marker

NoiseType = Literal["gaussian", "uniform"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ObservationState:
    commands: xax.FrozenDict[str, Array]
    physics_state: PhysicsState


def add_noise(
    observation: Array,
    rng: PRNGKeyArray,
    noise_type: NoiseType,
    noise: float,
    curriculum_level: Array,
) -> Array:
    match noise_type:
        case "gaussian":
            return observation + jax.random.normal(rng, observation.shape) * noise * curriculum_level
        case "uniform":
            return observation + (jax.random.uniform(rng, observation.shape) * 2 - 1) * noise * curriculum_level
        case _:
            raise ValueError(f"Invalid noise type: {noise_type}")


@attrs.define(frozen=True, kw_only=True)
class Observation(ABC):
    """Base class for observations."""

    noise: float = attrs.field(default=0.0)
    noise_type: NoiseType = attrs.field(default="gaussian")

    @abstractmethod
    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        """Gets the observation from the state.

        Args:
            state: The inputs from which the obseravtion can be extracted.
            rng: A PRNGKeyArray to use for the noise

        Returns:
            The observation
        """

    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        """Override to add noise to the observation.

        Args:
            observation: The raw observation from the state
            curriculum_level: The current curriculum level, a scalar between
                zero and one.
            rng: A PRNGKeyArray to use for the noise

        Returns:
            The observation with noise added
        """
        return jax.tree.map(lambda x: add_noise(x, rng, self.noise_type, self.noise, curriculum_level), observation)

    def get_markers(self) -> Collection[Marker]:
        return []

    def __call__(self, state: ObservationState, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        obs_rng, noise_rng = jax.random.split(rng)
        raw_observation = self.observe(state, obs_rng)
        return self.add_noise(raw_observation, curriculum_level, noise_rng)

    def get_name(self) -> str:
        """Get the name of the observation."""
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def observation_name(self) -> str:
        return self.get_name()


@attrs.define(frozen=True, kw_only=True)
class BasePositionObservation(Observation):
    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        qpos = state.physics_state.data.qpos[0:3]  # (3,)
        return qpos


@attrs.define(frozen=True, kw_only=True)
class BaseOrientationObservation(Observation):
    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        qpos = state.physics_state.data.qpos[3:7]  # (4,)
        return qpos


@attrs.define(frozen=True, kw_only=True)
class BaseLinearVelocityObservation(Observation):
    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        qvel = state.physics_state.data.qvel[0:3]  # (3,)
        return qvel


@attrs.define(frozen=True, kw_only=True)
class BaseAngularVelocityObservation(Observation):
    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        qvel = state.physics_state.data.qvel[3:6]  # (3,)
        return qvel


@attrs.define(frozen=True, kw_only=True)
class JointPositionObservation(Observation):
    freejoint_first: bool = attrs.field(default=True)

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        if self.freejoint_first:
            return state.physics_state.data.qpos[7:]  # (N,)
        else:
            return state.physics_state.data.qpos  # (N,)


@attrs.define(frozen=True, kw_only=True)
class JointVelocityObservation(Observation):
    freejoint_first: bool = attrs.field(default=True)

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        if self.freejoint_first:
            return state.physics_state.data.qvel[6:]  # (N,)
        else:
            return state.physics_state.data.qvel  # (N,)


@attrs.define(frozen=True, kw_only=True)
class CenterOfMassInertiaObservation(Observation):
    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        # Skip the first entry (world body) and flatten
        cinert = state.physics_state.data.cinert[1:].ravel()  # Shape will be (nbody-1, 10)
        return cinert


@attrs.define(frozen=True, kw_only=True)
class CenterOfMassVelocityObservation(Observation):
    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        # Skip the first entry (world body) and flatten
        cvel = state.physics_state.data.cvel[1:].ravel()  # Shape will be (nbody-1, 6)
        return cvel


@attrs.define(frozen=True, kw_only=True)
class ActuatorForceObservation(Observation):
    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.actuator_force  # Shape will be (nu,)


@attrs.define(frozen=True, kw_only=True)
class SensorObservation(Observation):
    sensor_name: str = attrs.field()
    sensor_idx_range: tuple[int, int | None] = attrs.field()

    @classmethod
    def create(
        cls,
        *,
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
        return f"{super().get_name()}_{self.sensor_name}"

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        start, end = self.sensor_idx_range
        sensor_data = state.physics_state.data.sensordata[start:end].ravel()
        return sensor_data


@attrs.define(frozen=True, kw_only=True)
class BaseLinearAccelerationObservation(Observation):
    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.qacc[0:3]


@attrs.define(frozen=True, kw_only=True)
class BaseAngularAccelerationObservation(Observation):
    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.qacc[3:6]


@attrs.define(frozen=True, kw_only=True)
class ActuatorAccelerationObservation(Observation):
    freejoint_first: bool = attrs.field(default=True)

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        if self.freejoint_first:
            return state.physics_state.data.qacc[6:]
        else:
            return state.physics_state.data.qacc


@attrs.define(frozen=True, kw_only=True)
class ContactObservation(Observation):
    geom_idxs: tuple[int, ...] = attrs.field()
    contact_group: str | None = attrs.field(default=None)

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        geom_names: str | Collection[str],
        contact_group: str | None = None,
        noise: float = 0.0,
    ) -> Self:
        """Create a sensor observation from a physics model."""
        if isinstance(geom_names, str):
            geom_names = [geom_names]
        geom_idxs = [get_geom_data_idx_from_name(physics_model, name) for name in geom_names]
        return cls(
            geom_idxs=tuple(geom_idxs),
            noise=noise,
            contact_group=contact_group,
        )

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        geom_idxs = jnp.array(self.geom_idxs)
        contact = geoms_colliding(state.physics_state.data, geom_idxs, geom_idxs).any(axis=-1)
        return contact

    def get_name(self) -> str:
        if self.contact_group is not None:
            return f"{super().get_name()}_{self.contact_group}"
        else:
            return super().get_name()


@attrs.define(frozen=True, kw_only=True)
class FeetContactObservation(Observation):
    foot_left: tuple[int, ...] = attrs.field()
    foot_right: tuple[int, ...] = attrs.field()
    floor_geom: tuple[int, ...] = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        foot_left_geom_names: str | Collection[str],
        foot_right_geom_names: str | Collection[str],
        floor_geom_names: str | Collection[str],
        noise: float = 0.0,
    ) -> Self:
        """Create a sensor observation from a physics model."""
        if isinstance(foot_left_geom_names, str):
            foot_left_geom_names = [foot_left_geom_names]
        if isinstance(foot_right_geom_names, str):
            foot_right_geom_names = [foot_right_geom_names]
        if isinstance(floor_geom_names, str):
            floor_geom_names = [floor_geom_names]

        foot_left_idxs = [get_geom_data_idx_from_name(physics_model, name) for name in foot_left_geom_names]
        foot_right_idxs = [get_geom_data_idx_from_name(physics_model, name) for name in foot_right_geom_names]
        floor_geom_idxs = [get_geom_data_idx_from_name(physics_model, name) for name in floor_geom_names]
        return cls(
            foot_left=tuple(foot_left_idxs),
            foot_right=tuple(foot_right_idxs),
            floor_geom=tuple(floor_geom_idxs),
            noise=noise,
        )

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        foot_left = jnp.array(self.foot_left)
        foot_right = jnp.array(self.foot_right)
        floor = jnp.array(self.floor_geom)
        contact_1 = geoms_colliding(state.physics_state.data, foot_left, floor).any(axis=-1)
        contact_2 = geoms_colliding(state.physics_state.data, foot_right, floor).any(axis=-1)
        return jnp.stack([contact_1, contact_2], axis=-1)


@attrs.define(frozen=True, kw_only=True)
class FeetPositionObservation(Observation):
    foot_left: int = attrs.field()
    foot_right: int = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        foot_left_body_name: str,
        foot_right_body_name: str,
        noise: float = 0.0,
    ) -> Self:
        foot_left_idx = get_body_data_idx_from_name(physics_model, foot_left_body_name)
        foot_right_idx = get_body_data_idx_from_name(physics_model, foot_right_body_name)
        return cls(
            foot_left=foot_left_idx,
            foot_right=foot_right_idx,
            noise=noise,
        )

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        foot_left_pos = state.physics_state.data.xpos[self.foot_left]
        foot_right_pos = state.physics_state.data.xpos[self.foot_right]
        return jnp.stack([foot_left_pos, foot_right_pos], axis=-2)


@attrs.define(frozen=True, kw_only=True)
class FeetOrientationObservation(Observation):
    foot_left: int = attrs.field()
    foot_right: int = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        foot_left_body_name: str,
        foot_right_body_name: str,
        noise: float = 0.0,
    ) -> Self:
        foot_left_idx = get_body_data_idx_from_name(physics_model, foot_left_body_name)
        foot_right_idx = get_body_data_idx_from_name(physics_model, foot_right_body_name)
        return cls(
            foot_left=foot_left_idx,
            foot_right=foot_right_idx,
            noise=noise,
        )

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        foot_left_quat = state.physics_state.data.xquat[self.foot_left]
        foot_right_quat = state.physics_state.data.xquat[self.foot_right]
        return jnp.stack([foot_left_quat, foot_right_quat], axis=-2)


@attrs.define(frozen=True, kw_only=True)
class TimestepObservation(Observation):
    """Returns the current timestep in the episode."""

    def observe(self, state: ObservationState, rng: PRNGKeyArray) -> Array:
        time = state.physics_state.data.time
        if not isinstance(time, Array):
            time = jnp.array(time)
        return time.reshape(1)
