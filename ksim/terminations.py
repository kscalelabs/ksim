"""Defines the base termination class."""

__all__ = [
    "Termination",
    "PitchTooGreatTermination",
    "RollTooGreatTermination",
    "MinimumHeightTermination",
    "FallTermination",
    "IllegalContactTermination",
    "BadZTermination",
    "FastAccelerationTermination",
]

import functools
import logging
from abc import ABC, abstractmethod
from typing import Collection, Literal, Self

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array

from ksim.types import PhysicsData, PhysicsModel
from ksim.utils.mujoco import get_geom_data_idx_by_name, get_sensor_data_idxs_by_name

logger = logging.getLogger(__name__)

SensorType = Literal["quaternion_orientation", "gravity_vector", "base_orientation"]


@attrs.define(frozen=True, kw_only=True)
class Termination(ABC):
    """Base class for terminations."""

    @abstractmethod
    def __call__(self, state: PhysicsData) -> Array:
        """Checks if the environment has terminated. Shape of output should be (num_envs)."""

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def termination_name(self) -> str:
        return self.get_name()


@attrs.define(frozen=True, kw_only=True)
class PitchTooGreatTermination(Termination):
    """Terminates the episode if the pitch is too great."""

    max_pitch: float

    def __call__(self, state: PhysicsData) -> Array:
        quat = state.qpos[3:7]
        pitch = jnp.arctan2(2 * quat[1] * quat[2] - 2 * quat[0] * quat[3], 1 - 2 * quat[1] ** 2 - 2 * quat[2] ** 2)
        return jnp.abs(pitch) > self.max_pitch


@attrs.define(frozen=True, kw_only=True)
class RollTooGreatTermination(Termination):
    """Terminates the episode if the roll is too great."""

    max_roll: float

    def __call__(self, state: PhysicsData) -> Array:
        quat = state.qpos[3:7]
        roll = jnp.arctan2(2 * quat[1] * quat[2] + 2 * quat[0] * quat[3], 1 - 2 * quat[2] ** 2 - 2 * quat[3] ** 2)
        return jnp.abs(roll) > self.max_roll


@attrs.define(frozen=True, kw_only=True)
class MinimumHeightTermination(Termination):
    """Terminates the episode if the robot is too low."""

    min_height: float

    def __call__(self, state: PhysicsData) -> Array:
        return state.qpos[2] < self.min_height


# ML: we should rewrite this to be fully understandable from the class name.
# Don't love that we combine logic and bury stuff like gravity[2] < 0.0.
@attrs.define(frozen=True, kw_only=True)
class FallTermination(Termination):
    """Terminates the episode if the robot falls."""

    sensor_name: str
    sensor_type: SensorType
    max_pitch: float = attrs.field(default=0.78)

    @xax.jit(static_argnames=["self"])
    def __call__(self, state: PhysicsData) -> Array:
        match self.sensor_type:
            case "quaternion_orientation":
                quat = state.qpos[3:7]
                pitch = jnp.arctan2(
                    2 * quat[1] * quat[2] - 2 * quat[0] * quat[3],
                    1 - 2 * quat[1] ** 2 - 2 * quat[2] ** 2,
                )
                return jnp.abs(pitch) > self.max_pitch

            case "gravity_vector":
                gravity = state.sensor[self.sensor_name]  # ML: does this exist?
                return gravity[2] < 0.0

            case "base_orientation":
                quat = state.qpos[3:7]
                pitch = jnp.arctan2(
                    2 * quat[1] * quat[2] - 2 * quat[0] * quat[3],
                    1 - 2 * quat[1] ** 2 - 2 * quat[2] ** 2,
                )
                return jnp.abs(pitch) > self.max_pitch

    @classmethod
    def create_from_quaternion_sensor(cls, physics_model: PhysicsModel, quaternion_sensor: str) -> Self:
        try:
            _ = get_sensor_data_idxs_by_name(physics_model)[quaternion_sensor]
            return cls(
                sensor_name=quaternion_sensor,
                sensor_type="quaternion_orientation",
            )
        except KeyError:
            raise ValueError(f"Quaternion sensor {quaternion_sensor} not found.")

    @classmethod
    def create_from_projected_gravity_sensor(cls, physics_model: PhysicsModel, projected_gravity_sensor: str) -> Self:
        try:
            _ = get_sensor_data_idxs_by_name(physics_model)[projected_gravity_sensor]
            return cls(
                sensor_name=projected_gravity_sensor,
                sensor_type="gravity_vector",
            )
        except KeyError:
            raise ValueError(f"Projected gravity sensor {projected_gravity_sensor} not found.")

    @classmethod
    def create(
        cls,
        physics_model: PhysicsModel,
        quaternion_sensor: str | None = None,
        projected_gravity_sensor: str | None = None,
    ) -> Self:
        if quaternion_sensor and projected_gravity_sensor:
            raise ValueError("Only one of quaternion or projected gravity sensor can be specified.")

        if quaternion_sensor:
            return cls.create_from_quaternion_sensor(physics_model, quaternion_sensor)

        if projected_gravity_sensor:
            return cls.create_from_projected_gravity_sensor(physics_model, projected_gravity_sensor)

        return cls(
            sensor_name="base",
            sensor_type="base_orientation",
        )


@attrs.define(frozen=True, kw_only=True)
class IllegalContactTermination(Termination):
    """Terminates when illegal contact is detected between specified geoms."""

    illegal_geom_idxs: jax.Array
    contact_eps: float = -0.001

    def __call__(self, state: PhysicsData) -> Array:
        if state.ncon == 0:
            return jnp.array(False)

        illegal_geom1 = jnp.isin(state.contact.geom1, self.illegal_geom_idxs)
        illegal_geom2 = jnp.isin(state.contact.geom2, self.illegal_geom_idxs)
        illegal_contact = jnp.logical_or(illegal_geom1, illegal_geom2)
        significant_contact = jnp.where(illegal_contact, state.contact.dist < self.contact_eps, False).any()

        return jnp.array(significant_contact)

    def __hash__(self) -> int:
        """Convert JAX arrays to tuples for hashing."""
        return hash((tuple(self.illegal_geom_idxs), self.contact_eps))

    @classmethod
    def create(
        cls,
        physics_model: PhysicsModel,
        geom_names: Collection[str],
        contact_eps: float = -1e-3,
    ) -> Self:
        illegal_geom_idxs = []
        geom_name_set = set(geom_names)
        for geom_name, geom_idx in get_geom_data_idx_by_name(physics_model).items():
            if geom_name in geom_name_set:
                illegal_geom_idxs.append(geom_idx)
                geom_name_set.remove(geom_name)

        if geom_name_set:
            choices = sorted(list(get_geom_data_idx_by_name(physics_model).keys()))
            raise ValueError(f"Geoms {geom_name_set} not found in model. Choices are: {choices}")

        illegal_geom_idxs = jnp.array(illegal_geom_idxs)

        return cls(
            contact_eps=contact_eps,
            illegal_geom_idxs=illegal_geom_idxs,
        )


@attrs.define(frozen=True, kw_only=True)
class BadZTermination(Termination):
    """Terminates the episode if the robot is unhealthy."""

    unhealthy_z_lower: float = attrs.field()
    unhealthy_z_upper: float = attrs.field()

    def __call__(self, state: PhysicsData) -> Array:
        height = state.qpos[2]
        return (height < self.unhealthy_z_lower) | (height > self.unhealthy_z_upper)


@attrs.define(frozen=True, kw_only=True)
class FastAccelerationTermination(Termination):
    """Terminates the episode if the robot is moving too fast."""

    # Good default value for Mujoco physics errors.
    max_ang_vel: float = attrs.field(default=100.0)
    max_lin_vel: float = attrs.field(default=100.0)

    def __call__(self, state: PhysicsData) -> Array:
        lin_vel = state.cvel[..., :3]
        ang_vel = state.cvel[..., 3:]
        return (lin_vel > self.max_lin_vel).any() | (ang_vel > self.max_ang_vel).any()
