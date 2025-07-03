"""Defines the base termination class."""

__all__ = [
    "Termination",
    "NotUprightTermination",
    "MinimumHeightTermination",
    "IllegalContactTermination",
    "BadZTermination",
    "HighVelocityTermination",
    "HighRootVelocityTermination",
    "FarFromOriginTermination",
    "EpisodeLengthTermination",
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
from ksim.utils.mujoco import get_geom_data_idx_by_name

logger = logging.getLogger(__name__)

SensorType = Literal["quaternion_orientation", "gravity_vector", "base_orientation"]


@attrs.define(frozen=True, kw_only=True)
class Termination(ABC):
    """Base class for terminations."""

    @abstractmethod
    def __call__(self, state: PhysicsData, curriculum_level: Array) -> Array:
        """Checks if the environment has terminated. Shape of output should be (num_envs).

        Args:
            state: The current state of the environment.
            curriculum_level: The current curriculum level.

        Returns:
            A integer array of shape (num_envs) indicating. If the episode
            has been terminated, the value should be either 1 or -1, where
            1 represents a successful episode and -1 represents a failed episode.
            It should be 0 if the episode is not terminated.
        """

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def termination_name(self) -> str:
        return self.get_name()


@attrs.define(frozen=True, kw_only=True)
class NotUprightTermination(Termination):
    """Terminates the episode if the pitch is too great."""

    max_radians: float = attrs.field(validator=attrs.validators.gt(0.0))

    def __call__(self, state: PhysicsData, curriculum_level: Array) -> Array:
        gravity = jnp.array([0.0, 0.0, 1.0])
        quat = state.qpos[..., 3:7]
        gravity_vec = xax.rotate_vector_by_quat(gravity, quat, inverse=True)[..., 2]
        return jnp.where(jnp.arccos(gravity_vec) > self.max_radians, -1, 0)


@attrs.define(frozen=True, kw_only=True)
class MinimumHeightTermination(Termination):
    """Terminates the episode if the robot is too low."""

    min_height: float = attrs.field(validator=attrs.validators.gt(0.0))

    def __call__(self, state: PhysicsData, curriculum_level: Array) -> Array:
        return jnp.where(state.qpos[2] < self.min_height, -1, 0)


@attrs.define(frozen=True, kw_only=True)
class IllegalContactTermination(Termination):
    """Terminates when illegal contact is detected between specified geoms."""

    illegal_geom_idxs: jax.Array
    contact_eps: float = -0.001

    def __call__(self, state: PhysicsData, curriculum_level: Array) -> Array:
        if state.ncon == 0:
            return jnp.array(False)

        illegal_geom1 = jnp.isin(state.contact.geom1, self.illegal_geom_idxs)
        illegal_geom2 = jnp.isin(state.contact.geom2, self.illegal_geom_idxs)
        illegal_contact = jnp.logical_or(illegal_geom1, illegal_geom2)
        significant_contact = jnp.where(illegal_contact, state.contact.dist < self.contact_eps, False).any()

        return jnp.where(significant_contact, -1, 0)

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

    def __call__(self, state: PhysicsData, curriculum_level: Array) -> Array:
        height = state.qpos[2]
        return jnp.where((height < self.unhealthy_z_lower) | (height > self.unhealthy_z_upper), -1, 0)


@attrs.define(frozen=True, kw_only=True)
class HighVelocityTermination(Termination):
    """Terminates the episode if the robot is moving too fast."""

    # Good default value for Mujoco physics errors.
    max_lin_vel: float = attrs.field(default=100.0)
    max_ang_vel: float = attrs.field(default=100.0)

    def __call__(self, state: PhysicsData, curriculum_level: Array) -> Array:
        lin_vel = jnp.linalg.norm(state.cvel[..., :3], axis=-1)
        ang_vel = jnp.linalg.norm(state.cvel[..., 3:], axis=-1)
        return jnp.where((lin_vel > self.max_lin_vel).any() | (ang_vel > self.max_ang_vel).any(), -1, 0)


@attrs.define(frozen=True, kw_only=True)
class HighRootVelocityTermination(Termination):
    """Terminates the episode if the robot's root is moving too fast."""

    max_lin_vel: float = attrs.field()
    max_ang_vel: float = attrs.field()

    def __call__(self, state: PhysicsData, curriculum_level: Array) -> Array:
        lin_vel = jnp.linalg.norm(state.qvel[..., :3], axis=-1)
        ang_vel = jnp.linalg.norm(state.qvel[..., 3:6], axis=-1)
        return jnp.where((lin_vel > self.max_lin_vel) | (ang_vel > self.max_ang_vel), -1, 0)


@attrs.define(frozen=True, kw_only=True)
class FarFromOriginTermination(Termination):
    """Terminates the episode if the robot is too far from the origin.

    Defaults to a positive termination.
    """

    max_dist: float = attrs.field(validator=attrs.validators.gt(0.0))
    pos_termination: bool = attrs.field(default=True)

    def __call__(self, state: PhysicsData, curriculum_level: Array) -> Array:
        termination_value = 1 if self.pos_termination else -1
        return jnp.where(jnp.linalg.norm(state.qpos[..., :3], axis=-1) > self.max_dist, termination_value, 0)


@attrs.define(frozen=True, kw_only=True)
class EpisodeLengthTermination(Termination):
    """Terminates the episode if the robot has been alive for too long.

    This defaults to a positive termination.
    """

    max_length_sec: float = attrs.field(validator=attrs.validators.gt(0.0))
    disable_at_curriculum_level: int = attrs.field(default=None)
    pos_termination: bool = attrs.field(default=True)

    def __call__(self, state: PhysicsData, curriculum_level: Array) -> Array:
        termination_value = 1 if self.pos_termination else -1
        long_episodes = jnp.where(state.time > self.max_length_sec, termination_value, 0)
        if self.disable_at_curriculum_level is not None:
            return jnp.where(curriculum_level < self.disable_at_curriculum_level, 0, long_episodes)

        return long_episodes
