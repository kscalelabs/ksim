"""Defines the base termination class."""

import functools
from abc import ABC, abstractmethod
from typing import Collection, Generic, TypeVar

import attrs
import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array
from mujoco import mjx

from ksim.utils.data import BuilderData


@attrs.define(frozen=True, kw_only=True)
class Termination(ABC):
    """Base class for terminations."""

    @abstractmethod
    def __call__(self, state: mjx.Data) -> Array:
        """Checks if the environment has terminated. Shape of output should be (num_envs)."""

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def termination_name(self) -> str:
        return self.get_name()


T = TypeVar("T", bound=Termination)


class TerminationBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, data: BuilderData) -> T:
        """Builds a termination from a MuJoCo model."""


@attrs.define(frozen=True, kw_only=True)
class PitchTooGreatTermination(Termination):
    """Terminates the episode if the pitch is too great."""

    max_pitch: float

    @eqx.filter_jit
    def __call__(self, state: mjx.Data) -> Array:
        quat = state.qpos[3:7]
        pitch = jnp.arctan2(
            2 * quat[1] * quat[2] - 2 * quat[0] * quat[3], 1 - 2 * quat[1] ** 2 - 2 * quat[2] ** 2
        )
        return jnp.abs(pitch) > self.max_pitch


@attrs.define(frozen=True, kw_only=True)
class RollTooGreatTermination(Termination):
    """Terminates the episode if the roll is too great."""

    max_roll: float

    @eqx.filter_jit
    def __call__(self, state: mjx.Data) -> Array:
        quat = state.qpos[3:7]
        roll = jnp.arctan2(
            2 * quat[1] * quat[2] + 2 * quat[0] * quat[3], 1 - 2 * quat[2] ** 2 - 2 * quat[3] ** 2
        )
        return jnp.abs(roll) > self.max_roll


@attrs.define(frozen=True, kw_only=True)
class MinimumHeightTermination(Termination):
    """Terminates the episode if the robot is too low."""

    min_height: float

    @eqx.filter_jit
    def __call__(self, state: mjx.Data) -> Array:
        return state.qpos[2] < self.min_height


@attrs.define(frozen=True, kw_only=True)
class IllegalContactTermination(Termination):
    """Terminates when illegal contact is detected between specified geoms."""

    illegal_geom_idxs: jax.Array
    contact_eps: float = -0.001

    def __call__(self, state: mjx.Data) -> Array:
        if state.ncon == 0:
            return jnp.array(False)

        illegal_geom1 = jnp.isin(state.contact.geom1, self.illegal_geom_idxs)
        illegal_geom2 = jnp.isin(state.contact.geom2, self.illegal_geom_idxs)
        illegal_contact = jnp.logical_or(illegal_geom1, illegal_geom2)
        significant_contact = jnp.where(
            illegal_contact, state.contact.dist < self.contact_eps, False
        ).any()

        return jnp.array(significant_contact)

    def __hash__(self) -> int:
        """Convert JAX arrays to tuples for hashing."""
        return hash((tuple(self.illegal_geom_idxs), self.contact_eps))


@attrs.define(frozen=True, kw_only=True)
class IllegalContactTerminationBuilder(TerminationBuilder[IllegalContactTermination]):
    body_names: Collection[str] = attrs.field()
    contact_eps: float = attrs.field(default=-1e-3)

    def __call__(self, data: BuilderData) -> IllegalContactTermination:
        illegal_geom_idxs = []
        for geom_idx, body_name in data.mujoco_mappings.geom_idx_to_body_name.items():
            if body_name in self.body_names:
                illegal_geom_idxs.append(geom_idx)

        illegal_geom_idxs = jnp.array(illegal_geom_idxs)

        return IllegalContactTermination(
            contact_eps=self.contact_eps,
            illegal_geom_idxs=illegal_geom_idxs,
        )
