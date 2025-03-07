"""Defines the base reset class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import PRNGKeyArray
from mujoco import mjx

from ksim.utils.data import BuilderData


@attrs.define(frozen=True, kw_only=True)
class Reset(ABC):
    """Base class for resets."""

    @abstractmethod
    def __call__(self, data: mjx.Data, rng: PRNGKeyArray) -> mjx.Data:
        """Resets the environment."""

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def reset_name(self) -> str:
        return self.get_name()


T = TypeVar("T", bound=Reset)


class ResetBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, data: BuilderData) -> T:
        """Builds a reset from a MuJoCo model."""


@attrs.define(frozen=True, kw_only=True)
class XYPositionReset(Reset):
    """Resets the position of the robot to a random point within a bounding box."""

    bounds: tuple[float, float, float, float]
    padding_prct: float = attrs.field(default=0.1)
    hfield_data: jnp.ndarray

    def __call__(self, data: mjx.Data, rng: PRNGKeyArray) -> mjx.Data:
        # TODO: figure out why this is not jit-able.
        x, y, ztop, _ = self.bounds
        prct = 1.0 - self.padding_prct
        x, y = x * prct, y * prct

        # Generate random position within bounds
        rng_split, keyx, keyy = jax.random.split(rng, 3)
        dx = jax.random.uniform(keyx, (1,), minval=-x, maxval=x)
        dy = jax.random.uniform(keyy, (1,), minval=-y, maxval=y)

        qpos_j = data.qpos
        qpos_j = qpos_j.at[0:1].set(dx)
        qpos_j = qpos_j.at[1:2].set(dy)

        # Only modify Z position if we have non-empty height field data
        # Check both the shape and if any elements are non-zero
        has_hfield_data = (self.hfield_data.size > 0) & (jnp.any(self.hfield_data != 0))

        def update_z_position(_: None) -> jax.Array:
            # Convert dx, dy to indices for the hfield_data
            nx, ny = self.hfield_data.shape
            x_bound, y_bound = self.bounds[0], self.bounds[1]

            # Map from [-x_bound, x_bound] to [0, nx-1] and [-y_bound, y_bound] to [0, ny-1]
            x_idx = jnp.clip(
                ((dx.squeeze() + x_bound) / (2 * x_bound) * (nx - 1)).astype(jnp.int32), 0, nx - 1
            )
            y_idx = jnp.clip(
                ((dy.squeeze() + y_bound) / (2 * y_bound) * (ny - 1)).astype(jnp.int32), 0, ny - 1
            )

            # Get the height at the sampled position
            z = self.hfield_data[x_idx, y_idx]
            return qpos_j.at[2:3].set(z + ztop)

        def keep_original_z(_: None) -> jax.Array:
            return qpos_j

        # Conditionally update the z position
        qpos_j = jax.lax.cond(has_hfield_data, update_z_position, keep_original_z, None)

        data = data.replace(qpos=qpos_j)
        return data

    def __hash__(self) -> int:
        """Convert JAX arrays to bytes for hashing."""
        array_bytes = self.hfield_data.tobytes()
        return hash((self.bounds, self.padding_prct, self.hfield_data.shape, array_bytes))


@attrs.define(frozen=True, kw_only=True)
class XYPositionResetBuilder(ResetBuilder[XYPositionReset]):
    """Builds a XYPositionReset from a MuJoCo model."""

    padding_prct: float = attrs.field(default=0.1)

    def __call__(self, data: BuilderData) -> XYPositionReset:
        x, y, ztop, zbottom = data.model.hfield_size.flatten().tolist()
        # Convert to integers properly to avoid deprecation warning
        nx = (
            int(data.model.hfield_nrow.item())
            if hasattr(data.model.hfield_nrow, "item")
            else int(data.model.hfield_nrow)
        )
        ny = (
            int(data.model.hfield_ncol.item())
            if hasattr(data.model.hfield_ncol, "item")
            else int(data.model.hfield_ncol)
        )
        hfield_data = data.model.hfield_data.reshape(nx, ny)
        return XYPositionReset(
            bounds=(x, y, ztop, zbottom),
            hfield_data=hfield_data,
            padding_prct=self.padding_prct,
        )


@attrs.define(frozen=True, kw_only=True)
class RandomJointPositionReset(Reset):
    """Adds uniformly sampled noise to default joint positions."""

    range: tuple[float, float]

    def __call__(self, data: mjx.Data, rng: PRNGKeyArray) -> mjx.Data:
        qpos = data.qpos
        qpos = qpos + jax.random.uniform(
            rng, qpos.shape, minval=self.range[0], maxval=self.range[1]
        )
        data = data.replace(qpos=qpos)
        return data


@attrs.define(frozen=True, kw_only=True)
class RandomJointVelocityReset(Reset):
    """Adds uniformly sampled noise to default joint velocities."""

    range: tuple[float, float]

    def __call__(self, data: mjx.Data, rng: PRNGKeyArray) -> mjx.Data:
        qvel = data.qvel
        qvel = qvel + jax.random.uniform(
            rng, qvel.shape, minval=self.range[0], maxval=self.range[1]
        )
        data = data.replace(qvel=qvel)
        return data
