"""Defines the base reset class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import attrs
import jax.numpy as jnp
import mujoco.mjx as mjx
import xax
from jaxtyping import PRNGKeyArray

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
        # # TODO: figure out why this is not jit-able.
        # x, y, ztop, _ = self.bounds

        # prct = 1.0 - self.padding_prct
        # x, y = x * prct, y * prct

        # rng, keyx, keyy = jax.random.split(rng, 3)
        # dx = jax.random.uniform(keyx, (1,), minval=-x, maxval=x)
        # dy = jax.random.uniform(keyy, (1,), minval=-y, maxval=y)

        # qpos_j = data.qpos
        # qpos_j = qpos_j.at[0:1].set(dx)
        # qpos_j = qpos_j.at[1:2].set(dy)

        # z = self.hfield_data[dx.astype(int), dy.astype(int)]
        # qpos_j = qpos_j.at[2:3].set(z + ztop)

        # data = data.replace(qpos=qpos_j)
        data = data.replace(qpos=jnp.array([0.0, 0.0, 1.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        return data

    def __hash__(self) -> int:
        """Convert JAX arrays to bytes for hashing."""
        array_bytes = self.hfield_data.tobytes()
        return hash((self.bounds, self.padding_prct, self.hfield_data.shape, array_bytes))



@attrs.define(frozen=True, kw_only=True)
class XYPositionReset(Reset):
    """Resets the position of the robot to a random point within a bounding box."""

    bounds: tuple[float, float, float, float]
    padding_prct: float = attrs.field(default=0.1)
    hfield_data: jnp.ndarray

    def __call__(self, data: mjx.Data, rng: PRNGKeyArray) -> mjx.Data:
        # # TODO: figure out why this is not jit-able.
        # x, y, ztop, _ = self.bounds

        # prct = 1.0 - self.padding_prct
        # x, y = x * prct, y * prct

        # rng, keyx, keyy = jax.random.split(rng, 3)
        # dx = jax.random.uniform(keyx, (1,), minval=-x, maxval=x)
        # dy = jax.random.uniform(keyy, (1,), minval=-y, maxval=y)

        # qpos_j = data.qpos
        # qpos_j = qpos_j.at[0:1].set(dx)
        # qpos_j = qpos_j.at[1:2].set(dy)

        # z = self.hfield_data[dx.astype(int), dy.astype(int)]
        # qpos_j = qpos_j.at[2:3].set(z + ztop)

        # data = data.replace(qpos=qpos_j)
        # pfb30: inspect that
        data = data.replace(qpos=jnp.array([0.0, 0.0, 1.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
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
        x, y, ztop, zbottom = 0, 0, 0, 0  #     data.model.hfield_size.flatten().tolist()
        nx, ny = 256, 256  # int(data.model.hfield_nrow), int(data.model.hfield_ncol)
        hfield_data = 0  # data.model.hfield_data.reshape(nx, ny)
        return XYPositionReset(
            bounds=(x, y, ztop, zbottom),
            hfield_data=hfield_data,
            padding_prct=self.padding_prct,
        )


@attrs.define(frozen=True, kw_only=True)
class XYPositionResetKbot(Reset):
    """Resets the position of the robot to a random point within a bounding box."""

    bounds: tuple[float, float, float, float]
    padding_prct: float = attrs.field(default=0.1)
    hfield_data: jnp.ndarray

    def __call__(self, data: mjx.Data, rng: PRNGKeyArray) -> mjx.Data:
        # # TODO: figure out why this is not jit-able.
        # x, y, ztop, _ = self.bounds

        # prct = 1.0 - self.padding_prct
        # x, y = x * prct, y * prct

        # rng, keyx, keyy = jax.random.split(rng, 3)
        # dx = jax.random.uniform(keyx, (1,), minval=-x, maxval=x)
        # dy = jax.random.uniform(keyy, (1,), minval=-y, maxval=y)

        # qpos_j = data.qpos
        # qpos_j = qpos_j.at[0:1].set(dx)
        # qpos_j = qpos_j.at[1:2].set(dy)

        # z = self.hfield_data[dx.astype(int), dy.astype(int)]
        # qpos_j = qpos_j.at[2:3].set(z + ztop)

        # data = data.replace(qpos=qpos_j)
        data = data.replace(qpos=jnp.array(
            [0.       , 0.       , 1.0821425, 1.       , 0.       , 0.       ,
                0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
                0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
                0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
                0.       , 0.       , 0.       ]
        ))
        return data

    def __hash__(self) -> int:
        """Convert JAX arrays to bytes for hashing."""
        array_bytes = self.hfield_data.tobytes()
        return hash((self.bounds, self.padding_prct, self.hfield_data.shape, array_bytes))


@attrs.define(frozen=True, kw_only=True)
class XYPositionResetKbotBuilder(ResetBuilder[XYPositionResetKbot]):
    """Builds a XYPositionReset from a MuJoCo model."""

    padding_prct: float = attrs.field(default=0.1)

    def __call__(self, data: BuilderData) -> XYPositionReset:
        x, y, ztop, zbottom = 0, 0, 0, 0  #     data.model.hfield_size.flatten().tolist()
        nx, ny = 256, 256  # int(data.model.hfield_nrow), int(data.model.hfield_ncol)
        hfield_data = 0  # data.model.hfield_data.reshape(nx, ny)
        return XYPositionResetKbot(
            bounds=(x, y, ztop, zbottom),
            hfield_data=hfield_data,
            padding_prct=self.padding_prct,
        )