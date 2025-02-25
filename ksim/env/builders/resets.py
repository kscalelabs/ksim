"""Defines the base reset class."""

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xax
from jaxtyping import PRNGKeyArray

from ksim.env.base_env import EnvState
from ksim.utils.data import BuilderData


@jax.tree_util.register_dataclass
@dataclass
class ResetData:
    rng: PRNGKeyArray
    state: EnvState


class Reset(eqx.Module, ABC):
    @abstractmethod
    def __call__(self, data: ResetData) -> ResetData:
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
        """Builds a reset from a MuJoCo model.

        Args:
            data: The data to build the reset from.

        Returns:
            A reset that can be applied to a state.
        """


class XYPositionReset(Reset):
    bounds: tuple[float, float, float, float]
    padding_prct: float
    hfield_data: jnp.ndarray

    def __init__(
        self,
        *,
        bounds: tuple[float, float, float, float],
        hfield_data: np.ndarray,
        padding_prct: float = 0.1,
    ) -> None:
        """Initialize the reset.

        Args:
            bounds: Bounds for where to position the robot.
            hfield_data: The height field data.
            padding_prct: The percentage of the bounds to pad.
        """
        super().__init__()

        self.bounds = bounds
        self.padding_prct = padding_prct
        self.hfield_data = jnp.array(hfield_data)

    def __call__(self, data: ResetData) -> ResetData:
        x, y, ztop, _ = self.bounds

        prct = 1.0 - self.padding_prct
        x, y = x * prct, y * prct

        # Generate random position
        rng, keyx, keyy = jax.random.split(data.rng, 3)
        dx = jax.random.uniform(keyx, (1,), minval=-x, maxval=x)
        dy = jax.random.uniform(keyy, (1,), minval=-y, maxval=y)

        # Update position while maintaining small height above ground
        qpos_j = data.state.q
        qpos_j = qpos_j.at[0:1].set(dx)
        qpos_j = qpos_j.at[1:2].set(dy)

        # Make sure the Z position is above the ground.
        z = self.hfield_data[dx.astype(int), dy.astype(int)]
        qpos_j = qpos_j.at[2:3].set(z + ztop)

        data.state = data.state.replace(q=qpos_j)
        data.rng = rng
        return data


class XYPositionResetBuilder(ResetBuilder[XYPositionReset]):
    def __init__(
        self,
        *,
        padding_prct: float = 0.1,
    ) -> None:
        super().__init__()

        self.padding_prct = padding_prct

    def __call__(self, data: BuilderData) -> XYPositionReset:
        x, y, ztop, zbottom = data.model.hfield_size.flatten().tolist()
        nx, ny = int(data.model.hfield_nrow), int(data.model.hfield_ncol)
        hfield_data = data.model.hfield_data.reshape(nx, ny)
        return XYPositionReset(
            bounds=(x, y, ztop, zbottom),
            hfield_data=hfield_data,
            padding_prct=self.padding_prct,
        )
