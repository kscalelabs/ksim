"""Defines the base observation class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
import mujoco
import xax
from brax.base import State


class Observation(eqx.Module, ABC):
    @abstractmethod
    def __call__(self, state: State) -> jnp.ndarray:
        """Gets the observation from the state."""

    @classmethod
    def get_name(cls) -> str:
        return xax.camelcase_to_snakecase(cls.__name__)

    @functools.cached_property
    def observation_name(self) -> str:
        return self.get_name()


O = TypeVar("O", bound=Observation)


class ObservationBuilder(ABC, Generic[O]):
    @abstractmethod
    def __call__(self, mj_model: mujoco.MjModel) -> O:
        """Builds an observation from a MuJoCo model.

        Args:
            mj_model: The MuJoCo model to build the observation from.

        Returns:
            An observation that can be applied to a state.
        """
