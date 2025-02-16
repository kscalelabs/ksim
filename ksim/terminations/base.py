"""Defines the base termination class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
import mujoco
import xax
from brax.base import State


class Termination(eqx.Module, ABC):
    @abstractmethod
    def __call__(self, state: State) -> jnp.ndarray:
        """Checks if the environment has terminated.

        Args:
            state: The current state to check.

        Returns:
            A boolean tensor indicating whether the environment has terminated.
            This tensor should usually be of shape (num_envs).
        """

    @classmethod
    def get_name(cls) -> str:
        return xax.camelcase_to_snakecase(cls.__name__)

    @functools.cached_property
    def termination_name(self) -> str:
        return self.get_name()


T = TypeVar("T", bound=Termination)


class TerminationBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, mj_model: mujoco.MjModel) -> T:
        """Builds a termination from a MuJoCo model.

        Args:
            mj_model: The MuJoCo model to build the termination from.

        Returns:
            A termination that can be applied to a state.
        """
