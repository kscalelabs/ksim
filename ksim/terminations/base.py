"""Defines the base termination class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import jax.numpy as jnp
import xax

from ksim.state.base import State

Tstate = TypeVar("Tstate", bound=State)


class Termination(ABC, Generic[Tstate]):
    @abstractmethod
    def __call__(self, state: Tstate) -> jnp.ndarray:
        """Checks if the environment has terminated.

        Args:
            state: The current state to check.

        Returns:
            A boolean tensor indicating whether the environment has terminated.
            This tensor should usually be of shape (num_envs).
        """

    @classmethod
    def get_name(cls) -> str:
        return xax.snakecase_to_camelcase(cls.__name__)

    @functools.cached_property
    def termination_name(self) -> str:
        return self.get_name()
