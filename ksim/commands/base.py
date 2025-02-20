"""Defines the base command class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
import xax
from jaxtyping import PRNGKeyArray

from ksim.utils.data import BuilderData


class Command(eqx.Module, ABC):
    @abstractmethod
    def __call__(self, rng: PRNGKeyArray) -> jnp.ndarray:
        """Gets the command to perform."""

    @eqx.filter_jit
    def update(self, prev_command: jnp.ndarray, rng: PRNGKeyArray, time: jnp.ndarray) -> jnp.ndarray:
        """Optionally updates the command to a new command."""
        return prev_command

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def command_name(self) -> str:
        return self.get_name()


T = TypeVar("T", bound=Command)


class CommandBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, data: BuilderData) -> T:
        """Builds a command from a MuJoCo model.

        Args:
            data: The data to build the command from.

        Returns:
            A command that can be applied to a state.
        """
