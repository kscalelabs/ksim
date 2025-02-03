"""Defines the base observation class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import jax.numpy as jnp
import xax
from brax.mjx.base import State

Tstate = TypeVar("Tstate", bound=State)


class Observation(ABC, Generic[Tstate]):
    @abstractmethod
    def __call__(self, state: Tstate) -> jnp.ndarray:
        """Resets the environment."""

    @classmethod
    def get_name(cls) -> str:
        return xax.snakecase_to_camelcase(cls.__name__)

    @functools.cached_property
    def observation_name(self) -> str:
        return self.get_name()
