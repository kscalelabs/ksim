"""Defines the base observation class."""

import functools
from abc import ABC, abstractmethod

import equinox as eqx
import jax.numpy as jnp
import xax
from brax.mjx.base import State as MjxState


class Observation(eqx.Module, ABC):
    @abstractmethod
    def __call__(self, state: MjxState) -> jnp.ndarray:
        """Resets the environment."""

    @classmethod
    def get_name(cls) -> str:
        return xax.snakecase_to_camelcase(cls.__name__)

    @functools.cached_property
    def observation_name(self) -> str:
        return self.get_name()
