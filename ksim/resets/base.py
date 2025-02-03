"""Defines the base reset class."""

import functools
from abc import ABC, abstractmethod

import xax
from brax.envs.base import State as BraxState


class Reset(ABC):
    @abstractmethod
    def __call__(self, state: BraxState) -> BraxState:
        """Resets the environment."""

    @classmethod
    def get_name(cls) -> str:
        return xax.snakecase_to_camelcase(cls.__name__)

    @functools.cached_property
    def reset_name(self) -> str:
        return self.get_name()
