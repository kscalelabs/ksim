"""Defines the base reset class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import xax

from ksim.state.base import State

Tstate = TypeVar("Tstate", bound=State)


class Reset(ABC, Generic[Tstate]):
    @abstractmethod
    def __call__(self, state: Tstate) -> Tstate:
        """Resets the environment."""

    @classmethod
    def get_name(cls) -> str:
        return xax.snakecase_to_camelcase(cls.__name__)

    @functools.cached_property
    def reset_name(self) -> str:
        return self.get_name()
