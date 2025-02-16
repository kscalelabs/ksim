"""Defines the base reset class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
import xax
from brax.base import State
from flax import struct
from jaxtyping import PRNGKeyArray

from ksim.utils.data import BuilderData


@struct.dataclass
class ResetData:
    rng: PRNGKeyArray
    state: State


class Reset(eqx.Module, ABC):
    @abstractmethod
    def __call__(self, data: ResetData) -> ResetData:
        """Resets the environment."""

    @classmethod
    def get_name(cls) -> str:
        return xax.camelcase_to_snakecase(cls.__name__)

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
