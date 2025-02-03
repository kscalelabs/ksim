"""Defines the base reset class."""

import functools
from abc import ABC, abstractmethod

import equinox as eqx
import xax
from brax.mjx.base import State as MjxState
from flax import struct
from jaxtyping import PRNGKeyArray


@struct.dataclass
class ResetData:
    rng: PRNGKeyArray
    state: MjxState


class Reset(eqx.Module, ABC):
    @abstractmethod
    def __call__(self, data: ResetData) -> ResetData:
        """Resets the environment."""

    @classmethod
    def get_name(cls) -> str:
        return xax.snakecase_to_camelcase(cls.__name__)

    @functools.cached_property
    def reset_name(self) -> str:
        return self.get_name()
