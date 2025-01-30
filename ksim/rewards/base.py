"""Defines a base interface for defining reward functions."""

import functools
import logging
from abc import abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
import xax

from ksim.state.base import State

logger = logging.getLogger(__name__)

Tstate = TypeVar("Tstate", bound=State)


class Reward(eqx.Module, Generic[Tstate]):
    """Base class for defining reward functions."""

    scale: float

    def __init__(self, scale: float) -> None:
        self.scale = scale

        # Reward functions should end with either "Reward" or "Penalty", which
        # we use here to check if the scale is positive or negative.
        name = self.reward_name
        if name.lower().endswith("reward"):
            if self.scale < 0:
                logger.warning("Reward function %s has a negative scale: %f", name, self.scale)
        elif name.lower().endswith("penalty"):
            if self.scale > 0:
                logger.warning("Penalty function %s has a positive scale: %f", name, self.scale)
        else:
            logger.warning("Reward function %s does not end with 'Reward' or 'Penalty': %f", name, self.scale)

    @abstractmethod
    def __call__(self, state: Tstate) -> jnp.ndarray: ...

    @classmethod
    def get_name(cls) -> str:
        return xax.snakecase_to_camelcase(cls.__name__)

    @functools.cached_property
    def reward_name(self) -> str:
        return self.get_name()
