"""Defines a base interface for defining reward functions."""

import functools
import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
import xax
from brax.base import State
from brax.envs.base import State as BraxState

from ksim.utils.data import BuilderData

logger = logging.getLogger(__name__)


class Reward(eqx.Module, ABC):
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

    def post_accumulate(self, reward: jnp.ndarray) -> jnp.ndarray:
        """Runs a post-epoch accumulation step.

        This function is called after the reward has been accumulated over the
        entire epoch. It can be used to normalize the reward, or apply some
        accumulation function - for example, you might might want to only
        start providing the reward or penalty after a certain number of steps
        have passed.

        Args:
            reward: The accumulated reward over the epoch.
        """
        return reward

    @abstractmethod
    def __call__(self, prev_state: BraxState, action: jnp.ndarray, next_state: State) -> jnp.ndarray: ...

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def reward_name(self) -> str:
        return self.get_name()


T = TypeVar("T", bound=Reward)


class RewardBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, data: BuilderData) -> T:
        """Builds a reward from a MuJoCo model.

        Args:
            data: The data to build the reward from.

        Returns:
            A reward that can be applied to a state.
        """
