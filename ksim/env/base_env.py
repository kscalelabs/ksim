"""Base JAX centric environment class."""

from abc import ABC, abstractmethod
from typing import Callable

from brax.envs.base import State as BraxState
from jaxtyping import Array, PRNGKeyArray


class BaseEnv(ABC):
    """Base environment class."""

    @abstractmethod
    def reset(self, rng: PRNGKeyArray) -> BraxState: ...

    @abstractmethod
    def step(self, prev_state: BraxState, action: Array) -> BraxState: ...

    @abstractmethod
    def unroll_trajectories(
        self,
        action_fn: Callable[[Array], Array],
        rng: PRNGKeyArray,
        max_trajectory_steps: int,
    ) -> BraxState: ...

    """Unroll the model for a given number of steps.

    Args:
        action_fn: The action function to unroll.
        rng: The random key.
        max_trajectory_steps: The maximum number of steps to unroll.

    Returns:
        The unrolled trajectories (num_steps, num_envs, ...).
    """

    @property
    @abstractmethod
    def observation_size(self) -> int: ...

    @property
    @abstractmethod
    def action_size(self) -> int: ...

    @property
    @abstractmethod
    def num_envs(self) -> int: ...
