"""Base JAX centric environment class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax
from jaxtyping import Array, PRNGKeyArray

from ksim.env.types import EnvState


class BaseEnv(ABC):
    """Base environment class."""

    @abstractmethod
    def reset(self, rng: PRNGKeyArray) -> EnvState: ...

    @abstractmethod
    def step(
        self,
        prev_state: EnvState,
        action: Array,
        rng: PRNGKeyArray,
        action_log_prob: Array,
    ) -> EnvState:
        """Step the environment."""
        ...

    @abstractmethod
    def unroll_trajectories(
        self,
        action_log_prob_fn: Callable[[EnvState, PRNGKeyArray], Tuple[Array, Array]],
        rng: PRNGKeyArray,
        num_steps: int,
        num_envs: int,
        **kwargs: Any,
    ) -> EnvState: ...

    @property
    @abstractmethod
    def observation_size(self) -> int: ...

    @property
    @abstractmethod
    def action_size(self) -> int: ...
