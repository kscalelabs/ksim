"""Base JAX centric environment class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax
from jaxtyping import Array, PRNGKeyArray


@jax.tree_util.register_dataclass
@dataclass
class EnvState:
    """Base environment state class."""

    obs: dict[str, Array]
    reward: Array
    done: Array
    info: dict[str, Any]


class BaseEnv(ABC):
    """Base environment class."""

    @abstractmethod
    def reset(self, rng: PRNGKeyArray) -> EnvState: ...

    @abstractmethod
    def step(self, prev_state: EnvState, action: Array, rng: Array | None = None) -> EnvState:
        """Step the environment.

        Args:
            prev_state: The previous state of the environment.
            action: The action to take.
            rng: Optional random key for stochastic environments.

        Returns:
            The next state of the environment.
        """
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
