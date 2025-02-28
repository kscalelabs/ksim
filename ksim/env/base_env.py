"""Base JAX centric environment class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.types import EnvState, PhysicsData
from ksim.model.formulations import ActorCriticModel


class BaseEnv(ABC):
    """Base environment class."""

    @abstractmethod
    def get_dummy_env_state(  # exists for compilation purposes
        self,
        rng: PRNGKeyArray,
    ) -> EnvState: ...

    @abstractmethod
    def reset(
        self,
        model: ActorCriticModel,
        params: PyTree,
        rng: PRNGKeyArray,
    ) -> EnvState: ...

    @abstractmethod
    def step(
        self,
        model: ActorCriticModel,
        params: PyTree,
        prev_state: EnvState,
        rng: PRNGKeyArray,
        action_log_prob: Array,
    ) -> EnvState: ...

    @abstractmethod
    def unroll_trajectories(
        self,
        model: ActorCriticModel,
        params: PyTree,
        rng: PRNGKeyArray,
        num_steps: int,
        num_envs: int,
        return_data: bool = False,
    ) -> tuple[EnvState, PhysicsData]: ...

    @abstractmethod
    def render_trajectory(
        self,
        trajectory: list[PhysicsData],
        width: int = 640,
        height: int = 480,
    ) -> list[np.ndarray]: ...

    @property
    @abstractmethod
    def observation_size(self) -> int: ...

    @property
    @abstractmethod
    def action_size(self) -> int: ...
