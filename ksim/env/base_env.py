"""Base JAX centric environment class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.types import EnvState, KScaleActionModelType
from ksim.model.formulations import ActionModel, ActorCriticModel
from ksim.model.types import ActionLogProbFn


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
        model: ActorCriticModel,
        params: PyTree,
        rng: PRNGKeyArray,
        num_steps: int,
        num_envs: int,
    ) -> EnvState: ...

    @abstractmethod
    def unroll_trajectories_and_render(
        self,
        rng: PRNGKeyArray,
        num_steps: int,
        render_dir: Path,
        actions: KScaleActionModelType | ActionModel | None = None,
        width: int = 640,
        height: int = 480,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], EnvState]: ...

    @property
    @abstractmethod
    def observation_size(self) -> int: ...

    @property
    @abstractmethod
    def action_size(self) -> int: ...
