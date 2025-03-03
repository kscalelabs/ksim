"""Base JAX centric environment class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
import xax
from jaxtyping import PRNGKeyArray, PyTree
from omegaconf import MISSING

from ksim.env.types import EnvState, PhysicsData
from ksim.model.formulations import ActorCriticModel


@jax.tree_util.register_dataclass
@dataclass
class BaseEnvConfig(xax.Config):
    # robot model configuration options
    robot_model_name: str = xax.field(value=MISSING, help="The name of the model to use.")
    robot_model_scene: str = xax.field(value="patch", help="The scene to use for the model.")
    render_camera: str = xax.field(value="tracking_camera", help="The camera to use for rendering.")
    render_width: int = xax.field(value=640, help="The width of the rendered image.")
    render_height: int = xax.field(value=480, help="The height of the rendered image.")
    render_dir: Path = xax.field(value="render", help="The directory to save rendered images to.")
    viz_action: str = xax.field(value="policy", help="The action to use for visualization.")


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
        prev_env_state: EnvState,
        rng: PRNGKeyArray,
        **kwargs: Any,
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
