"""Base JAX centric environment class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import jax
import numpy as np
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from omegaconf import MISSING

from ksim.env.types import EnvState, PhysicsData, PhysicsModel
from ksim.model.base import ActorCriticAgent
from ksim.rewards import Reward
from ksim.terminations import Termination


@jax.tree_util.register_dataclass
@dataclass
class BaseEnvConfig(xax.Config):
    # robot model configuration options
    robot_model_name: str = xax.field(value=MISSING, help="The name of the model to use.")
    robot_model_scene: str = xax.field(value="smooth", help="The scene to use for the model.")
    render_camera: str = xax.field(value="tracking_camera", help="The camera to use for rendering.")
    render_width: int = xax.field(value=640, help="The width of the rendered image.")
    render_height: int = xax.field(value=480, help="The height of the rendered image.")
    render_dir: Path = xax.field(value="render", help="The directory to save rendered images to.")
    viz_action: str = xax.field(value="policy", help="The action to use for visualization.")
    ctrl_dt: float = xax.field(value=0.02, help="The control time step.")
    dt: float = xax.field(value=0.001, help="The simulation time step.")


Config = TypeVar("Config", bound=BaseEnvConfig)


class BaseEnv(ABC, Generic[Config]):
    """Base environment class with functions designed to be scannable.

    This is why we return the physics carry data from many functions. The way
    to conceptualize this is that `unroll_trajectories` is the main function.
    The other functions serve to guide scannable implementations.
    """

    config: Config

    def __init__(self, config: Config) -> None:
        self.config = config

    @abstractmethod
    def get_init_physics_model(self) -> PhysicsModel:
        """Get the initial physics model for the environment (L)."""

    @abstractmethod
    def get_dummy_env_states(self, num_envs: int) -> EnvState:
        """Get the dummy environment states for the environment (EL)."""

    @abstractmethod
    def reset(
        self,
        model: ActorCriticAgent,
        variables: PyTree[Array],
        rng: PRNGKeyArray,
        physics_model_L: PhysicsModel,
    ) -> tuple[EnvState, PhysicsData | None]:
        """Reset the environment (EL)."""

    @abstractmethod
    def step(
        self,
        model: ActorCriticAgent,
        variables: PyTree[Array],
        env_state_L_t_minus_1: EnvState,
        rng: PRNGKeyArray,
        physics_data_L_t: PhysicsData,
        physics_model_L: PhysicsModel,
    ) -> tuple[EnvState, PhysicsData | None]:
        """Step the environment (EL)."""

    @abstractmethod
    def unroll_trajectories(
        self,
        model: ActorCriticAgent,
        variables: PyTree[Array],
        rng: PRNGKeyArray,
        num_steps: int,
        num_envs: int,
        env_state_EL_t_minus_1: EnvState,
        physics_data_EL_t: PhysicsData,
        physics_model_L: PhysicsModel,
        return_intermediate_data: bool = False,
    ) -> tuple[EnvState, PhysicsData, Array]:
        """Returns env state trajectory (TEL) and physics data (EL or TEL).

        If return_intermediate_data is True, the physics data is returned as a
        trajectory (TEL). Otherwise, only the final physics data is returned (for
        memory efficiency).
        """

    @abstractmethod
    def render_trajectory(
        self,
        model: ActorCriticAgent,
        variables: PyTree[Array],
        rng: PRNGKeyArray,
        *,
        num_steps: int,
        width: int = 640,
        height: int = 480,
        camera: int | None = None,
    ) -> tuple[list[np.ndarray], EnvState]:
        """Render trajectory and return list of images."""

    @property
    @abstractmethod
    def observation_size(self) -> int:
        """Get the size of the observation space."""

    @property
    @abstractmethod
    def action_size(self) -> int:
        """Get the size of the action space."""
