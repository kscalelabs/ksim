"""Default humanoid mjx env."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Collection, Literal, Tuple, TypeVar, cast, get_args

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xax
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx
from mujoco.renderer import Renderer
from mujoco_scenes.mjcf import load_mjmodel

from ksim.env.base_env import BaseEnv, EnvState
from ksim.env.builders.commands import Command, CommandBuilder
from ksim.env.builders.observation import Observation, ObservationBuilder
from ksim.env.builders.resets import Reset, ResetBuilder, ResetData
from ksim.env.builders.rewards import Reward, RewardBuilder
from ksim.env.builders.terminations import Termination, TerminationBuilder
from ksim.env.mjx.actuators.mit_actuator import MITPositionActuators
from ksim.env.mjx.mjx_env import KScaleEnvConfig, MjxEnv
from ksim.env.mjx.types import MjxEnvState
from ksim.utils.mujoco import make_mujoco_mappings
from ksim.utils.robot_model import get_model_and_metadata

logger = logging.getLogger(__name__)

T = TypeVar("T")


# The new stateless environment – note that we do not call any stateful methods.
class DefaultHumanoidEnv(MjxEnv):
    """An environment for massively parallel rollouts, stateless to obj state and system parameters.

    In this design:
      - All state (a MjxEnvState) is passed in and returned by reset and step.
      - The underlying Mujoco model (here referred to as `mjx_model`) is provided to step/reset.
      - Rollouts are performed by vectorizing (vmap) the reset and step functions,
        with a final trajectory of shape (time, num_envs, ...).
      - The step wrapper only computes a reset (via jax.lax.cond) if the done flag is True.
    """

    def __init__(
        self,
        config: KScaleEnvConfig,
        terminations: Collection[Termination | TerminationBuilder],
        resets: Collection[Reset | ResetBuilder],
        rewards: Collection[Reward | RewardBuilder],
        observations: Collection[Observation | ObservationBuilder],
        commands: Collection[Command | CommandBuilder] = (),
    ) -> None:
        self.config = config

        if self.config.max_action_latency < self.config.min_action_latency:
            raise ValueError(
                f"Maximum action latency ({self.config.max_action_latency}) must be greater than "
                f"minimum action latency ({self.config.min_action_latency})"
            )
        if self.config.min_action_latency < 0:
            raise ValueError(f"Action latency ({self.config.min_action_latency}) must be non-negative")

        self.min_action_latency_step = round(self.config.min_action_latency / self.config.dt)
        self.max_action_latency_step = round(self.config.max_action_latency / self.config.dt)

        # getting the robot model and metadata
        robot_model_path, robot_model_metadata = asyncio.run(
            get_model_and_metadata(
                self.config.robot_model_name,
                cache=not self.config.ignore_cached_urdf,
            )
        )

        logger.info("Loading robot model %s", robot_model_path)
        mj_model = load_mjmodel(robot_model_path, self.config.robot_model_scene)
        self.mujoco_mappings = make_mujoco_mappings(mj_model)
        self.actuators = MITPositionActuators(
            actuators_metadata=robot_model_metadata.actuators,
            mujoco_mappings=self.mujoco_mappings,
        )

        assert self.config.ctrl_dt % self.config.dt == 0, "ctrl_dt must be a multiple of dt"
        self._expected_dt_per_ctrl_dt = int(self.config.ctrl_dt / self.config.dt)
