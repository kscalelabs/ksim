"""Defines a standard task interface for training reinforcement learning agents."""

__all__ = [
    "TeacherStudentConfig",
    "TeacherStudentTask",
]

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

import jax
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from omegaconf import MISSING

from ksim.task.rl import RLConfig, RLTask, RolloutConstants, RolloutEnvState, RolloutSharedState
from ksim.types import (
    LoggedTrajectory,
    Rewards,
    Trajectory,
)

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class TeacherStudentConfig(RLConfig):
    # Training parameters.
    num_envs: int = xax.field(
        value=MISSING,
        help="The number of training environments to run in parallel.",
    )
    batch_size: int = xax.field(
        value=MISSING,
        help="The number of model update batches per trajectory batch. ",
    )
    rollout_length_seconds: float = xax.field(
        value=MISSING,
        help="The number of seconds to rollout each environment during training.",
    )


Config = TypeVar("Config", bound=TeacherStudentConfig)


class TeacherStudentTask(RLTask[Config], Generic[Config], ABC):
    """Base class for reinforcement learning tasks."""

    def update_model(
        self,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectories: Trajectory,
        rewards: Rewards,
        rollout_env_states: RolloutEnvState,
        rollout_shared_state: RolloutSharedState,
        rollout_constants: RolloutConstants,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, PyTree, xax.FrozenDict[str, Array], LoggedTrajectory]:
        raise NotImplementedError("TeacherStudentTask does not implement update_model.")
