"""Defines a teacher mixin for tasks."""

__all__ = [
    "TeacherMixin",
]

import bdb
import itertools
import logging
from pathlib import Path
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Generic, Iterable, TypeVar

import attrs
import chex
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
import tqdm
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from omegaconf import DictConfig, OmegaConf

from ksim.rewards import Reward
from ksim.task.ppo import PPOConfig, PPOTask, PPOVariables
from ksim.task.rl import (
    LoggedTrajectory,
    RLTask,
    RewardState,
    RLLoopCarry,
    RLLoopConstants,
    RolloutConstants,
    RolloutEnvState,
    RolloutSharedState,
    get_viewer,
)
from ksim.types import PhysicsModel, Trajectory

TEACHER_OUTPUT_KEY = "_teacher_output"

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class TeacherConfig(PPOConfig):
    """Configuration for teacher-student training."""

    # Teacher model path
    teacher_model_path: str = xax.field(
        value="",
        help="The path to the teacher model directory.",
    )


Config = TypeVar("Config", bound=TeacherConfig)

def save_teacher(task: PPOTask[Config]) -> None:
    """Save the teacher model to a path."""
    arrays, statics = task.teacher_arrays, task.teacher_statics
    eqx.tree_serialise_leaves(task.exp_dir / "teacher_policy" / "arrays.eqx", arrays)
    eqx.tree_serialise_leaves(task.exp_dir / "teacher_policy" / "statics.eqx", statics)


@attrs.define(frozen=True, kw_only=True)
class TeacherReward(Reward):
    """Reward based on teacher output for teacher-student training."""

    def get_reward(self, trajectory: Trajectory) -> Array:
        if trajectory.aux_outputs is None or TEACHER_OUTPUT_KEY not in trajectory.aux_outputs:
            raise ValueError(
                "TeacherReward auxiliary output is missing! Make sure you are using it within the context of a teacher-student task, "
                "which populates the auxiliary output for you."
            )

        teacher_dist = trajectory.aux_outputs[TEACHER_OUTPUT_KEY]
        student_action = trajectory.action

        assert isinstance(teacher_dist, distrax.Distribution)

        # compute the log probability of the student action under the teacher distribution
        log_prob = teacher_dist.log_prob(student_action)

        # reward high log probability actions
        reward = jnp.sum(log_prob, axis=-1)
        return reward


class StudentTask(PPOTask[Config], Generic[Config], ABC):
    """Student task.

    This task extends PPO to include teacher-student training.
    """

    teacher_arrays: tuple[PyTree, ...] | None = None
    teacher_statics: tuple[PyTree, ...] | None = None

    def load_teacher(self, path: Path) -> None:
        """Load the teacher model from a path."""
        self.teacher_arrays = eqx.tree_deserialise_leaves(path / "arrays.eqx")
        self.teacher_statics = eqx.tree_deserialise_leaves(path / "statics.eqx")


    def teacher_policy(self) -> PyTree:
        if self.teacher_statics is None or self.teacher_arrays is None:
            raise ValueError("Teacher model not loaded! Call load_teacher() first.")

        return eqx.combine(self.teacher_arrays, self.teacher_statics)

    def postprocess_trajectory(
        self,
        constants: RolloutConstants,
        env_states: RolloutEnvState,
        shared_state: RolloutSharedState,
        trajectory: Trajectory,
    ) -> Trajectory:
        trajectory = super().postprocess_trajectory(
                constants=constants,
                env_states=env_states,
            shared_state=shared_state,
        )

        teacher_policy = self.teacher_policy()

        action_dist = teacher_policy.run_teacher(trajectory.obs)

        chex.assert_shape(action_dist, (trajectory.num_steps, *trajectory.action_shape))

        # adds the teacher distribution to the aux outputs
        aux_outputs = trajectory.aux_outputs.unfreeze() if trajectory.aux_outputs else {}
        aux_outputs[TEACHER_OUTPUT_KEY] = action_dist
        trajectory = replace(trajectory, aux_outputs=xax.FrozenDict(aux_outputs))

        return trajectory
