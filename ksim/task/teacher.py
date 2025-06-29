"""Defines a teacher mixin for tasks."""

__all__ = [
    "StudentTask",
    "save_teacher",
    "StudentConfig",
    "TeacherReward",
]

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generic, TypeVar

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.rewards import Reward
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.task.rl import (
    RolloutConstants,
    RolloutEnvState,
    RolloutSharedState,
)
from ksim.types import Trajectory

TEACHER_OUTPUT_KEY = "_teacher_output"

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class StudentConfig(PPOConfig):
    """Configuration for teacher-student training."""

    # Teacher model path
    teacher_model_path: str = xax.field(
        value="",
        help="The path to the teacher model directory.",
    )


Config = TypeVar("Config", bound=StudentConfig)


@attrs.define(frozen=True, kw_only=True)
class TeacherReward(Reward):
    """Reward based on teacher output for teacher-student training."""

    def get_reward(self, trajectory: Trajectory) -> Array:
        if trajectory.aux_outputs is None or TEACHER_OUTPUT_KEY not in trajectory.aux_outputs:
            raise ValueError(
                "TeacherReward auxiliary output is missing! Make sure you are using it "
                "within the context of a teacher-student task, "
                "which populates the auxiliary output for you."
            )

        teacher_dist = trajectory.aux_outputs[TEACHER_OUTPUT_KEY]
        student_action = trajectory.action

        assert isinstance(teacher_dist, distrax.Distribution)

        log_prob = teacher_dist.log_prob(student_action)
        log_prob_max = teacher_dist.log_prob(teacher_dist.mode())  # value at mode
        reward = jnp.exp(log_prob - log_prob_max)  # 1.0 at the teacher mean
        return reward.mean(axis=-1)


class StudentTask(PPOTask[Config], Generic[Config], ABC):
    """Student task.

    This task extends PPO to include teacher-student training.
    """

    _teacher_policy: PyTree | None = None

    def run(self) -> None:
        """Load the teacher model before running the task."""
        self.load_teacher(Path(self.config.teacher_model_path), self.get_teacher_template())
        super().run()

    @abstractmethod
    def get_teacher_template(self) -> PyTree:
        """Get the teacher model template."""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def run_teacher(self, obs: xax.FrozenDict[str, Array], cmd: xax.FrozenDict[str, Array]) -> distrax.Distribution:
        """Run the teacher model.

        obs and cmd have shape (timesteps, ...)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def load_teacher(self, path: Path, like: PyTree) -> None:
        """Load the teacher model from a path."""
        self._teacher_policy = eqx.tree_deserialise_leaves(path / "teacher.eqx", like=like)

    @property
    def teacher_policy(self) -> PyTree:
        if self._teacher_policy is None:
            raise ValueError("Teacher model not loaded! Call load_teacher() first.")

        return self._teacher_policy

    def postprocess_trajectory(
        self,
        constants: RolloutConstants,
        env_states: RolloutEnvState,
        shared_state: RolloutSharedState,
        trajectory: Trajectory,
        rng: PRNGKeyArray,
    ) -> Trajectory:
        trajectory = super().postprocess_trajectory(
            constants=constants,
            env_states=env_states,
            shared_state=shared_state,
            trajectory=trajectory,
            rng=rng,
        )

        action_dist = self.run_teacher(trajectory.obs, trajectory.command)

        # adds the teacher distribution to the aux outputs
        aux_outputs = trajectory.aux_outputs.unfreeze() if trajectory.aux_outputs else {}
        aux_outputs[TEACHER_OUTPUT_KEY] = action_dist
        trajectory = replace(trajectory, aux_outputs=xax.FrozenDict(aux_outputs))

        return trajectory


def save_teacher(policy: PyTree, path: Path) -> None:
    """Save the teacher model to a path."""
    os.makedirs(path, exist_ok=True)
    eqx.tree_serialise_leaves(path / "teacher.eqx", policy)
