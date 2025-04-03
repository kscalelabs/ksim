"""Teacher-student training for default humanoid walking task.

We take a trained teacher policy and distill it into an LSTM-based student
policy with a KL-divergence loss, using lots of domain randomization to
help it transfer to the real world effectively.
"""

__all__ = [
    "TeacherStudentConfig",
    "TeacherStudentTask",
    "TeacherStudentVariables",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import distrax
import jax
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.task.rl import RLConfig, RLTask
from ksim.types import Rewards, SingleTrajectory, Trajectory


@jax.tree_util.register_dataclass
@dataclass
class TeacherStudentVariables:
    teacher_distribution: distrax.Distribution
    student_distribution: distrax.Distribution


@jax.tree_util.register_dataclass
@dataclass
class TeacherStudentConfig(RLConfig):
    # Batching parameters.
    num_passes: int = xax.field(
        value=1,
        help="The number of update passes over the set of trajectories",
    )


Config = TypeVar("Config", bound=TeacherStudentConfig)


class TeacherStudentTask(RLTask[Config], Generic[Config], ABC):
    """Base class for teacher-student tasks."""

    @abstractmethod
    def get_teacher_student_outputs(
        self,
        model: PyTree,
        trajectories: Trajectory,
        rng: PRNGKeyArray,
    ) -> TeacherStudentVariables:
        """Gets the teacher and student outputs for the given trajectories.

        This function should call the teacher and student models, which should
        each output distributions over the action space. We will later update
        the student distribution to match the teacher distribution.

        Args:
            model: The user-provided model.
            trajectories: The batch of trajectories to get probabilities for.
            rng: A random seed.

        Returns:
            The log probabilities of the given actions, with shape (B, T, *A).
        """

    def update_model(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectories: Trajectory,
        rewards: Rewards,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, xax.FrozenDict[str, Array], SingleTrajectory]:
        """Runs PPO updates on a given set of trajectory batches.

        Args:
            model_arr: The mutable part of the model to update.
            model_static: The static part of the model to update.
            optimizer: The optimizer to use.
            opt_state: The optimizer state.
            trajectories: The trajectories to update the model on.
            rewards: The rewards for the trajectories.
            rng: A random seed.

        Returns:
            A tuple containing the updated parameters, optimizer state, metrics,
            and the single trajectory to log.
        """
        # TODO
        raise NotImplementedError
