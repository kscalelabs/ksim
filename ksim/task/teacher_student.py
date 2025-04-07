"""Teacher-student training for default humanoid walking task.

We take a trained teacher policy and distill it into an LSTM-based student
policy with a KL-divergence loss, using lots of domain randomization to
help it transfer to the real world effectively.

We structure this as a supervised learning task, where we collect a large
dataset of state-action pairs in simulation using a powerful teacher policy,
then use this dataset to fine-tune a student policy using a more restricted
set of observations and other constrants which are helpful for making the
policy transfer to the real world.

In practice, the dataset of trajectories can grow quite large, and it is
impractical to write a large dataset to disk. Instead, we use the teacher model
to generate trajectories on-the-fly. This has the added benefit that the
student model is only updated on new samples.
"""

__all__ = [
    "TeacherStudentConfig",
    "TeacherStudentTask",
    "TeacherVariables",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import distrax
import jax
import xax
from jaxtyping import PRNGKeyArray, PyTree

from ksim.types import Trajectory


@jax.tree_util.register_dataclass
@dataclass
class TeacherVariables:
    action_dist_tj: distrax.Distribution


@jax.tree_util.register_dataclass
@dataclass
class StudentVariables:
    action_dist_tj: distrax.Distribution


@jax.tree_util.register_dataclass
@dataclass
class TeacherStudentConfig(xax.Config):
    # Batching parameters.
    num_passes: int = xax.field(
        value=1,
        help="The number of update passes over the set of trajectories",
    )


Config = TypeVar("Config", bound=TeacherStudentConfig)


class TeacherStudentTask(xax.Task[Config], Generic[Config], ABC):
    """Base class for teacher-student tasks."""

    @abstractmethod
    def get_teacher_distribution(
        self,
        model: PyTree,
        trajectory: Trajectory,
        rng: PRNGKeyArray,
    ) -> TeacherVariables:
        """Gets the teacher outputs for the given trajectory.

        This function should call the teacher model, which should output a
        distribution over the action space. We will later update the student
        distribution to match the teacher distribution.

        Args:
            model: The user-provided model.
            trajectory: The trajectory for the distribution, with shape (T, *).
            rng: A random seed.

        Returns:
            The teacher distribution variables, with shape (T, *A).
        """

    @abstractmethod
    def get_student_distribution(
        self,
        model: PyTree,
        trajectory: Trajectory,
        rng: PRNGKeyArray,
    ) -> StudentVariables:
        """Gets the student outputs for the given trajectory.

        This function should call the student model, which should output a
        distribution over the action space. We will later update the student
        distribution to match the teacher distribution.

        Args:
            model: The user-provided model.
            trajectory: The trajectory for the distribution, with shape (T, *).
            rng: A random seed.

        Returns:
            The student distribution variables, with shape (T, *A).
        """
