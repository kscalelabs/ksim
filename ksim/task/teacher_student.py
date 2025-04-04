"""Teacher-student training for default humanoid walking task.

We take a trained teacher policy and distill it into an LSTM-based student
policy with a KL-divergence loss, using lots of domain randomization to
help it transfer to the real world effectively.
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
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.task.rl import RLConfig, RLTask
from ksim.types import Rewards, SingleTrajectory, Trajectory


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
        """Runs teacher-student updates on a given set of trajectory batches.

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
        # Shuffling causes a strange kernel caching issue on GPUs.
        # rng, indices_rng = jax.random.split(rng)
        # indices = jax.random.permutation(indices_rng, trajectories.done.shape[0])
        indices = jnp.arange(trajectories.done.shape[0])
        indices = indices.reshape(self.num_batches, self.batch_size)

        # Gets the on-policy variables for each trajectory before updating the model.
        model = eqx.combine(model_arr, model_static)

        def on_policy_scan_fn(
            carry: PRNGKeyArray,
            xt: Array,
        ) -> tuple[PRNGKeyArray, TeacherVariables]:
            trajectory_batch = jax.tree.map(lambda x: x[xt], trajectories)
            rng, policy_vars_rng = jax.random.split(carry)
            policy_vars_rngs = jax.random.split(policy_vars_rng, trajectory_batch.done.shape[0])
            policy_vars_fn = jax.vmap(self.get_teacher_distribution, in_axes=(None, 0, 0))
            teacher_variables: TeacherVariables = policy_vars_fn(model, trajectory_batch, policy_vars_rngs)
            teacher_variables = jax.tree.map(jax.lax.stop_gradient, teacher_variables)
            return rng, teacher_variables

        rng, teacher_variables = jax.lax.scan(on_policy_scan_fn, rng, indices)
        teacher_variables = jax.tree.map(lambda x: x.reshape(x.shape[0] * x.shape[1], *x.shape[2:]), teacher_variables)

        raise NotImplementedError
