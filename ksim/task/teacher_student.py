"""Defines a standard task interface for training reinforcement learning agents."""

__all__ = [
    "TeacherStudentVariables",
    "TeacherStudentConfig",
    "TeacherStudentTask",
]

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Mapping, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.task.rl import RLConfig, RLTask, RolloutConstants, RolloutEnvState, RolloutSharedState
from ksim.types import (
    LoggedTrajectory,
    Rewards,
    Trajectory,
)

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class TeacherStudentVariables:
    log_probs: Array
    values: Array
    entropy: Array | None = None
    aux_losses: Mapping[str, Array] | None = None


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
    """Base class for reinforcement learning tasks."""

    @abstractmethod
    def get_ppo_variables(
        self,
        model: PyTree,
        trajectory: Trajectory,
        model_carry: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[PPOVariables, PyTree]:
        """Gets the variables required for computing PPO loss.

        Args:
            model: The user-provided model.
            trajectory: The trajectory to get PPO variables for.
            model_carry: The model carry from the previous rollout.
            rng: A random seed.

        Returns:
            The PPO variables and the next carry for the model.
        """

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
        """Runs teacher-student updates on a given set of trajectory batches.

        Args:
            optimizer: The optimizer to use.
            opt_state: The optimizer state.
            trajectories: The trajectories to update the model on. (num_envs, num_steps, leaf_dim)
            rewards: The rewards for the trajectories. (num_envs, num_steps)
            rollout_env_states: The environment variables inputs into the rollout.
            rollout_shared_state: The shared state inputs into the rollout.
            rollout_constants: The constant inputs into the rollout.
            rng: A random seed.

        Returns:
            A tuple containing the updated parameters, optimizer state, next
            model carry, metrics, and the single trajectory to log.
        """
        # We preserve rollout ordering and split batches by envs.
        indices = jnp.arange(trajectories.done.shape[0])  # (num_envs)
        indices_by_batch = indices.reshape(self.num_batches, self.batch_size)  # (num_batches, rollouts per batch)

        model = eqx.combine(rollout_shared_state.model_arr, rollout_constants.model_static)

        on_policy_rngs = jax.random.split(rng, self.config.num_envs)
        on_policy_variables, _ = jax.vmap(self.get_ppo_variables, in_axes=(None, 0, 0, 0))(
            model, trajectories, rollout_env_states.model_carry, on_policy_rngs
        )  # (num_envs, num_steps, ppo_vars)

        # Loops over the trajectory batches and applies gradient updates.
        def update_model_in_batch(
            carry_training_state: tuple[PyTree, optax.OptState, PRNGKeyArray],
            batch_indices: Array,
        ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], tuple[xax.FrozenDict[str, Array], LoggedTrajectory]]:
            model_arr, opt_state, rng = carry_training_state
            rng, batch_rng = jax.random.split(rng)

            # Gets the current batch of trajectories and rewards.
            trajectory_batch = jax.tree.map(lambda x: x[batch_indices], trajectories)
            reward_batch = jax.tree.map(lambda x: x[batch_indices], rewards)
            carry_batch = jax.tree.map(lambda x: x[batch_indices], rollout_env_states.model_carry)
            on_policy_variables_batch = jax.tree.map(lambda x: x[batch_indices], on_policy_variables)

            model_arr, opt_state, metrics, logged_traj = self._single_step(
                model_arr=model_arr,
                model_static=rollout_constants.model_static,
                optimizer=optimizer,
                opt_state=opt_state,
                trajectories=trajectory_batch,
                rewards=reward_batch,
                init_carry=carry_batch,
                on_policy_variables=on_policy_variables_batch,
                rng=batch_rng,
            )

            return (model_arr, opt_state, rng), (metrics, logged_traj)

        # Applies N steps of gradient updates.
        def update_model_accross_batches(
            carry_training_state: tuple[PyTree, optax.OptState, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], tuple[xax.FrozenDict[str, Array], LoggedTrajectory]]:
            carry_training_state, (metrics, trajs_for_logging) = jax.lax.scan(
                update_model_in_batch, carry_training_state, indices_by_batch
            )

            # Each batch saves one trajectory for logging, get the last.
            traj_for_logging = jax.tree.map(lambda x: x[-1], trajs_for_logging)

            return carry_training_state, (metrics, traj_for_logging)

        carry_training_state = (rollout_shared_state.model_arr, opt_state, rng)

        # Applies gradient update accross all batches num_passes times.
        carry_training_state, (metrics, trajs_for_logging) = jax.lax.scan(
            update_model_accross_batches, carry_training_state, length=self.config.num_passes
        )

        # Get the last logged trajectory accross all full dataset passes.
        logged_traj = jax.tree.map(lambda x: x[-1], trajs_for_logging)

        # Getting the next model carry using the updated model.
        # Yes, this does recompute the PPO variables, but the impact is small.
        off_policy_rngs = jax.random.split(rng, self.config.num_envs)
        _, next_model_carrys = jax.vmap(self.get_ppo_variables, in_axes=(None, 0, 0, 0))(
            model, trajectories, rollout_env_states.model_carry, off_policy_rngs
        )

        model_arr, opt_state, _ = carry_training_state
        return model_arr, opt_state, next_model_carrys, metrics, logged_traj
