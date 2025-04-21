"""Defines a task for training a policy using AMP, building on PPO."""

__all__ = [
    "AMPConfig",
    "AMPTask",
    "AMPReward",
]

from abc import abstractmethod
from dataclasses import dataclass, replace as dataclass_replace
from typing import Generic, TypeVar

import attrs
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.rewards import Reward
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.task.rl import RolloutConstants, RolloutEnvState, RolloutSharedState
from ksim.types import LoggedTrajectory, RewardState, Trajectory

DISCRIMINATOR_OUTPUT_KEY = "_discriminator_output"


@jax.tree_util.register_dataclass
@dataclass
class AMPConfig(PPOConfig):
    """Configuration for Adversarial Motion Prior training."""

    # Discriminator parameters
    num_discriminator_updates: int = xax.field(
        value=1,
        help="The number of times to pass the discriminator.",
    )
    w_gp: float = xax.field(
        value=10.0,
        help="Gradient penalty coefficient for discriminator (WGAN-GP style).",
    )
    discriminator_learning_rate: float = xax.field(
        value=1e-4,
        help="Learning rate for the discriminator.",
    )


Config = TypeVar("Config", bound=AMPConfig)


@attrs.define(frozen=True, kw_only=True)
class AMPReward(Reward):
    """Reward based on discriminator output for AMP training."""

    def get_reward(self, trajectory: Trajectory) -> Array:
        if trajectory.aux_outputs is None or DISCRIMINATOR_OUTPUT_KEY not in trajectory.aux_outputs:
            raise ValueError(
                "AMPReward auxiliary output is missing! Make sure you are using it within the context of an AMP task, "
                "which populates the auxiliary output for you."
            )

        discriminator_logits = trajectory.aux_outputs[DISCRIMINATOR_OUTPUT_KEY]

        # LSGAN-based reward: r = max(0, 1 - 0.25 * (D(s, s') - 1)^2)
        reward = jnp.maximum(0.0, 1.0 - 0.25 * jnp.square(discriminator_logits - 1.0))

        return reward


class AMPTask(PPOTask[Config], Generic[Config]):
    """Adversarial Motion Prior task.

    This task extends PPO to include adversarial training with a discriminator
    that tries to distinguish between real motion data and policy-generated motion.
    """

    @abstractmethod
    def get_real_motions(self) -> PyTree:
        """Loads the set of real motions.

        This should load N real motions into memory.

        Returns:
            The motions as a PyTree, most likely an array with shape (B, T, N).
        """

    @abstractmethod
    def call_discriminator(self, model: PyTree, motion: PyTree) -> Array:
        """Calls the discriminator on a given motion.

        Args:
            model: The model returned by `get_model`
            motion: The motion in question, either the real motion from the
                dataset or a motion derived from a trajectory.

        Returns:
            The discriminator logits, as an array with with shape (T). We
            convert these to a probability using a sigmoid activation function.
        """

    @abstractmethod
    def trajectory_to_motion(self, trajectory: Trajectory) -> PyTree:
        """Converts a trajectory to a motion.

        Args:
            trajectory: The trajectory to convert (with no batch dimension)

        Returns:
            The motion derived from the trajectory, for example, something like
            `trajectory.qpos` as an array. This should match whatever the real
            motions are.
        """

    def motion_to_qpos(self, motion: PyTree) -> Array:
        """Converts a motion to `qpos` array.

        This function is just used for replaying the motion on the robot model
        for visualization purposes.

        Args:
            motion: The motion, without the batch dimension.

        Returns:
            The `qpos` array, with shape (T, N).
        """
        raise NotImplementedError(
            "`motion_to_qpos(motion: PyTree) -> Array` is not implemented, so "
            "you cannot visualize the motion dataset. You should implement "
            "this function in your downstream class, depending on how you are "
            "representing your motions."
        )

    @xax.jit(static_argnames=["self", "rollout_constants"], jit_level=3)
    def step_engine(
        self,
        rollout_constants: RolloutConstants,
        rollout_env_state: RolloutEnvState,
        rollout_shared_state: RolloutSharedState,
    ) -> tuple[Trajectory, RolloutEnvState]:
        transition, next_rollout_env_state = super().step_engine(
            rollout_constants=rollout_constants,
            rollout_env_state=rollout_env_state,
            rollout_shared_state=rollout_shared_state,
        )

        # Recombines the mutable and static parts of the model.
        model = eqx.combine(rollout_shared_state.model_arr, rollout_constants.model_static)

        # Runs the discriminator on the trajectory.
        motion = self.trajectory_to_motion(transition)
        discriminator_logits = self.call_discriminator(model, motion)

        # Adds the discriminator output to the auxiliary outputs.
        aux_outputs = transition.aux_outputs.unfreeze() if transition.aux_outputs else {}
        aux_outputs[DISCRIMINATOR_OUTPUT_KEY] = discriminator_logits
        transition = dataclass_replace(transition, aux_outputs=xax.FrozenDict(aux_outputs))

        return transition, next_rollout_env_state

    def update_model(
        self,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectories: Trajectory,
        rewards: RewardState,
        rollout_env_states: RolloutEnvState,
        rollout_shared_state: RolloutSharedState,
        rollout_constants: RolloutConstants,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, PyTree, xax.FrozenDict[str, Array], LoggedTrajectory]:
        """Runs PPO updates on a given set of trajectory batches, with AMP updates as well.

        This extends the PPO update to include adversarial training with the
        discriminator, and applies a few additional checks.

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
        if all(not isinstance(reward, AMPReward) for reward in rollout_constants.rewards):
            raise ValueError("AMPReward is missing! Make sure to add it to your `get_rewards` function!")

        # Applies PPO updates.
        model_arr, opt_state, next_model_carrys, metrics, logged_traj = super().update_model(
            optimizer=optimizer,
            opt_state=opt_state,
            trajectories=trajectories,
            rewards=rewards,
            rollout_env_states=rollout_env_states,
            rollout_shared_state=rollout_shared_state,
            rollout_constants=rollout_constants,
            rng=rng,
        )

        # Gets the real and fake motions.
        real_motions = self.get_real_motions()
        fake_motions = jax.vmap(self.trajectory_to_motion, in_axes=0)(trajectories)
        chex.assert_trees_all_equal_sizes(real_motions, fake_motions)

        carry_training_state = (rollout_shared_state.model_arr, opt_state, rng)

        # Applies gradient updates across all batches num_discriminator_passes times.
        carry_training_state, (metrics, trajs_for_logging) = jax.lax.scan(
            update_model_across_batches, carry_training_state, length=self.config.num_discriminator_updates
        )

        return model_arr, opt_state, next_model_carrys, metrics, logged_traj
