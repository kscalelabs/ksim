"""Defines a task for training a policy using AMP, building on PPO."""

__all__ = [
    "AMPConfig",
    "AMPTask",
    "AMPReward",
]

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace as dataclass_replace
from typing import Generic, TypeVar

import attrs
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.rewards import Reward
from ksim.task.ppo import PPOConfig, PPOTask, PPOVariables
from ksim.task.rl import (
    LoggedTrajectory,
    RewardState,
    RLLoopCarry,
    RLLoopConstants,
    RolloutConstants,
    RolloutEnvState,
    RolloutSharedState,
)
from ksim.types import Trajectory

DISCRIMINATOR_OUTPUT_KEY = "_discriminator_output"

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class AMPConfig(PPOConfig):
    """Configuration for Adversarial Motion Prior training."""

    # Discriminator parameters
    num_discriminator_updates: int = xax.field(
        value=1,
        help="The number of times to pass the discriminator.",
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


class AMPTask(PPOTask[Config], Generic[Config], ABC):
    """Adversarial Motion Prior task.

    This task extends PPO to include adversarial training with a discriminator
    that tries to distinguish between real motion data and policy-generated motion.
    """

    @abstractmethod
    def get_policy_model(self, key: PRNGKeyArray) -> PyTree:
        """Returns the policy model."""

    @abstractmethod
    def get_discriminator_model(self, key: PRNGKeyArray) -> PyTree:
        """Returns the discriminator model."""

    def get_model(self, key: PRNGKeyArray) -> tuple[PyTree, PyTree]:
        policy_key, discriminator_key = jax.random.split(key)
        return self.get_policy_model(policy_key), self.get_discriminator_model(discriminator_key)

    @abstractmethod
    def get_policy_optimizer(self) -> optax.GradientTransformation:
        """Returns the optimizer for the policy model."""

    @abstractmethod
    def get_discriminator_optimizer(self) -> optax.GradientTransformation:
        """Returns the optimizer for the discriminator model."""

    def get_optimizer(self) -> tuple[optax.GradientTransformation, optax.GradientTransformation]:
        return (
            self.get_policy_optimizer(),
            self.get_discriminator_optimizer(),
        )

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

    @xax.jit(static_argnames=["self", "constants"], jit_level=3)
    def step_engine(
        self,
        constants: RolloutConstants,
        env_states: RolloutEnvState,
        shared_state: RolloutSharedState,
    ) -> tuple[Trajectory, RolloutEnvState]:
        transition, next_env_state = super().step_engine(
            constants=constants,
            env_states=env_states,
            shared_state=shared_state,
        )

        # Recombines the mutable and static parts of the discriminator model.
        disc_model_arr = shared_state.model_arrs[1]
        disc_model_static = constants.model_statics[1]
        disc_model = eqx.combine(disc_model_arr, disc_model_static)

        # Runs the discriminator on the trajectory.
        motion = self.trajectory_to_motion(transition)
        discriminator_logits = self.call_discriminator(disc_model, motion)

        # Adds the discriminator output to the auxiliary outputs.
        aux_outputs = transition.aux_outputs.unfreeze() if transition.aux_outputs else {}
        aux_outputs[DISCRIMINATOR_OUTPUT_KEY] = discriminator_logits
        transition = dataclass_replace(transition, aux_outputs=xax.FrozenDict(aux_outputs))

        return transition, next_env_state

    @xax.jit(static_argnames=["self", "model_static"], jit_level=5)
    def _get_amp_disc_loss_and_metrics(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        trajectories: Trajectory,
    ) -> tuple[Array, xax.FrozenDict[str, Array]]:
        """Computes the PPO loss and additional metrics.

        Args:
            model_arr: The mutable part of the model to optimize.
            model_static: The static part of the model to optimize.
            trajectories: The batch of trajectories to compute the loss and metrics for.
            rewards: The rewards for the trajectories.
            init_carry: The initial carry for the model.
            on_policy_variables: The PPO variables from the on-policy rollout.
            rng: A random seed.

        Returns:
            A tuple containing the loss value as a scalar, a dictionary of
            metrics to log, and the single trajectory to log.
        """
        model = eqx.combine(model_arr, model_static)

        real_motions = self.get_real_motions()
        sim_motions = self.trajectory_to_motion(trajectories)

        # Computes the discriminator loss.
        real_disc_logits = self.call_discriminator(model, real_motions)
        sim_disc_logits = self.call_discriminator(model, sim_motions)

        # Computes the discriminator loss, using LSGAN.
        real_disc_loss = 0.5 * jnp.mean(jnp.square(real_disc_logits - 1.0))
        sim_disc_loss = 0.5 * jnp.mean(jnp.square(sim_disc_logits))

        disc_loss = real_disc_loss + sim_disc_loss

        disc_metrics = {
            "real_logits": real_disc_logits,
            "sim_logits": sim_disc_logits,
        }

        return disc_loss, xax.FrozenDict(disc_metrics)

    @xax.jit(static_argnames=["self", "model_static"], jit_level=3)
    def _get_disc_metrics_and_grads(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        trajectories: Trajectory,
    ) -> tuple[dict[str, Array], PyTree]:
        loss_fn = jax.grad(self._get_amp_disc_loss_and_metrics, argnums=0, has_aux=True)
        loss_fn = xax.jit(static_argnums=[1], jit_level=3)(loss_fn)
        grads, metrics = loss_fn(model_arr, model_static, trajectories)
        return metrics, grads

    @xax.jit(static_argnames=["self", "constants"], jit_level=4)
    def _single_step(
        self,
        trajectories: Trajectory,
        rewards: RewardState,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        on_policy_variables: PPOVariables,
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopCarry, xax.FrozenDict[str, Array], LoggedTrajectory]:
        carry, metrics, logged_traj = super()._single_step(
            trajectories=trajectories,
            rewards=rewards,
            constants=constants,
            carry=carry,
            on_policy_variables=on_policy_variables,
            rng=rng,
        )

        # Gets the discriminator model and optimizer.
        model_arr = carry.shared_state.model_arrs[1]
        model_static = constants.constants.model_statics[1]
        optimizer = constants.optimizer[1]
        opt_state = carry.opt_state[1]

        # Computes the metrics and PPO gradients.
        disc_metrics, grads = self._get_disc_metrics_and_grads(
            model_arr=model_arr,
            model_static=model_static,
            trajectories=trajectories,
        )

        # Applies the gradients with clipping.
        new_model_arr, new_opt_state, disc_grad_metrics = self.apply_gradients_with_clipping(
            model_arr=model_arr,
            grads=grads,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Updates the carry with the new model and optimizer states.
        carry = dataclass_replace(
            carry,
            shared_state=dataclass_replace(
                carry.shared_state,
                model_arrs=xax.tuple_insert(carry.shared_state.model_arrs, 1, new_model_arr),
            ),
            opt_state=xax.tuple_insert(carry.opt_state, 1, new_opt_state),
        )

        # Gets the metrics dictionary.
        metrics: xax.FrozenDict[str, Array] = xax.FrozenDict(metrics.unfreeze() | disc_metrics | disc_grad_metrics)

        return carry, metrics, logged_traj

    def update_model(
        self,
        *,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        trajectories: Trajectory,
        rewards: RewardState,
        rng: PRNGKeyArray,
    ) -> tuple[
        RLLoopCarry,
        xax.FrozenDict[str, Array],
        LoggedTrajectory,
    ]:
        # Checks that AMPReward is used.
        if not any(isinstance(r, AMPReward) for r in constants.constants.rewards):
            raise ValueError("AMPReward is not used! This is required for AMP training.")

        return super().update_model(
            constants=constants,
            carry=carry,
            trajectories=trajectories,
            rewards=rewards,
            rng=rng,
        )
