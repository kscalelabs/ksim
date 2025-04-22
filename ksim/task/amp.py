"""Defines a task for training a policy using AMP, building on PPO."""

__all__ = [
    "AMPConfig",
    "AMPTask",
    "AMPReward",
]

import bdb
import datetime
import logging
import signal
import sys
import textwrap
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace as dataclass_replace
from threading import Thread
from typing import Generic, TypeVar

import attrs
import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from omegaconf import DictConfig, OmegaConf

from ksim.rewards import Reward
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.task.rl import RolloutConstants, RolloutEnvState, RolloutSharedState, _RLTrainLoopCarry, get_viewer
from ksim.types import LoggedTrajectory, Metrics, PhysicsModel, RewardState, Trajectory

DISCRIMINATOR_OUTPUT_KEY = "_discriminator_output"

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiscriminatorEnvState(RolloutEnvState):
    pass


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiscriminatorSharedState(RolloutSharedState):
    """Variables used across all environments."""

    disc_model_arr: PyTree
    real_motions: PyTree


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiscriminatorConstants(RolloutConstants):
    """Constants for the rollout loop."""

    disc_model_static: PyTree


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _AMPTrainLoopCarry(_RLTrainLoopCarry):
    rollout_env_states: DiscriminatorEnvState
    rollout_shared_state: DiscriminatorSharedState
    disc_opt_state: optax.OptState


@jax.tree_util.register_dataclass
@dataclass
class AMPConfig(PPOConfig):
    """Configuration for Adversarial Motion Prior training."""

    # Discriminator parameters
    num_discriminator_updates: int = xax.field(
        value=1,
        help="The number of times to pass the discriminator.",
    )
    wasserstein_gradient_penalty: float = xax.field(
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

    @xax.jit(static_argnames=["self", "rollout_constants"], jit_level=3)
    def step_engine(
        self,
        rollout_constants: DiscriminatorConstants,
        rollout_env_state: DiscriminatorEnvState,
        rollout_shared_state: DiscriminatorSharedState,
    ) -> tuple[Trajectory, RolloutEnvState]:
        transition, next_rollout_env_state = super().step_engine(
            rollout_constants=rollout_constants,
            rollout_env_state=rollout_env_state,
            rollout_shared_state=rollout_shared_state,
        )

        # Recombines the mutable and static parts of the model.
        model = eqx.combine(rollout_shared_state.disc_model_arr, rollout_constants.disc_model_static)

        # Runs the discriminator on the trajectory.
        motion = self.trajectory_to_motion(transition)
        discriminator_logits = self.call_discriminator(model, motion)

        # Adds the discriminator output to the auxiliary outputs.
        aux_outputs = transition.aux_outputs.unfreeze() if transition.aux_outputs else {}
        aux_outputs[DISCRIMINATOR_OUTPUT_KEY] = discriminator_logits
        transition = dataclass_replace(transition, aux_outputs=xax.FrozenDict(aux_outputs))

        return transition, next_rollout_env_state

    def update_model_with_discriminator(
        self,
        *,
        pol_optimizer: optax.GradientTransformation,
        pol_opt_state: optax.OptState,
        disc_optimizer: optax.GradientTransformation,
        disc_opt_state: optax.OptState,
        trajectories: Trajectory,
        rewards: RewardState,
        rollout_env_states: DiscriminatorEnvState,
        rollout_shared_state: DiscriminatorSharedState,
        rollout_constants: DiscriminatorConstants,
        rng: PRNGKeyArray,
    ) -> tuple[
        tuple[PyTree, optax.OptState, PyTree],
        tuple[PyTree, optax.OptState],
        xax.FrozenDict[str, Array],
        LoggedTrajectory,
    ]:
        """Runs PPO updates on a given set of trajectory batches, with AMP updates as well.

        This extends the PPO update to include adversarial training with the
        discriminator, and applies a few additional checks.

        Args:
            pol_optimizer: The optimizer to use for the policy model.
            pol_opt_state: The optimizer state for the policy model.
            disc_optimizer: The optimizer to use for the discriminator model.
            disc_opt_state: The optimizer state for the discriminator model.
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

        pol_rng, disc_rng = jax.random.split(rng)

        # Applies PPO updates.
        pol_model_arr, pol_opt_state, pol_next_model_carry, metrics, logged_traj = self.update_model(
            optimizer=pol_optimizer,
            opt_state=pol_opt_state,
            trajectories=trajectories,
            rewards=rewards,
            rollout_env_states=rollout_env_states,
            rollout_shared_state=rollout_shared_state,
            rollout_constants=rollout_constants,
            rng=pol_rng,
        )

        # Gets the real and fake motions.
        fake_motions = jax.vmap(self.trajectory_to_motion, in_axes=0)(trajectories)
        chex.assert_trees_all_equal_sizes(rollout_shared_state.real_motions, fake_motions)

        carry_training_state = (rollout_shared_state.disc_model_arr, disc_opt_state, disc_rng)

        # Applies gradient updates across all batches num_discriminator_passes times.
        carry_training_state, (metrics, trajs_for_logging) = jax.lax.scan(
            update_model_across_batches, carry_training_state, length=self.config.num_discriminator_updates
        )

        disc_model_arr, disc_opt_state, _ = carry_training_state

        return (
            (pol_model_arr, pol_opt_state, pol_next_model_carry),
            (disc_model_arr, disc_opt_state),
            metrics,
            logged_traj,
        )

    @xax.jit(
        static_argnames=[
            "self",
            "pol_optimizer",
            "disc_optimizer",
            "rollout_constants",
        ],
        jit_level=1,
    )
    def _amp_train_loop_step(
        self,
        carry: _AMPTrainLoopCarry,
        pol_optimizer: optax.GradientTransformation,
        disc_optimizer: optax.GradientTransformation,
        rollout_constants: DiscriminatorConstants,
        state: xax.State,
        rng: PRNGKeyArray,
    ) -> tuple[_AMPTrainLoopCarry, Metrics, LoggedTrajectory]:
        """Runs a single step of the RL training loop."""

        def single_step_fn(
            carry_i: _AMPTrainLoopCarry,
            rng: PRNGKeyArray,
        ) -> tuple[_AMPTrainLoopCarry, tuple[Metrics, LoggedTrajectory]]:
            # Rolls out a new trajectory.
            vmapped_unroll = jax.vmap(self._single_unroll, in_axes=(None, 0, None))
            trajectories, rewards, next_rollout_env_states = vmapped_unroll(
                rollout_constants,
                carry_i.rollout_env_states,
                carry_i.rollout_shared_state,
            )

            # The reward carry is updated every rollout using the last episode.
            next_reward_carry = rewards.carry

            # Runs update on the previous trajectory.
            (
                (pol_model_arr, pol_opt_state, pol_next_model_carry),
                (disc_model_arr, disc_opt_state),
                train_metrics,
                logged_traj,
            ) = self.update_model_with_discriminator(
                pol_optimizer=pol_optimizer,
                pol_opt_state=carry_i.opt_state,
                disc_optimizer=disc_optimizer,
                disc_opt_state=carry_i.disc_opt_state,
                trajectories=trajectories,
                rewards=rewards,
                rollout_env_states=carry_i.rollout_env_states,
                rollout_shared_state=carry_i.rollout_shared_state,
                rollout_constants=rollout_constants,
                rng=rng,
            )

            # Store all the metrics to log.
            metrics = Metrics(
                train=train_metrics,
                reward=xax.FrozenDict(self.get_reward_metrics(trajectories, rewards)),
                termination=xax.FrozenDict(self.get_termination_metrics(trajectories)),
                curriculum_level=carry_i.rollout_env_states.curriculum_state.level,
            )

            # Steps the curriculum.
            curriculum_state = rollout_constants.curriculum(
                trajectory=trajectories,
                rewards=rewards,
                training_state=state,
                prev_state=carry_i.rollout_env_states.curriculum_state,
            )

            next_carry = _AMPTrainLoopCarry(
                opt_state=pol_opt_state,
                rollout_env_states=DiscriminatorEnvState(
                    commands=next_rollout_env_states.commands,
                    physics_state=next_rollout_env_states.physics_state,
                    randomization_dict=next_rollout_env_states.randomization_dict,
                    model_carry=pol_next_model_carry,
                    reward_carry=next_reward_carry,
                    obs_carry=next_rollout_env_states.obs_carry,
                    curriculum_state=curriculum_state,
                    rng=next_rollout_env_states.rng,
                ),
                rollout_shared_state=DiscriminatorSharedState(
                    model_arr=pol_model_arr,
                    physics_model=carry_i.rollout_shared_state.physics_model,
                    real_motions=carry_i.rollout_shared_state.real_motions,
                    disc_model_arr=disc_model_arr,
                ),
                disc_opt_state=disc_opt_state,
            )

            return next_carry, (metrics, logged_traj)

        # Runs the single step function over the inputs.
        rngs = jax.random.split(rng, self.config.epochs_per_log_step)
        carry, (metrics, logged_traj) = jax.lax.scan(single_step_fn, carry, rngs)

        # Convert any array with more than one element to a histogram.
        metrics = jax.tree.map(lambda x: self.get_histogram(x) if isinstance(x, Array) and x.size > 1 else x, metrics)

        # Only get final trajectory and rewards.
        logged_traj = jax.tree.map(lambda arr: arr[-1], logged_traj)

        return carry, metrics, logged_traj

    def run_training(self) -> None:
        """Wraps the training loop to include discriminator models."""
        with self:
            rng = self.prng_key()
            self.set_loggers()

            if xax.is_master():
                Thread(target=self.log_state, daemon=True).start()

            # Gets the model and optimizer variables.
            rng, model_rng = jax.random.split(rng)
            models, optimizers, opt_states, state = self.load_initial_state(model_rng, load_optimizer=True)

            if len(models) != 2 or len(optimizers) != 2 or len(opt_states) != 2:
                raise ValueError(
                    "AMP training expects two models, two optimizers and two optimizer states. "
                    "The first model should be the policy model, and the second model should be the discriminator. "
                    f"Found {len(models)} models, {len(optimizers)} optimizers and {len(opt_states)} optimizer states."
                )
            pol_model, disc_model = models
            pol_optimizer, disc_optimizer = optimizers
            pol_opt_state, disc_opt_state = opt_states

            # Logs model and optimizer information.
            logger.log(
                xax.LOG_PING,
                "Policy model size: %s parameters",
                f"{xax.get_pytree_param_count(pol_model):,}",
            )
            logger.log(
                xax.LOG_PING,
                "Discriminator model size: %s parameters",
                f"{xax.get_pytree_param_count(disc_model):,}",
            )
            logger.log(
                xax.LOG_PING,
                "Policy optimizer size: %s parameters",
                f"{xax.get_pytree_param_count(pol_optimizer):,}",
            )
            logger.log(
                xax.LOG_PING,
                "Discriminator optimizer size: %s parameters",
                f"{xax.get_pytree_param_count(disc_optimizer):,}",
            )

            # Loads the Mujoco model and logs some information about it.
            mj_model: PhysicsModel = self.get_mujoco_model()
            mj_model = self.set_mujoco_model_opts(mj_model)
            mujoco_info = OmegaConf.to_yaml(DictConfig(self.get_mujoco_model_info(mj_model)))
            self.logger.log_file("mujoco_info.yaml", mujoco_info)

            # Loads the MJX model, and initializes the loop variables.
            mjx_model = self.get_mjx_model(mj_model)
            randomizers = self.get_physics_randomizers(mjx_model)

            # JAX requires that we partition the model into mutable and static
            # parts in order to use lax.scan, so that `arr` can be a PyTree.
            pol_model_arr, pol_model_static = eqx.partition(pol_model, self.model_partition_fn)
            disc_model_arr, disc_model_static = eqx.partition(disc_model, self.model_partition_fn)

            rollout_constants = self._get_discriminator_constants(
                mj_model=mjx_model,
                pol_model_static=pol_model_static,
                disc_model_static=disc_model_static,
                argmax_action=False,
            )

            carry = _AMPTrainLoopCarry(
                opt_state=pol_opt_state,
                rollout_env_states=self._get_discriminator_env_state(
                    rng=rng,
                    rollout_constants=rollout_constants,
                    mj_model=mjx_model,
                    randomizers=randomizers,
                ),
                rollout_shared_state=self._get_discriminator_shared_state(
                    mj_model=mjx_model,
                    pol_model_arr=pol_model_arr,
                    disc_model_arr=disc_model_arr,
                ),
                disc_opt_state=disc_opt_state,
            )

            # Creates the markers.
            markers = self.get_markers(
                commands=rollout_constants.commands,
                observations=rollout_constants.observations,
                rewards=rollout_constants.rewards,
                randomizers=randomizers,
            )

            # Creates the viewer.
            small_viewer = get_viewer(
                mj_model=mj_model,
                config=self.config,
                mode="offscreen",
                is_small=True,
            )
            full_viewer = get_viewer(
                mj_model=mj_model,
                config=self.config,
                mode="offscreen",
                is_small=False,
            )

            state = self.on_training_start(state)

            def _save() -> None:
                pol_model = eqx.combine(carry.rollout_shared_state.model_arr, rollout_constants.model_static)
                disc_model = eqx.combine(carry.rollout_shared_state.disc_model_arr, rollout_constants.disc_model_static)
                self.save_checkpoint(
                    models=[pol_model, disc_model],
                    optimizers=[pol_optimizer, disc_optimizer],
                    opt_states=[pol_opt_state, disc_opt_state],
                    state=state,
                )

            def on_exit() -> None:
                _save()

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

            # Clean up variables which are not part of the control loop.
            del pol_model_arr, pol_model_static, disc_model_arr, disc_model_static, mjx_model, randomizers

            is_first_step = True

            try:
                while not self.is_training_over(state):
                    # Runs the training loop.
                    with xax.ContextTimer() as timer:
                        valid_step = self.valid_step_timer(state)

                        state = state.replace(
                            phase="valid" if valid_step else "train",
                        )

                        state = self.on_step_start(state)

                        rng, update_rng = jax.random.split(rng)
                        carry, metrics, logged_traj = self._amp_train_loop_step(
                            carry=carry,
                            pol_optimizer=pol_optimizer,
                            disc_optimizer=disc_optimizer,
                            rollout_constants=rollout_constants,
                            state=state,
                            rng=update_rng,
                        )

                        if self.config.profile_memory:
                            carry = jax.block_until_ready(carry)
                            metrics = jax.block_until_ready(metrics)
                            logged_traj = jax.block_until_ready(logged_traj)
                            jax.profiler.save_device_memory_profile(self.exp_dir / "train_loop_step.prof")

                        self.log_train_metrics(metrics)
                        self.log_state_timers(state)

                        if self.should_checkpoint(state):
                            _save()

                        state = self.on_step_end(state)

                        # Updates the step and sample counts.
                        num_steps = self.config.epochs_per_log_step
                        num_samples = self.rollout_num_samples * self.config.epochs_per_log_step

                        if valid_step:
                            state = state.replace(
                                num_valid_steps=state.num_valid_steps + num_steps,
                                num_valid_samples=state.num_valid_samples + num_samples,
                            )

                            full_size_render = state.num_valid_steps.item() % self.config.render_full_every_n_steps == 0
                            if full_size_render:
                                self._log_logged_trajectory_video(
                                    logged_traj=logged_traj,
                                    markers=markers,
                                    viewer=full_viewer,
                                    key="trajectory",
                                )
                                self._log_logged_trajectory_graphs(
                                    logged_traj=logged_traj,
                                    log_callback=lambda key, value, namespace: self.logger.log_image(
                                        key=key, value=value, namespace=namespace
                                    ),
                                )
                            else:
                                self._log_logged_trajectory_video(
                                    logged_traj=logged_traj,
                                    markers=markers,
                                    viewer=small_viewer,
                                    key="trajectory_small",
                                )

                        else:
                            state = state.replace(
                                num_steps=state.num_steps + num_steps,
                                num_samples=state.num_samples + num_samples,
                            )

                        self.write_logs(state)

                    # Update  state with the elapsed time.
                    elapsed_time = timer.elapsed_time
                    if valid_step:
                        state = state.replace(
                            valid_elapsed_time_s=state.valid_elapsed_time_s + elapsed_time,
                        )
                    else:
                        state = state.replace(
                            elapsed_time_s=state.elapsed_time_s + elapsed_time,
                        )

                    if is_first_step:
                        is_first_step = False
                        logger.log(
                            xax.LOG_STATUS,
                            "First step time: %s",
                            xax.format_timedelta(datetime.timedelta(seconds=elapsed_time), short=True),
                        )

                # Save the checkpoint when done.
                _save()

            except xax.TrainingFinishedError:
                if xax.is_master():
                    msg = f"Finished training after {state.num_steps}steps and {state.num_samples} samples"
                    xax.show_info(msg, important=True)
                _save()

            except (KeyboardInterrupt, bdb.BdbQuit):
                if xax.is_master():
                    xax.show_info("Interrupted training", important=True)

            except BaseException:
                exception_tb = textwrap.indent(xax.highlight_exception_message(traceback.format_exc()), "  ")
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()
                _save()

            finally:
                state = self.on_training_end(state)
