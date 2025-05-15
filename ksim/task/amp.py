"""Defines a task for training a policy using AMP, building on PPO."""

__all__ = [
    "AMPConfig",
    "AMPTask",
    "AMPReward",
]

import bdb
import itertools
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Generic, Iterable, TypeVar

import attrs
import chex
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

from ksim.debugging import JitLevel
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
    get_viewer,
)
from ksim.types import PhysicsModel, Trajectory

DISCRIMINATOR_OUTPUT_KEY = "_discriminator_output"
REAL_MOTIONS_KEY = "_real_motions"

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class AMPConfig(PPOConfig):
    """Configuration for Adversarial Motion Prior training."""

    # Toggle this to visualize the motion on the robot.
    run_motion_viewer: bool = xax.field(
        value=False,
        help="If true, the motion will be visualized on the robot.",
    )
    run_motion_viewer_loop: bool = xax.field(
        value=True,
        help="If true, the motion will be looped.",
    )


Config = TypeVar("Config", bound=AMPConfig)


@attrs.define(frozen=True, kw_only=True)
class AMPReward(Reward):
    """Reward based on discriminator output for AMP training."""

    def get_reward(self, trajectory: Trajectory) -> Array:
        if DISCRIMINATOR_OUTPUT_KEY not in trajectory.aux_outputs:
            raise ValueError(
                "AMPReward auxiliary output is missing! Make sure you are using it within the context of an AMP task, "
                "which populates the auxiliary output for you."
            )

        discriminator_logits = trajectory.aux_outputs[DISCRIMINATOR_OUTPUT_KEY]
        reward = discriminator_logits + 1.0
        return reward


class AMPTask(PPOTask[Config], Generic[Config], ABC):
    """Adversarial Motion Prior task.

    This task extends PPO to include adversarial training with a discriminator
    that tries to distinguish between real motion data and policy-generated motion.
    """

    def run(self) -> None:
        match self.config.run_mode.lower():
            case "view_motion":
                self.run_motion_viewer(
                    num_steps=(
                        None
                        if self.config.viewer_num_seconds is None
                        else round(self.config.viewer_num_seconds / self.config.ctrl_dt)
                    ),
                    save_renders=self.config.viewer_save_renders,
                    loop=self.config.run_motion_viewer_loop,
                )

            case _:
                return super().run()

    def run_motion_viewer(
        self,
        num_steps: int | None,
        save_renders: bool = False,
        loop: bool = True,
    ) -> None:
        """Provides an easy-to-use interface for viewing motions on the robot.

        This function cycles through the set of provided motions, rendering
        them on the robot model.

        Args:
            num_steps: The number of steps to run the environment for. If not
                provided, run until the user manually terminates the
                environment visualizer.
            save_renders: If provided, save the rendered video to the given path.
            loop: If true, loop through the motions.
        """
        save_path = self.exp_dir / "renders" / f"render_{time.monotonic()}" if save_renders else None
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)

        with self, jax.disable_jit():
            self.set_loggers()

            # Loads the Mujoco model and logs some information about it.
            mj_model = self.get_mujoco_model()
            mj_model = self.set_mujoco_model_opts(mj_model)
            mujoco_info = OmegaConf.to_yaml(DictConfig(self.get_mujoco_model_info(mj_model)))
            self.logger.log_file("mujoco_info.yaml", mujoco_info)

            # Creates the viewer.
            viewer = get_viewer(mj_model=mj_model, config=self.config, save_path=save_path)

            # Gets the real motions and converts them to qpos arrays.
            real_motions = self.get_real_motions(mj_model)
            qpos = self.motion_to_qpos(real_motions)
            chex.assert_shape(qpos, (None, None, mj_model.nq))
            qpos = qpos.reshape(-1, qpos.shape[-1])

            iterator: Iterable[int] = tqdm.trange(qpos.shape[0] if num_steps is None else min(num_steps, qpos.shape[0]))
            frames: list[np.ndarray] = []

            if loop:
                iterator = itertools.cycle(iterator)

            try:
                for i in iterator:
                    # Logs the frames to render.
                    viewer.data.qpos[:] = np.array(qpos[i])
                    mujoco.mj_forward(viewer.model, viewer.data)

                    if save_path is None:
                        viewer.render()
                    else:
                        frames.append(viewer.read_pixels())

            except (KeyboardInterrupt, bdb.BdbQuit):
                logger.info("Keyboard interrupt, exiting environment loop")

            if save_path is not None:
                self._save_viewer_video(frames, save_path)

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
    def get_real_motions(self, mj_model: mujoco.MjModel) -> PyTree:
        """Loads the set of real motions.

        This should load N real motions into memory.

        Args:
            mj_model: The Mujoco model to load the motions from.

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
            trajectory: The trajectory to convert (with batch dimension)

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
            motion: The full motion, including the batch dimension.

        Returns:
            The `qpos` array, with shape (B, T, N).
        """
        raise NotImplementedError(
            "`motion_to_qpos(motion: PyTree) -> Array` is not implemented, so "
            "you cannot visualize the motion dataset. You should implement "
            "this function in your downstream class, depending on how you are "
            "representing your motions."
        )

    def _get_shared_state(
        self,
        *,
        mj_model: mujoco.MjModel,
        physics_model: PhysicsModel,
        model_arrs: tuple[PyTree, ...],
    ) -> RolloutSharedState:
        shared_state = super()._get_shared_state(
            mj_model=mj_model,
            physics_model=physics_model,
            model_arrs=model_arrs,
        )
        shared_state = replace(
            shared_state,
            aux_values=xax.FrozenDict(
                shared_state.aux_values.unfreeze()
                | {
                    REAL_MOTIONS_KEY: self.get_real_motions(mj_model),
                }
            ),
        )
        return shared_state

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
            trajectory=trajectory,
        )

        # Recombines the mutable and static parts of the discriminator model.
        disc_model_arr = shared_state.model_arrs[1]
        disc_model_static = constants.model_statics[1]
        disc_model = eqx.combine(disc_model_arr, disc_model_static)

        # Runs the discriminator on the trajectory.
        motion = self.trajectory_to_motion(trajectory)
        discriminator_logits = self.call_discriminator(disc_model, motion)
        chex.assert_equal_shape([discriminator_logits, trajectory.done])

        # Adds the discriminator output to the auxiliary outputs.
        aux_outputs = trajectory.aux_outputs.unfreeze() if trajectory.aux_outputs else {}
        aux_outputs[DISCRIMINATOR_OUTPUT_KEY] = discriminator_logits
        trajectory = replace(trajectory, aux_outputs=xax.FrozenDict(aux_outputs))

        return trajectory

    def get_disc_losses(
        self,
        real_disc_logits: Array,
        sim_disc_logits: Array,
    ) -> tuple[Array, Array]:
        real_disc_loss = jnp.mean((real_disc_logits - 1) ** 2)
        sim_disc_loss = jnp.mean((sim_disc_logits + 1) ** 2)
        return real_disc_loss, sim_disc_loss

    @xax.jit(static_argnames=["self", "model_static"], jit_level=JitLevel.RL_CORE)
    def _get_amp_disc_loss_and_metrics(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        trajectories: Trajectory,
        real_motions: PyTree,
    ) -> tuple[Array, xax.FrozenDict[str, Array]]:
        """Computes the PPO loss and additional metrics.

        Args:
            model_arr: The array part of the discriminator model.
            model_static: The static part of the discriminator model.
            trajectories: The trajectories to compute the loss on.
            real_motions: The real motions to compute the loss on.

        Returns:
            A tuple containing the loss value as a scalar, a dictionary of
            metrics to log, and the single trajectory to log.
        """
        model = eqx.combine(model_arr, model_static)

        sim_motions = self.trajectory_to_motion(trajectories)

        # Computes the discriminator loss.
        disc_fn = xax.vmap(self.call_discriminator, in_axes=(None, 0), jit_level=JitLevel.RL_CORE)
        real_disc_logits = disc_fn(model, real_motions)
        sim_disc_logits = disc_fn(model, sim_motions)
        real_disc_loss, sim_disc_loss = self.get_disc_losses(real_disc_logits, sim_disc_logits)

        disc_loss = real_disc_loss + sim_disc_loss

        disc_metrics = {
            "real_logits": real_disc_logits,
            "sim_logits": sim_disc_logits,
        }

        return disc_loss, xax.FrozenDict(disc_metrics)

    @xax.jit(static_argnames=["self", "model_static"], jit_level=JitLevel.RL_CORE)
    def _get_disc_metrics_and_grads(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        trajectories: Trajectory,
        real_motions: PyTree,
    ) -> tuple[xax.FrozenDict[str, Array], PyTree]:
        loss_fn = jax.grad(self._get_amp_disc_loss_and_metrics, argnums=0, has_aux=True)
        loss_fn = xax.jit(static_argnums=[1], jit_level=JitLevel.RL_CORE)(loss_fn)
        grads, metrics = loss_fn(model_arr, model_static, trajectories, real_motions)
        return metrics, grads

    @xax.jit(static_argnames=["self", "constants"], jit_level=JitLevel.RL_CORE)
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
            real_motions=carry.shared_state.aux_values[REAL_MOTIONS_KEY],
        )

        # Applies the gradients with clipping.
        new_model_arr, new_opt_state, disc_grad_metrics = self.apply_gradients_with_clipping(
            model_arr=model_arr,
            grads=grads,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Updates the carry with the new model and optimizer states.
        carry = replace(
            carry,
            shared_state=replace(
                carry.shared_state,
                model_arrs=xax.tuple_insert(carry.shared_state.model_arrs, 1, new_model_arr),
            ),
            opt_state=xax.tuple_insert(carry.opt_state, 1, new_opt_state),
        )

        # Gets the metrics dictionary.
        metrics: xax.FrozenDict[str, Array] = xax.FrozenDict(
            metrics.unfreeze() | disc_metrics.unfreeze() | disc_grad_metrics
        )

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
