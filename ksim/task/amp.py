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
from kmv.app.viewer import DefaultMujocoViewer, QtViewer
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
    get_default_viewer,
    get_qt_viewer,
)
from ksim.types import PhysicsModel, Trajectory

DISCRIMINATOR_OUTPUT_KEY = "_discriminator_output"
REAL_MOTIONS_KEY = "_real_motions"

logger = logging.getLogger(__name__)


def _loop_slice(one_clip: Array, start: Array, window_t: int) -> Array:
    """Return a T-long window starting at `start`, looping if clip shorter."""
    t_real = one_clip.shape[0]
    idx = (jnp.arange(window_t) + start) % t_real
    return one_clip[idx]


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
    amp_reference_batch_size: int = xax.field(
        value=32,
        help="The batch size for reference motion batching in AMP training.",
    )
    amp_grad_penalty_coef: float = xax.field(
        value=10.0,
        help="The coefficient for the gradient penalty in AMP training.",
    )
    amp_reference_noise: float = xax.field(
        value=0.01,
        help="The noise to add to the reference motion batch in AMP training.",
    )
    amp_reference_noise_min_multiplier: float = xax.field(
        value=1.0,
        help="Decay the noise level by this multiplier at the maximum curriculum level.",
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
        reward = jnp.maximum(0.0, 1.0 - 0.25 * jnp.square(discriminator_logits - 1.0))
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
        slowdown: float = 1.0,
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
            slowdown: The slowdown factor for the motion viewer.
        """
        save_path = self.exp_dir / "renders" / f"render_{time.monotonic()}" if save_renders else None
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)

        with self, jax.disable_jit():
            self.set_loggers()

            # Loads the Mujoco model and logs some information about it.
            mj_model = self.get_mujoco_model()
            mj_model = self.set_mujoco_model_opts(mj_model)
            ref_data = mujoco.MjData(mj_model)
            mujoco_info = OmegaConf.to_yaml(DictConfig(self.get_mujoco_model_info(mj_model)))
            self.logger.log_file("mujoco_info.yaml", mujoco_info)

            viewer: DefaultMujocoViewer | QtViewer
            # Creates the viewer.
            if save_path is None:
                viewer = get_qt_viewer(mj_model=mj_model, config=self.config)
            else:
                viewer = get_default_viewer(mj_model=mj_model, config=self.config)

            # Gets the real motions and converts them to qpos arrays.
            real_motions = self.get_real_motions(mj_model)
            qpos = self.motion_to_qpos(real_motions)
            chex.assert_shape(qpos, (None, None, mj_model.nq))
            qpos = qpos.reshape(-1, qpos.shape[-1])

            iterator: Iterable[int] = tqdm.trange(qpos.shape[0] if num_steps is None else min(num_steps, qpos.shape[0]))
            frames: list[np.ndarray] = []

            if loop:
                iterator = itertools.cycle(iterator)

            target_time = time.time() + self.config.ctrl_dt * slowdown
            try:
                for i in iterator:
                    # Logs the frames to render.
                    if save_path is None:
                        assert isinstance(viewer, QtViewer)
                        viewer.push_state(
                            qpos=np.array(qpos[i]), qvel=np.zeros_like(ref_data.qvel), sim_time=i * self.config.ctrl_dt
                        )
                        logger.debug("Sleeping for %s seconds", target_time - time.time())
                        time.sleep(max(0, target_time - time.time()))
                        target_time += self.config.ctrl_dt * slowdown
                    else:
                        assert isinstance(viewer, DefaultMujocoViewer)
                        viewer.data.qpos[:] = np.array(qpos[i])
                        mujoco.mj_forward(viewer.model, viewer.data)
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
    def call_discriminator(self, model: PyTree, motion: PyTree, rng: PRNGKeyArray) -> Array:
        """Calls the discriminator on a given motion.

        Args:
            model: The model returned by `get_model`
            motion: The motion in question, either the real motion from the
                dataset or a motion derived from a trajectory.
            rng: The random number generator.

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
        rng: PRNGKeyArray,
    ) -> Trajectory:
        disc_rng, postprocess_rng = jax.random.split(rng)

        trajectory = super().postprocess_trajectory(
            constants=constants,
            env_states=env_states,
            shared_state=shared_state,
            trajectory=trajectory,
            rng=postprocess_rng,
        )

        # Recombines the mutable and static parts of the discriminator model.
        disc_model_arr = shared_state.model_arrs[1]
        disc_model_static = constants.model_statics[1]
        disc_model = eqx.combine(disc_model_arr, disc_model_static)

        # Runs the discriminator on the trajectory.
        motion = self.trajectory_to_motion(trajectory)
        discriminator_logits = self.call_discriminator(disc_model, motion, disc_rng)
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

    def _grad_penalty(
        self,
        disc_model: PyTree,
        real_motions: PyTree,  # (B, T, ...)
        rng: PRNGKeyArray,
    ) -> Array:
        """R1 gradient penalty  (w.r.t. the *input* features)."""

        def disc_on_motion(motion: PyTree, key: PRNGKeyArray) -> Array:
            return self.call_discriminator(disc_model, motion, key).sum()

        grad_fn = jax.grad(disc_on_motion, argnums=0)

        batch_size = jax.tree_util.tree_leaves(real_motions)[0].shape[0]
        keys = jax.random.split(rng, batch_size)
        grads = jax.vmap(grad_fn)(real_motions, keys)

        def leaf_sqnorm(leaf: Array) -> Array:
            return jnp.sum(jnp.square(leaf), axis=tuple(range(1, leaf.ndim)))

        per_sample_sq = sum(jax.tree_util.tree_leaves(jax.tree_util.tree_map(leaf_sqnorm, grads)))  # (B,)

        return jnp.mean(per_sample_sq)

    @xax.jit(static_argnames=["self", "model_static"], jit_level=JitLevel.RL_CORE)
    def _get_amp_disc_loss_and_metrics(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        trajectories: Trajectory,
        real_motions: PyTree,
        carry: RLLoopCarry,
        rng: PRNGKeyArray,
    ) -> tuple[Array, xax.FrozenDict[str, Array]]:
        """Computes the PPO loss and additional metrics.

        Args:
            model_arr: The array part of the discriminator model.
            model_static: The static part of the discriminator model.
            trajectories: The trajectories to compute the loss on.
            real_motions: The real motions to compute the loss on.
            carry: The carry from the previous step.
            rng: The random number generator.

        Returns:
            A tuple containing the loss value as a scalar, a dictionary of
            metrics to log, and the single trajectory to log.
        """
        model = eqx.combine(model_arr, model_static)

        sim_motions = self.trajectory_to_motion(trajectories)

        # Computes the discriminator loss.
        disc_fn = xax.vmap(self.call_discriminator, in_axes=(None, 0, 0), jit_level=JitLevel.RL_CORE)
        real_disc_rng, sim_disc_rng, gp_rng = jax.random.split(rng, 3)
        real_batch = jax.tree_util.tree_leaves(real_motions)[0].shape[0]
        sim_batch = jax.tree_util.tree_leaves(sim_motions)[0].shape[0]
        real_disc_logits = disc_fn(model, real_motions, jax.random.split(real_disc_rng, real_batch))
        sim_disc_logits = disc_fn(model, sim_motions, jax.random.split(sim_disc_rng, sim_batch))
        real_disc_loss, sim_disc_loss = self.get_disc_losses(real_disc_logits, sim_disc_logits)

        gp_loss = self.config.amp_grad_penalty_coef / 2 * self._grad_penalty(model, real_motions, gp_rng)

        disc_loss = real_disc_loss + sim_disc_loss + gp_loss

        disc_metrics = {
            "real_logits": real_disc_logits,
            "sim_logits": sim_disc_logits,
            "gp_loss": gp_loss,
        }

        return disc_loss, xax.FrozenDict(disc_metrics)

    @xax.jit(static_argnames=["self", "model_static"], jit_level=JitLevel.RL_CORE)
    def _get_disc_metrics_and_grads(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        trajectories: Trajectory,
        real_motions: PyTree,
        carry: RLLoopCarry,
        rng: PRNGKeyArray,
    ) -> tuple[xax.FrozenDict[str, Array], PyTree]:
        loss_fn = jax.grad(self._get_amp_disc_loss_and_metrics, argnums=0, has_aux=True)
        loss_fn = xax.jit(static_argnums=[1], jit_level=JitLevel.RL_CORE)(loss_fn)
        grads, metrics = loss_fn(model_arr, model_static, trajectories, real_motions, carry, rng)
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
        rng, rng_disc, rng_noise = jax.random.split(rng, 3)
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

        real_motions_full = carry.shared_state.aux_values[REAL_MOTIONS_KEY]

        sim_t = trajectories.done.shape[-1]

        real_batch = self._make_real_batch(
            motions=real_motions_full,
            window_t=sim_t,
            batch_b=self.config.amp_reference_batch_size,
            rng=rng_disc,
        )

        # Computes the metrics and PPO gradients.
        disc_metrics, grads = self._get_disc_metrics_and_grads(
            model_arr=model_arr,
            model_static=model_static,
            trajectories=trajectories,
            real_motions=real_batch,
            carry=carry,
            rng=rng_noise,
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

    @staticmethod  # ensure consistent calling convention
    def _make_real_batch(
        motions: PyTree,
        window_t: int,
        batch_b: int,
        rng: PRNGKeyArray,
    ) -> PyTree:
        """Sample a batch of windowed motion snippets from a PyTree of motions.

        Args:
            motions: A PyTree whose leaves are arrays of shape (B, T, ...).
            window_t: Length of the temporal window to sample.
            batch_b: Number of windows to sample.
            rng: PRNG key used for sampling.

        Returns:
            A PyTree with the same structure as ``motions`` whose leaves have
            shape (batch_b, window_t, ...).
        """
        num_motions = jax.tree_util.tree_leaves(motions)[0].shape[0]

        keys = jax.random.split(rng, batch_b + 1)
        clip_key, sample_keys = keys[0], keys[1:]

        # Sample which clip each element in the batch comes from.
        clip_idx = jax.random.randint(clip_key, (batch_b,), 0, num_motions)

        batch_clips = jax.tree_util.tree_map(lambda arr: arr[clip_idx], motions)

        def _sample_single(clip: PyTree, rng_key: PRNGKeyArray) -> PyTree:
            """Samples an unbiased window from a single motion clip."""
            # Length of the real clip (may differ across clips).
            t_real = jax.tree_util.tree_leaves(clip)[0].shape[0]
            start = jax.random.randint(rng_key, (), 0, t_real)  # unbiased start index
            return jax.tree_util.tree_map(lambda arr: _loop_slice(arr, start, window_t), clip)

        return jax.vmap(_sample_single)(batch_clips, sample_keys)
