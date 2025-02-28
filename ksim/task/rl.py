"""Defines a standard task interface for training reinforcement learning agents."""

import bdb
import logging
import os
import signal
import sys
import textwrap
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Thread
from typing import Generic, Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import mediapy as mp
import optax
import xax
from dpshdl.dataset import Dataset
from flax import linen as nn
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree
from omegaconf import MISSING
from PIL import Image

from ksim.builders.loggers import (
    AverageRewardLog,
    EpisodeLengthLog,
    LoggingData,
    ModelUpdateLog,
)
from ksim.env.base_env import BaseEnv, EnvState
from ksim.env.mjx.mjx_env import KScaleEnvConfig
from ksim.model.formulations import ActorCriticModel
from ksim.task.types import RolloutTimeLossComponents
from ksim.utils.jit import legit_jit
from ksim.utils.pytree import slice_pytree

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class RLConfig(KScaleEnvConfig, xax.Config):
    action: str = xax.field(
        value="train",
        help="The action to take; should be either `train` or `env`.",
    )
    max_episode_length: float = xax.field(
        value=MISSING,
        help="The maximum episode length, in seconds.",
    )
    num_steps_per_trajectory: int = xax.field(
        value=MISSING,
        help="The number of steps in a trajectory.",
    )
    num_envs: int = xax.field(
        value=MISSING,
        help="The number of training environments to run in parallel.",
    )
    default_action_model: str = xax.field(
        value="zero",
        help="The default action model to use if `actions` is not specified.",
    )
    observation_size: int = xax.field(
        value=MISSING,
        help="The size of the observation space.",
    )
    action_size: int = xax.field(
        value=MISSING,
        help="The size of the action space.",
    )
    num_learning_epochs: int = xax.field(
        value=5,
        help="Number of learning epochs per PPO update.",
    )
    minibatch_size: int = xax.field(
        value=MISSING,
        help="The size of each minibatch.",
    )


Config = TypeVar("Config", bound=RLConfig)


class RLTask(xax.Task[Config], Generic[Config], ABC):
    """Base class for reinforcement learning tasks."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.log_items = [EpisodeLengthLog(), AverageRewardLog(), ModelUpdateLog()]
        super().__init__(config)

    ####################
    # Abstract methods #
    ####################

    @abstractmethod
    def get_environment(self) -> BaseEnv: ...

    @abstractmethod
    def viz_environment(self) -> None: ...

    @abstractmethod
    def get_init_actor_carry(self) -> jnp.ndarray | None: ...

    @abstractmethod
    def get_rollout_time_loss_components(
        self,
        model: ActorCriticModel,
        params: PyTree,
        trajectory_dataset: EnvState,
    ) -> RolloutTimeLossComponents: ...

    @abstractmethod
    def model_update(
        self,
        model: ActorCriticModel,
        params: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        env_state_batch: EnvState,
        rollout_time_loss_components: RolloutTimeLossComponents,
    ) -> tuple[PyTree, optax.OptState, Array, FrozenDict[str, Array]]: ...

    ##############
    # Properties #
    ##############

    @property
    def dataset_size(self) -> int:
        """The size of the dataset."""
        return self.config.num_envs * self.config.num_steps_per_trajectory

    @property
    def num_minibatches(self) -> int:
        """The number of minibatches in the dataset."""
        assert self.dataset_size % self.config.minibatch_size == 0
        return self.dataset_size // self.config.minibatch_size

    ########################
    # XAX-specific methods #
    ########################

    def get_dataset(self, phase: Literal["train", "valid"]) -> Dataset:
        """Get the dataset for the current task."""
        raise NotImplementedError("Reinforcement learning tasks do not require datasets.")

    def get_batch_size(self) -> int:
        """Get the batch size for the current task."""
        # TODO: this is a hack for xax... need to implement mini batching properly later.
        return 1

    def run(self) -> None:
        """Highest level entry point for RL tasks, determines what to run."""
        match self.config.action:
            case "train":
                self.run_training()

            case "env":
                model, _, _, _ = self.load_initial_state(self.prng_key())
                self.run_environment(model)

            case "viz":
                self.viz_environment()

            case _:
                raise ValueError(f"Invalid action: {self.config.action}. Should be one of `train` or `env`.")

    #########################
    # Logging and Rendering #
    #########################

    def get_render_name(self, state: xax.State | None = None) -> str:
        time_string = time.strftime("%Y%m%d_%H%M%S")
        if state is None:
            return f"render_{time_string}"
        return f"render_{state.num_steps}_{time_string}"

    def run_environment(
        self,
        model: ActorCriticModel,
        # params: PyTree | None = None,
        state: xax.State | None = None,
    ) -> None:
        """Run the environment with rendering and logging."""
        with self:
            start_time = time.time()
            rng = self.prng_key()
            env = self.get_environment()
            render_name = self.get_render_name(state)
            render_dir = self.exp_dir / self.config.render_dir / render_name

            os.makedirs(render_dir, exist_ok=True)
            end_time = time.time()
            print(f"Time taken for environment setup: {end_time - start_time} seconds")

            logger.log(logging.INFO, "Rendering to %s", render_dir)

            self.set_loggers()

            key, _ = jax.random.split(rng)

            start_time = time.time()
            params = self.get_init_params(key)
            end_time = time.time()
            print(f"Time taken for parameter initialization: {end_time - start_time} seconds")

            # Unroll trajectories and collect the frames for rendering
            logger.info("Unrolling trajectories")

            start_time = time.time()
            _, traj = env.unroll_trajectories(
                model, params, rng, self.max_trajectory_steps, self.config.num_envs, return_data=True
            )
            end_time = time.time()
            print(f"Time taken for trajectory unrolling: {end_time - start_time} seconds")

            start_time = time.time()
            mjx_data_traj = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=1), traj)

            mjx_data_list = [
                jax.tree_util.tree_map(lambda x: x[i], mjx_data_traj)
                for i in range(self.max_trajectory_steps)
            ]
            end_time = time.time()
            print(f"Time taken for trajectory processing: {end_time - start_time} seconds")

            start_time = time.time()
            frames = env.render_trajectory(
                trajectory=mjx_data_list, width=self.config.render_width, height=self.config.render_height
            )
            end_time = time.time()
            print(f"Time taken for rendering frames: {end_time - start_time} seconds")

            logger.info("Saving %d frames to %s", len(frames), render_dir)

            start_time = time.time()
            for i, frame in enumerate(frames):
                frame_path = render_dir / f"frame_{i:06d}.png"
                Image.fromarray(frame).save(frame_path)
            end_time = time.time()
            print(f"Time taken for saving frames: {end_time - start_time} seconds")

            start_time = time.time()
            mp.write_video(render_dir / "trajectory.mp4", frames, fps=1 / self.config.ctrl_dt)
            end_time = time.time()
            print(f"Time taken for video creation: {end_time - start_time} seconds")

    def log_state(self, env: BaseEnv) -> None:
        super().log_state()

        # self.logger.log_file("env_state.yaml", OmegaConf.to_yaml(env.get_state()))

    def log_trajectory(self, env: BaseEnv, trajectory: EnvState) -> None:
        for plot_key, img in env.generate_trajectory_plots(trajectory):
            self.logger.log_image(plot_key, img, namespace="traj")

        frames, fps = env.render_trajectory_video(trajectory)
        self.logger.log_video("trajectory", frames, fps=fps, namespace="video")

    def get_reward_stats(self, trajectory: EnvState, env: BaseEnv) -> dict[str, jnp.ndarray]:
        reward_stats: dict[str, jnp.ndarray] = {}

        # Gets the reward statistics.
        reward = jnp.where(trajectory.done[..., None], jnp.nan, trajectory.info["all_rewards"])
        for i, (key, _) in enumerate(env.rewards):
            reward_values = reward[..., i : i + 1].astype(jnp.float32)
            reward_stats[f"{key}/mean"] = jnp.nanmean(reward_values)
            reward_stats[f"{key}/std"] = jnp.nanstd(reward_values)

        return reward_stats

    def get_termination_stats(self, trajectory: EnvState, env: BaseEnv) -> dict[str, jnp.ndarray]:
        termination_stats: dict[str, jnp.ndarray] = {}

        # Gets the termination statistics.
        termination = trajectory.info["all_dones"].max(axis=-2).astype(jnp.float32)
        termination = termination.reshape(-1, termination.shape[-1])
        max_ids = termination.argmax(axis=-1)
        for i, (key, _) in enumerate(env.terminations):
            termination_stats[key] = (max_ids == i).astype(jnp.float32).mean()

        return termination_stats

    def log_trajectory_stats(self, env: BaseEnv, trajectory: EnvState) -> None:
        for key, value in self.get_reward_stats(trajectory, env).items():
            self.logger.log_scalar(key, value, namespace="reward")
        for key, value in self.get_termination_stats(trajectory, env).items():
            self.logger.log_scalar(key, value, namespace="termination")

        # Logs the mean episode length.
        mean_episode_length_steps = (~trajectory.done).sum(axis=-1).astype(jnp.float32).mean()
        mean_episode_length_seconds = mean_episode_length_steps * self.config.ctrl_dt
        self.logger.log_scalar("mean_episode_length", mean_episode_length_seconds, namespace="stats")

    ########################
    # Training and Running #
    ########################

    def get_init_params(self, key: PRNGKeyArray, pretrained: str | None = None) -> PyTree:
        """Get the initial parameters as a PyTree: assumes flax-compatible model."""
        env = self.get_environment()
        state = env.get_dummy_env_state(key)

        if pretrained is not None:
            # TODO: implement pretrained model loading.
            raise NotImplementedError("Pretrained models are not yet implemented.")

        model_key, init_key = jax.random.split(key, 2)
        model = self.get_model(model_key)
        assert isinstance(model, nn.Module), "Model must be an Flax linen module."
        return model.init(init_key, state.obs, state.command)

    @legit_jit(static_argnames=["self", "model"])
    def apply_actor(
        self,
        model: ActorCriticModel,
        params: PyTree,
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
    ) -> Array:
        """Apply the actor model to inputs."""
        res = model.apply(params, obs=obs, cmd=cmd, method="actor")
        assert isinstance(res, Array)
        return res

    @legit_jit(static_argnames=["self", "model", "env"], compile_timeout=10)
    def get_trajectory_dataset(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env: BaseEnv,
        rng: PRNGKeyArray,
    ) -> EnvState:
        """Rollout a batch of trajectory data.

        To avoid confusion, batch comprises 1 or more unrolled trajectory states stacked
        along the first axis, and minibatches are sampled from this batch.
        """

        # TODO: implement logic to handle randomize model initialization when creating batch
        rollout = env.unroll_trajectories(
            model=model,
            params=params,
            rng=rng,
            num_steps=self.config.num_steps_per_trajectory,
            num_envs=self.config.num_envs,
        )

        def flatten_rollout_array(x: Array) -> Array:
            """Flatten a rollout array."""
            reshaped = jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:]))
            assert reshaped.shape[0] == self.config.num_steps_per_trajectory * self.config.num_envs
            return reshaped

        # flattening (num_steps, num_envs, ...) -> (num_steps * num_envs, ...) in leaves
        flattened_rollout = jax.tree_util.tree_map(flatten_rollout_array, rollout)

        return flattened_rollout

    @legit_jit(static_argnames=["self"])
    def reshuffle_rollout(
        self,
        rollout_dataset: EnvState,
        rollout_time_loss_components: RolloutTimeLossComponents,
        rng: PRNGKeyArray,
    ) -> tuple[EnvState, RolloutTimeLossComponents]:
        """Reshuffle a rollout array."""
        # Generate permutation indices
        batch_size = self.dataset_size
        permutation = jax.random.permutation(rng, batch_size)

        # Apply permutation to rollout dataset
        def permute_array(x):
            # Handle arrays with proper shape checking
            if x.shape[0] == batch_size:
                return x[permutation]
            return x

        # Apply permutation to both structures
        reshuffled_rollout_dataset = jax.tree_util.tree_map(permute_array, rollout_dataset)
        reshuffled_rollout_time_loss_components = jax.tree_util.tree_map(
            permute_array, rollout_time_loss_components
        )

        return reshuffled_rollout_dataset, reshuffled_rollout_time_loss_components

    @legit_jit(static_argnames=["self"])
    def get_minibatch(
        self,
        rollout: EnvState,
        rollout_time_loss_components: RolloutTimeLossComponents,
        minibatch_idx: Array,
    ) -> tuple[EnvState, RolloutTimeLossComponents]:
        """Get a minibatch from the rollout."""
        starting_idx = minibatch_idx * self.config.minibatch_size
        minibatched_rollout = slice_pytree(rollout, starting_idx, self.config.minibatch_size)
        minibatched_rollout_time_loss_components = slice_pytree(
            rollout_time_loss_components, starting_idx, self.config.minibatch_size
        )
        return minibatched_rollout, minibatched_rollout_time_loss_components

    def train_loop(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env: BaseEnv,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        training_state: xax.State,
    ) -> None:
        """Runs the main RL training loop."""
        rng = self.prng_key()
        rng, train_rng = jax.random.split(rng, 2)

        while not self.is_training_over(training_state):
            with self.step_context("on_step_start"):
                training_state = self.on_step_start(training_state)

            # Unrolls a trajectory.
            start_time = time.time()
            reshuffle_rng, rollout_rng = jax.random.split(train_rng)
            trajectories_dataset = self.get_trajectory_dataset(model, params, env, rollout_rng)
            rollout_time = time.time() - start_time

            # running training on minibatches
            start_time = time.time()
            rollout_time_loss_components = self.get_rollout_time_loss_components(
                model, params, trajectories_dataset
            )
            for epoch_idx in range(self.config.num_learning_epochs):
                trajectories_dataset, rollout_time_loss_components = self.reshuffle_rollout(
                    trajectories_dataset, rollout_time_loss_components, reshuffle_rng
                )
                reshuffle_rng, _ = jax.random.split(reshuffle_rng)
                for minibatch_idx in range(self.num_minibatches):
                    minibatch_idx = jnp.array(minibatch_idx)  # TODO: scanning will do this anyways
                    minibatch, rollout_time_minibatch_loss_components = self.get_minibatch(
                        trajectories_dataset, rollout_time_loss_components, minibatch_idx
                    )
                    with self.step_context("update_state"):
                        params, opt_state, loss_val, metrics = self.model_update(
                            model,
                            params,
                            optimizer,
                            opt_state,
                            minibatch,
                            rollout_time_minibatch_loss_components,
                        )

                        # log metrics from the model update
                        metric_logging_data = LoggingData(
                            trajectory=trajectories_dataset,
                            update_metrics=metrics,
                            gradients=None,
                            loss=float(loss_val),
                            training_state=training_state,
                        )

                        with self.step_context("write_logs"):
                            training_state.raw_phase = "train"
                            for log_item in self.log_items:
                                log_item(self.logger, metric_logging_data)

                            self.logger.write(training_state)
                            training_state.num_steps += 1

            model_update_time = time.time() - start_time

            # Log the time taken for the model update.
            with self.step_context("write_logs"):
                self.logger.log_scalar("rollout_time", rollout_time, namespace="⏰")
                self.logger.log_scalar("model_update_time", model_update_time, namespace="⏰")
                self.logger.write(training_state)

            start_time = time.time()
            with self.step_context("on_step_end"):
                training_state = self.on_step_end(training_state)

            if self.should_checkpoint(training_state):
                self.save_checkpoint(
                    model=params, optimizer=optimizer, opt_state=opt_state, state=training_state
                )  # Update XAX to be Flax supportive...
            end_time = time.time()
            print(f"Time taken for on_step_end and save checkpoint: {end_time - start_time} seconds")

    def run_training(self) -> None:
        """Wraps the training loop and provides clean XAX integration."""
        with self:
            key = self.prng_key()
            self.set_loggers()
            env = self.get_environment()

            if xax.is_master():
                Thread(target=self.log_state, daemon=True, args=(env,)).start()

            key, model_key = jax.random.split(key)
            model, optimizer, opt_state, training_state = self.load_initial_state(model_key)

            training_state = self.on_training_start(training_state)

            def on_exit() -> None:
                self.save_checkpoint(model, optimizer, opt_state, training_state)

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

            params = self.get_init_params(key)
            opt_state = optimizer.init(params)

            try:
                self.train_loop(
                    model=model,
                    params=params,
                    env=env,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    training_state=training_state,
                )

            except xax.TrainingFinishedError:
                if xax.is_master():
                    msg = (
                        f"Finished training after {training_state.num_steps}"
                        f"steps and {training_state.num_samples} samples"
                    )
                    xax.show_info(msg, important=True)
                self.save_checkpoint(model, optimizer, opt_state, training_state)

            except (KeyboardInterrupt, bdb.BdbQuit):
                if xax.is_master():
                    xax.show_info("Interrupted training", important=True)

            except BaseException:
                exception_tb = textwrap.indent(xax.highlight_exception_message(traceback.format_exc()), "  ")
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()
                self.save_checkpoint(model, optimizer, opt_state, training_state)

            finally:
                training_state = self.on_training_end(training_state)
