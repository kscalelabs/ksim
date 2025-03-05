"""Defines a standard task interface for training reinforcement learning agents."""

import bdb
import logging
import signal
import sys
import textwrap
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Generic, Literal, TypeVar

import jax
import jax.numpy as jnp
import optax
import xax
from dpshdl.dataset import Dataset
from flax import linen as nn
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree
from omegaconf import MISSING

from ksim.builders.loggers import (
    AverageRewardLog,
    EpisodeLengthLog,
    LoggingData,
    ModelUpdateLog,
)
from ksim.env.base_env import BaseEnv, BaseEnvConfig, EnvState
from ksim.model.formulations import ActorCriticAgent
from ksim.task.types import RolloutTimeLossComponents
from ksim.utils.jit import legit_jit
from ksim.utils.pytree import slice_pytree
from ksim.utils.visualization import render_and_save_trajectory

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class RLConfig(BaseEnvConfig, xax.Config):
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
    num_learning_epochs: int = xax.field(
        value=1,
        help="Number of learning epochs per PPO update.",
    )
    minibatch_size: int = xax.field(
        value=MISSING,
        help="The size of each minibatch.",
    )
    pretrained: str | None = xax.field(
        value=None,
        help="The path to a saved run to load from.",
    )
    checkpoint_num: int | None = xax.field(
        value=None,
        help="The checkpoint number to load. Otherwise the latest checkpoint is loaded.",
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
    def get_init_actor_carry(self) -> jnp.ndarray | None: ...

    @abstractmethod
    def get_rollout_time_loss_components(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
        trajectory_dataset: EnvState,
        burn_in: bool = False,
    ) -> RolloutTimeLossComponents: ...

    @abstractmethod
    def model_update(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
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
        msg = (
            f"Dataset size of {self.dataset_size} must"
            f" be divisible by minibatch size of {self.config.minibatch_size}."
        )
        assert self.dataset_size % self.config.minibatch_size == 0, msg
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

            case _:
                raise ValueError(
                    f"Invalid action: {self.config.action}. Should be one of `train` or `env`."
                )

    #########################
    # Logging and Rendering #
    #########################

    def get_checkpoint_number_and_path(self) -> tuple[int, Path]:
        """Get the checkpoint number and path from config or latest checkpoint."""
        error_msg = "Tried to load pretrained checkpoint but no path was provided."
        assert self.config.pretrained is not None, error_msg

        pretrained_path = Path(self.config.pretrained)
        if not pretrained_path.exists():
            raise ValueError(f"Checkpoint not found at {pretrained_path}")

        if self.config.checkpoint_num is not None:
            checkpoint_path = (
                pretrained_path / "checkpoints" / f"ckpt.{self.config.checkpoint_num}.bin"
            )
            error_msg = (
                f"Checkpoint number {self.config.checkpoint_num} not found at {checkpoint_path}"
            )
            assert checkpoint_path.exists(), error_msg
            return self.config.checkpoint_num, checkpoint_path

        # Get the latest checkpoint in the folder
        ckpt_files = sorted(pretrained_path.glob("checkpoints/ckpt.*.bin"))
        if not ckpt_files:
            raise ValueError(f"No checkpoints found in {pretrained_path}/checkpoints/")
        checkpoint_path = ckpt_files[-1]
        ckpt_num = int(checkpoint_path.stem.split(".")[1])
        return ckpt_num, checkpoint_path

    def get_render_name(self, state: xax.State | None = None) -> str:
        """Get a unique name for the render directory based on state and checkpoint info."""
        time_string = time.strftime("%Y%m%d_%H%M%S")
        prefix = "render"

        # If training, label render with step count
        if state is not None:
            return f"{prefix}_{state.num_steps}_{time_string}"

        # If resuming, use the checkpoint number
        if self.config.pretrained is not None:
            ckpt_num, _ = self.get_checkpoint_number_and_path()
            return f"{prefix}_pretrained_{ckpt_num}_{time_string}"

        return f"{prefix}_no_state_{time_string}"

    def run_environment(
        self,
        model: ActorCriticAgent,
        state: xax.State | None = None,
    ) -> None:
        """Run the environment with rendering and logging."""
        with self:
            start_time = time.time()
            rng = self.prng_key()
            env = self.get_environment()
            render_name = self.get_render_name(state)
            render_dir = self.exp_dir / self.config.render_dir / render_name

            end_time = time.time()
            print(f"Time taken for environment setup: {end_time - start_time} seconds")

            logger.log(logging.INFO, "Rendering to %s", render_dir)

            self.set_loggers()

            key, _ = jax.random.split(rng)

            start_time = time.time()
            variables = self.get_init_variables(
                key, pretrained=self.config.pretrained, checkpoint_num=self.config.checkpoint_num
            )
            end_time = time.time()
            print(f"Time taken for parameter initialization: {end_time - start_time} seconds")

            # Unroll trajectories and collect the frames for rendering
            logger.info("Unrolling trajectories")

            render_and_save_trajectory(
                env=env,
                model=model,
                variables=variables,
                rng=rng,
                output_dir=render_dir,
                num_steps=self.config.num_steps_per_trajectory,
                width=self.config.render_width,
                height=self.config.render_height,
            )

    def log_state(self) -> None:
        super().log_state()

        # self.logger.log_file("env_state.yaml", OmegaConf.to_yaml(env.get_state()))

    def get_reward_stats(self, trajectory: EnvState, env: BaseEnv) -> dict[str, jnp.ndarray]:
        reward_stats: dict[str, jnp.ndarray] = {}

        terms = trajectory.reward_components
        for key, _ in env.rewards:
            statistic = terms[key]
            assert isinstance(statistic, Array)
            reward_stats[key] = jnp.mean(statistic)

        return reward_stats

    def get_termination_stats(self, trajectory: EnvState, env: BaseEnv) -> dict[str, jnp.ndarray]:
        termination_stats: dict[str, jnp.ndarray] = {}

        terms = trajectory.termination_components
        for key, _ in env.terminations:
            statistic = terms[key]
            assert isinstance(statistic, Array)
            termination_stats[key] = jnp.mean(statistic)

        return termination_stats

    def log_trajectory_stats(self, env: BaseEnv, trajectory: EnvState) -> None:
        for key, value in self.get_reward_stats(trajectory, env).items():
            self.logger.log_scalar(key, value, namespace="reward")
        for key, value in self.get_termination_stats(trajectory, env).items():
            self.logger.log_scalar(key, value, namespace="termination")

        # Logs the mean episode length.
        mean_episode_length_steps = (~trajectory.done).sum(axis=-1).astype(jnp.float32).mean()
        mean_episode_length_seconds = mean_episode_length_steps * self.config.ctrl_dt
        self.logger.log_scalar(
            "mean_episode_length", mean_episode_length_seconds, namespace="stats"
        )

    ########################
    # Training and Running #
    ########################

    def get_init_variables(
        self, key: PRNGKeyArray, pretrained: str | None = None, checkpoint_num: int | None = None
    ) -> PyTree:
        """Get the initial parameters as a PyTree: assumes flax-compatible model."""
        env = self.get_environment()
        state = env.get_dummy_env_state(key)

        if pretrained is not None:
            _, checkpoint_path = self.get_checkpoint_number_and_path()
            logger.info("Loading pretrained checkpoint from %s", checkpoint_path)
            return self.load_checkpoint(checkpoint_path, part="model")

        model_key, init_key = jax.random.split(key, 2)
        model = self.get_model(model_key)
        assert isinstance(model, nn.Module), "Model must be an Flax linen module."
        return model.init(init_key, state.obs, state.command)

    @abstractmethod
    def update_input_normalization_stats(
        self,
        variables: PyTree,
        trajectories_dataset: EnvState,
        rollout_time_loss_components: RolloutTimeLossComponents,
        initial_step: bool,
    ) -> PyTree: ...

    @legit_jit(static_argnames=["self", "model"])
    def apply_actor(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
    ) -> Array:
        """Apply the actor model to inputs."""
        res = model.apply(variables, obs=obs, cmd=cmd, method="actor")
        assert isinstance(res, Array)
        return res

    def get_trajectory_dataset(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
        env: BaseEnv,
        rng: PRNGKeyArray,
        burn_in: bool = False,
    ) -> tuple[EnvState, RolloutTimeLossComponents]:
        """Rollout a batch of trajectory data.

        To avoid confusion, batch comprises 1 or more unrolled trajectory states stacked
        along the first axis, and minibatches are sampled from this batch.
        """
        # TODO: implement logic to handle randomize model initialization when creating batch
        rollout, _ = env.unroll_trajectories(
            model=model,
            variables=variables,
            rng=rng,
            num_steps=self.config.num_steps_per_trajectory,
            num_envs=self.config.num_envs,
        )

        rollout_time_loss_components = self.get_rollout_time_loss_components(
            model,
            variables,
            rollout,
            burn_in=burn_in,
        )

        @legit_jit()
        def flatten_rollout_array(x: Array) -> Array:
            """Flatten a rollout array."""
            reshaped = jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:]))
            assert reshaped.shape[0] == self.config.num_steps_per_trajectory * self.config.num_envs
            return reshaped

        # flattening (num_steps, num_envs, ...) -> (num_steps * num_envs, ...) in leaves
        flattened_rollout = jax.tree_util.tree_map(flatten_rollout_array, rollout)
        flattened_rollout_time_loss_components = jax.tree_util.tree_map(
            flatten_rollout_array, rollout_time_loss_components
        )

        return flattened_rollout, flattened_rollout_time_loss_components

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
        permutation = jax.random.choice(rng, jnp.arange(batch_size), (batch_size,))

        # Apply permutation to rollout dataset
        def permute_array(x: Array) -> Array:
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

    def rl_train_loop(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
        env: BaseEnv,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        training_state: xax.State,
    ) -> None:
        """Runs the main RL training loop."""
        rng = self.prng_key()
        rng, burn_in_rng, train_rng = jax.random.split(rng, 3)

        # Burn in trajectory to get normalization statistics
        burn_in_trajectories_dataset, burn_in_rollout_time_loss_components = (
            self.get_trajectory_dataset(
                model,
                variables,
                env,
                burn_in_rng,
                burn_in=True,
            )
        )
        variables = self.update_input_normalization_stats(
            variables=variables,
            trajectories_dataset=burn_in_trajectories_dataset,
            rollout_time_loss_components=burn_in_rollout_time_loss_components,
            initial_step=True,
        )
        while not self.is_training_over(training_state):
            with self.step_context("on_step_start"):
                training_state = self.on_step_start(training_state)

            # Unrolls a trajectory.
            start_time = time.time()
            reshuffle_rng, rollout_rng = jax.random.split(train_rng)
            trajectories_dataset, rollout_time_loss_components = self.get_trajectory_dataset(
                model, variables, env, rollout_rng
            )
            rollout_time = time.time() - start_time
            self.log_trajectory_stats(env, trajectories_dataset)

            # running training on minibatches
            start_time = time.time()
            for epoch_idx in range(self.config.num_learning_epochs):
                trajectories_dataset, rollout_time_loss_components = self.reshuffle_rollout(
                    trajectories_dataset, rollout_time_loss_components, reshuffle_rng
                )
                reshuffle_rng, _ = jax.random.split(reshuffle_rng)
                for minibatch_idx in range(self.num_minibatches):
                    minibatch_idx_array = jnp.array(
                        minibatch_idx
                    )  # TODO: scanning will do this anyways
                    minibatch, rollout_time_minibatch_loss_components = self.get_minibatch(
                        trajectories_dataset, rollout_time_loss_components, minibatch_idx_array
                    )
                    with self.step_context("update_state"):
                        params, opt_state, loss_val, metrics = self.model_update(
                            model,
                            variables,
                            optimizer,
                            opt_state,
                            minibatch,
                            rollout_time_minibatch_loss_components,
                        )
                        variables["params"] = params
                        print(f"loss: {loss_val}")
                        print(f"metrics: {metrics}")

                        # log metrics from the model update
                        metric_logging_data = LoggingData(
                            trajectory=trajectories_dataset,
                            update_metrics=metrics,
                            gradients=None,
                            loss=float(loss_val),
                            training_state=training_state,
                        )

                        # print(f"{epoch_idx}, {minibatch_idx}: {metrics}")

                        with self.step_context("write_logs"):
                            training_state.raw_phase = "train"
                            for log_item in self.log_items:
                                log_item(self.logger, metric_logging_data)

                            self.logger.write(training_state)
                            training_state.num_steps += 1

            model_update_time = time.time() - start_time

            # updating normalization statistics for the next rollout
            # NOTE: for the first step, the normalization stats are not updated
            variables = self.update_input_normalization_stats(
                variables=variables,
                trajectories_dataset=trajectories_dataset,
                rollout_time_loss_components=rollout_time_loss_components,
                initial_step=False,
            )

            # Log the time taken for the model update.
            with self.step_context("write_logs"):
                self.logger.log_scalar("rollout_time", rollout_time, namespace="⏰")
                self.logger.log_scalar("model_update_time", model_update_time, namespace="⏰")
                self.logger.write(training_state)

            with self.step_context("on_step_end"):
                training_state = self.on_step_end(training_state)

            if self.should_checkpoint(training_state):
                self.save_checkpoint(
                    model=variables, optimizer=optimizer, opt_state=opt_state, state=training_state
                )  # Update XAX to be Flax supportive...

                render_name = self.get_render_name(training_state)
                render_dir = self.exp_dir / self.config.render_dir / render_name
                logger.info("Rendering to %s", render_dir)

                render_and_save_trajectory(
                    env=env,
                    model=model,
                    variables=variables,
                    rng=rng,
                    output_dir=render_dir,
                    num_steps=self.config.num_steps_per_trajectory,
                    width=self.config.render_width,
                    height=self.config.render_height,
                )

                logger.info("Done rendering to %s", render_dir)

    def run_training(self) -> None:
        """Wraps the training loop and provides clean XAX integration."""
        with self:
            key = self.prng_key()
            self.set_loggers()
            env = self.get_environment()

            if xax.is_master():
                Thread(target=self.log_state, daemon=True).start()

            key, model_key = jax.random.split(key)
            model, optimizer, opt_state, training_state = self.load_initial_state(model_key)

            training_state = self.on_training_start(training_state)

            def on_exit() -> None:
                self.save_checkpoint(model, optimizer, opt_state, training_state)

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

            variables = self.get_init_variables(key)
            opt_state = optimizer.init(variables["params"])

            try:
                self.rl_train_loop(
                    model=model,
                    variables=variables,
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
                exception_tb = textwrap.indent(
                    xax.highlight_exception_message(traceback.format_exc()), "  "
                )
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()
                self.save_checkpoint(model, optimizer, opt_state, training_state)

            finally:
                training_state = self.on_training_end(training_state)
