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
from threading import Thread
from typing import Generic, Literal, TypeVar

import equinox as eqx
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
from ksim.env.base_env import BaseEnv, EnvState
from ksim.env.mjx.mjx_env import (
    KScaleActionModelType,
    KScaleEnvConfig,
    cast_action_type,
)
from ksim.model.formulations import ActionModel, ActorCriticModel

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
    num_envs: int = xax.field(
        value=MISSING,
        help="The number of training environments to run in parallel.",
    )
    default_action_model: str = xax.field(
        value="zero",
        help="The default action model to use if `actions` is not specified.",
    )
    max_trajectory_seconds: float = xax.field(
        value=MISSING,
        help="The maximum trajectory length, in seconds.",
    )
    observation_size: int = xax.field(
        value=MISSING,
        help="The size of the observation space.",
    )
    action_size: int = xax.field(
        value=MISSING,
        help="The size of the action space.",
    )


Config = TypeVar("Config", bound=RLConfig)


class RLTask(xax.Task[Config], Generic[Config], ABC):
    """Base class for reinforcement learning tasks.

    Attributes:
        config: The RL configuration.
        max_trajectory_steps: The maximum number of steps in a trajectory.
    """

    max_trajectory_steps: int

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        self.max_trajectory_steps = round(self.config.max_trajectory_seconds / self.config.ctrl_dt)
        self.curr_logging_data = LoggingData()
        self.log_items = [EpisodeLengthLog(), AverageRewardLog(), ModelUpdateLog()]

    ####################
    # Abstract methods #
    ####################

    @abstractmethod
    def get_environment(self) -> BaseEnv: ...

    @abstractmethod
    def viz_environment(self) -> None: ...

    @abstractmethod
    def get_trajectory_batch(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env: BaseEnv,
        rng: PRNGKeyArray,
    ) -> EnvState: ...

    @abstractmethod
    def get_init_actor_carry(self) -> jnp.ndarray | None: ...

    @abstractmethod
    def model_update(
        self,
        model: ActorCriticModel,
        params: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        env_state_batch: EnvState,
    ) -> tuple[PyTree, optax.OptState, Array, dict[str, Array]]: ...

    ##############
    # Properties #
    ##############

    @property
    def max_steps_per_trajectory(self) -> int:
        """Max episode length (in seconds) divided by control time step."""
        return round(self.config.max_episode_length / self.config.ctrl_dt)

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
                self.run_environment()

            case "viz":
                self.viz_environment()

            case _:
                raise ValueError(
                    f"Invalid action: {self.config.action}. Should be one of `train` or `env`."
                )

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
        state: xax.State | None = None,
        actions: KScaleActionModelType | ActionModel | None = None,
    ) -> None:
        """Run the environment with rendering and logging."""
        with self:
            if actions is None:
                actions = cast_action_type(self.config.default_action_model)
            rng = self.prng_key()
            env = self.get_environment()
            render_name = self.get_render_name(state)
            render_dir = self.exp_dir / "renders" / render_name
            logger.log(logging.INFO, "Rendering to %s", render_dir)

            self.set_loggers()

            # Unroll trajectories and collect the frames for rendering
            frames, trajectory = env.unroll_trajectories_and_render(
                rng=rng,
                num_steps=10,  # self.max_steps_per_trajectory,
                render_dir=render_dir,
                actions=actions,
            )

            # Use the log items for additional metrics
            self.curr_logging_data = LoggingData(
                trajectory=trajectory,
                update_metrics={},
                gradients=None,
                loss=0.0,
                training_state="valid",
            )

            for log_item in self.log_items:
                if not isinstance(log_item, ModelUpdateLog):  # Skip model update logs
                    log_item(self.logger, self.curr_logging_data)

            # Log the rendered frames
            self.logger.log_images("trajectory", frames, namespace="video")

            logger.log(logging.INFO, "Finished running environment and logging metrics")

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
        state = env.reset(key)

        if pretrained is not None:
            # TODO: implement pretrained model loading.
            raise NotImplementedError("Pretrained models are not yet implemented.")

        model_key, init_key = jax.random.split(key, 2)
        model = self.get_model(model_key)
        assert isinstance(model, nn.Module), "Model must be an Flax linen module."
        return model.init(init_key, state.obs, state.commands)

    @eqx.filter_jit
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
            train_rng, step_rng = jax.random.split(train_rng)
            trajectories = self.get_trajectory_batch(model, params, env, step_rng)
            end_time = time.time()
            print(f"Time taken for trajectory batch: {end_time - start_time} seconds")

            # Updates the model on the collected trajectories.
            start_time = time.time()
            with self.step_context("update_state"):
                params, opt_state, loss_val, metrics = self.model_update(
                    model, params, optimizer, opt_state, trajectories
                )
            end_time = time.time()
            print(f"Time taken for model update: {end_time - start_time} seconds")

            start_time = time.time()
            self.curr_logging_data = LoggingData(
                trajectory=trajectories,
                update_metrics=metrics,
                gradients=None,
                loss=float(loss_val),
                training_state=training_state,
            )

            # Logs the trajectory statistics.
            with self.step_context("write_logs"):
                training_state.raw_phase = "train"
                for log_item in self.log_items:
                    log_item(self.logger, self.curr_logging_data)
                self.logger.write(training_state)
                training_state.num_steps += 1
            end_time = time.time()
            print(f"Time taken for logging: {end_time - start_time} seconds")

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
                exception_tb = textwrap.indent(
                    xax.highlight_exception_message(traceback.format_exc()), "  "
                )
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()
                self.save_checkpoint(model, optimizer, opt_state, training_state)

            finally:
                training_state = self.on_training_end(training_state)
