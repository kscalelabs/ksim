"""Defines a standard task interface for training reinforcement learning agents."""

import bdb
import functools
import logging
import signal
import sys
import textwrap
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Thread
from typing import Generic, Literal, NamedTuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from brax.base import System
from brax.envs.base import State as BraxState
from dpshdl.dataset import Dataset
from jaxtyping import PRNGKeyArray, PyTree
from omegaconf import MISSING, OmegaConf

from ksim.env.base_env import BaseEnv
from ksim.env.kscale_env import (
    KScaleActionModel,
    KScaleActionModelType,
    KScaleEnv,
    KScaleEnvConfig,
    cast_action_type,
)
from ksim.model.formulations import ActorCriticModel

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

    @abstractmethod
    def get_environment(self) -> KScaleEnv: ...

    def get_dataset(self, phase: Literal["train", "valid"]) -> Dataset:
        raise NotImplementedError("Reinforcement learning tasks do not require datasets.")

    @property
    def max_steps_per_trajectory(self) -> int:
        return round(self.config.max_episode_length / self.config.ctrl_dt)

    def get_render_name(self, state: xax.State | None = None) -> str:
        time_string = time.strftime("%Y%m%d_%H%M%S")
        if state is None:
            return f"render_{time_string}"
        return f"render_{state.num_steps}_{time_string}"

    def run_environment(
        self,
        state: xax.State | None = None,
        actions: KScaleActionModelType | KScaleActionModel | None = None,
    ) -> None:
        if actions is None:
            actions = cast_action_type(self.config.default_action_model)
        rng = self.prng_key()
        env = self.get_environment()
        render_name = self.get_render_name(state)
        render_dir = self.exp_dir / "renders" / render_name
        # logger.log(xax.LOG_STATUS, "Rendering to %s", render_dir)
        env.unroll_trajectories_and_render(
            rng=rng,
            num_steps=self.max_steps_per_trajectory,
            render_dir=render_dir,
            actions=actions,
        )

    @abstractmethod
    def get_init_actor_carry(self) -> jnp.ndarray | None: ...

    @abstractmethod
    def get_model_obs_from_state(self, state: BraxState) -> PyTree: ...

    # @abstractmethod
    # def get_actor_output(
    #     self,
    #     model: PyTree,
    #     sys: System,
    #     state: BraxState,
    #     rng: PRNGKeyArray,
    #     carry: jnp.ndarray | None,
    # ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    #     """Get the actor output.

    #     Args:
    #         model: The linen-based neural network model.
    #         params: The variable dictionary of the model {"params": {...}, "other_vars": {...}}
    #         sys: The system (see Brax)
    #         state: The state (see Brax)
    #         rng: The random key.
    #         carry: The carry state for recurrent models (not used here)

    #     Returns:
    #         The actor output and the carry state (keeping for consistency)
    #     """
    #     raise NotImplementedError("`get_output` must be implemented by the subclass")

    def log_state(self, env: KScaleEnv) -> None:
        super().log_state()

        self.logger.log_file("env_state.yaml", OmegaConf.to_yaml(env.get_state()))

    def log_trajectory(self, env: KScaleEnv, trajectory: BraxState) -> None:
        for plot_key, img in env.generate_trajectory_plots(trajectory):
            self.logger.log_image(plot_key, img, namespace="traj")

        frames, fps = env.render_trajectory_video(trajectory)
        self.logger.log_video("trajectory", frames, fps=fps, namespace="video")

    def get_reward_stats(self, trajectory: BraxState, env: KScaleEnv) -> dict[str, jnp.ndarray]:
        reward_stats: dict[str, jnp.ndarray] = {}

        # Gets the reward statistics.
        reward = jnp.where(trajectory.done[..., None], jnp.nan, trajectory.info["all_rewards"])
        for i, (key, _) in enumerate(env.rewards):
            reward_values = reward[..., i : i + 1].astype(jnp.float32)
            reward_stats[f"{key}/mean"] = jnp.nanmean(reward_values)
            reward_stats[f"{key}/std"] = jnp.nanstd(reward_values)

        return reward_stats

    def get_init_params(self, key: PRNGKeyArray, pretrained: str | None = None) -> PyTree:
        """Get the initial parameters.
        Args:
            key: The random key.

        Returns:
            The initial parameters.
        """
        env = self.get_environment()
        state = env.reset(key)

        if pretrained is not None:
            # TODO: implement pretrained model loading.
            raise NotImplementedError("Pretrained models are not yet implemented.")

        # Provide a dummy observation (of appropriate shape) for initialization.
        return self.get_model(key).init(key, self.get_model_obs_from_state(state))

    def get_termination_stats(
        self, trajectory: BraxState, env: KScaleEnv
    ) -> dict[str, jnp.ndarray]:
        termination_stats: dict[str, jnp.ndarray] = {}

        # Gets the termination statistics.
        termination = trajectory.info["all_dones"].max(axis=-2).astype(jnp.float32)
        termination = termination.reshape(-1, termination.shape[-1])
        max_ids = termination.argmax(axis=-1)
        for i, (key, _) in enumerate(env.terminations):
            termination_stats[key] = (max_ids == i).astype(jnp.float32).mean()

        return termination_stats

    def log_trajectory_stats(self, env: KScaleEnv, trajectory: BraxState) -> None:
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

    @abstractmethod
    def model_update(
        self,
        model: ActorCriticModel,
        params: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        batch: NamedTuple,
    ) -> tuple[PyTree, optax.OptState]: ...

    @abstractmethod
    def get_trajectory_batch(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env: BaseEnv,
        rng: PRNGKeyArray,
    ) -> NamedTuple:
        """Rollout the model for a given number of steps.
        Args:
            model: The model (see `apply_actor`)
            params: The variable dictionary of the model {"params": {...}, "other_vars": {...}}
            env: The environment (see `unroll_trajectories`)
            rng: The random key.

        Returns:
            A NamedTuple containing the trajectories.
        """
        ...

    def train_loop(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env: BaseEnv,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        training_state: xax.State,
    ) -> None:
        rng = self.prng_key()
        rng, train_rng, val_rng = jax.random.split(rng, 3)

        while not self.is_training_over(training_state):
            if self.valid_step_timer.is_valid_step(training_state):
                val_rng, step_val_rng = jax.random.split(val_rng)
                trajectory = self.get_trajectory_batch(model, params, env, step_val_rng)
                assert hasattr(trajectory, "done"), "Trajectory must contain a `done` field."
                # Calculate episode length by counting steps until done
                episode_lengths = jnp.sum(~trajectory.done) / jnp.sum(trajectory.done)  # type: ignore
                print(f"Average episode length: {episode_lengths}")

            #     # Perform logging.
            #     with self.step_context("write_logs"):
            #         state.raw_phase = "valid"
            #         # self.log_state_timers(state)
            #         # self.log_trajectory(env, trajectory)
            #         # self.log_trajectory_stats(env, trajectory)
            #         self.logger.write(state)
            #         state.num_valid_samples += 1

            with self.step_context("on_step_start"):
                training_state = self.on_step_start(training_state)

            # Unrolls a trajectory.
            train_rng, step_rng = jax.random.split(train_rng)
            trajectories = self.get_trajectory_batch(model, params, env, step_rng)

            # Updates the model on the collected trajectories.
            with self.step_context("update_state"):
                params, opt_state = self.model_update(
                    model, params, optimizer, opt_state, trajectories
                )

            # # Logs the trajectory statistics.
            with self.step_context("write_logs"):
                #     state.phase = "train"
                #     self.log_state_timers(state)
                #     self.log_trajectory_stats(env, trajectories)
                #     self.logger.write(state)
                training_state.num_steps += 1

            with self.step_context("on_step_end"):
                training_state = self.on_step_end(training_state)

            if self.should_checkpoint(training_state):
                self.save_checkpoint(
                    model=params, optimizer=optimizer, opt_state=opt_state, state=training_state
                )  # Update XAX to be Flax supportive...

    def run_training(self) -> None:
        """Runs the main PPO training loop."""
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
                    xax.show_info(
                        f"Finished training after {training_state.num_steps} steps, {training_state.num_samples} samples",
                        important=True,
                    )
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

    def get_batch_size(self) -> int:
        """Get the batch size for the current task."""
        # TODO: this is a hack... need to implement mini batching properly later.
        return 1

    def run(self) -> None:
        match self.config.action:
            case "train":
                self.run_training()

            case "env":
                self.run_environment()

            case _:
                raise ValueError(
                    f"Invalid action: {self.config.action}. Should be one of `train` or `env`."
                )
