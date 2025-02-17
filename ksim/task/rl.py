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
from typing import Generic, Literal, TypeVar

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

from ksim.env.brax import ActionModel, ActionModelType, KScaleEnv, KScaleEnvConfig, cast_action_type

logger = logging.getLogger(__name__)

T = TypeVar("T")


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


Config = TypeVar("Config", bound=RLConfig)


class RLTask(xax.Task[Config], Generic[Config], ABC):
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
        actions: ActionModelType | ActionModel | None = None,
    ) -> None:
        if actions is None:
            actions = cast_action_type(self.config.default_action_model)
        env = self.get_environment()
        render_name = self.get_render_name(state)
        render_dir = self.exp_dir / "renders" / render_name
        logger.log(xax.LOG_STATUS, "Rendering to %s", render_dir)
        env.unroll_trajectory_and_render(
            self.max_steps_per_trajectory,
            render_dir,
            seed=self.config.random_seed,
            actions=actions,
        )

    @eqx.filter_jit
    def get_init_carry(self) -> T: ...

    @eqx.filter_jit
    def get_actor_output(
        self,
        model: PyTree,
        sys: System,
        state: BraxState,
        rng: PRNGKeyArray,
        carry: T,
    ) -> tuple[jnp.ndarray, T]:
        """Runs the model on the given inputs.

        Args:
            model: The current model.
            sys: The system to run the model on.
            state: The current state of the environment.
            rng: The current RNG key.
            carry: The carry for the model.

        Returns:
            A tuple of the output and the new carry.
        """
        raise NotImplementedError("`get_output` must be implemented by the subclass")

    def log_state(self, env: KScaleEnv) -> None:
        super().log_state()

        self.logger.log_file("env_state.yaml", OmegaConf.to_yaml(env.get_state()))

    def log_trajectory(self, env: KScaleEnv, trajectory: BraxState) -> None:
        logger.debug("Logging trajectory plots")
        for plot_key, img in env.generate_trajectory_plots(trajectory):
            self.logger.log_image(plot_key, img, namespace="traj")

        logger.debug("Logging trajectory video")
        frames, fps = env.render_trajectory_video(trajectory)
        self.logger.log_video("trajectory", frames, fps=fps, namespace="video")

    def get_reward_stats(self, trajectory: BraxState, env: KScaleEnv) -> dict[str, jnp.ndarray]:
        reward_stats: dict[str, jnp.ndarray] = {}

        # Gets the reward statistics.
        reward = jnp.where(trajectory.done[..., None], jnp.nan, trajectory.reward)
        for i, key in enumerate(env.rewards.keys()):
            reward_values = reward[..., i : i + 1].astype(jnp.float32)
            reward_stats[f"{key}/mean"] = jnp.nanmean(reward_values)
            reward_stats[f"{key}/std"] = jnp.nanstd(reward_values)

        return reward_stats

    def get_termination_stats(self, trajectory: BraxState, env: KScaleEnv) -> dict[str, jnp.ndarray]:
        termination_stats: dict[str, jnp.ndarray] = {}

        # Gets the termination statistics.
        termination = trajectory.done.max(axis=-2).astype(jnp.float32)
        termination = termination.reshape(-1, termination.shape[-1])
        max_ids = termination.argmax(axis=-1)
        for i, key in enumerate(env.terminations.keys()):
            termination_stats[key] = (max_ids == i).astype(jnp.float32).mean()

        return termination_stats

    def log_trajectory_stats(self, env: KScaleEnv, trajectory: BraxState) -> None:
        for key, value in self.get_reward_stats(trajectory, env).items():
            self.logger.log_scalar(key, value, namespace="reward")
        for key, value in self.get_termination_stats(trajectory, env).items():
            self.logger.log_scalar(key, value, namespace="termination")

    @abstractmethod
    def model_update(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectory: BraxState,
    ) -> tuple[PyTree, optax.OptState]: ...

    @eqx.filter_jit
    def _single_unroll(self, rng: PRNGKeyArray, env: KScaleEnv, model: PyTree) -> BraxState:
        return env.unroll_trajectory(
            num_steps=self.max_trajectory_steps,
            rng=rng,
            init_carry=self.get_init_carry(),
            model=functools.partial(self.get_actor_output, model=model),
        )

    @eqx.filter_jit
    def _vmapped_unroll(self, rng: PRNGKeyArray, env: KScaleEnv, model: PyTree, num_envs: int) -> list[BraxState]:
        rngs = jax.random.split(rng, num_envs)
        return jax.vmap(self._single_unroll, in_axes=(0, None, None))(rngs, env, model)

    def train_loop(
        self,
        model: PyTree,
        env: KScaleEnv,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        state: xax.State,
    ) -> None:
        # Gets the training RNGs.
        rng = self.prng_key()
        rng, train_rng, val_rng = jax.random.split(rng, 3)

        while not self.is_training_over(state):
            if self.valid_step_timer.is_valid_step(state):
                logger.debug("Starting validation step")
                val_rng, step_val_rng = jax.random.split(val_rng)
                trajectory = self._single_unroll(step_val_rng, env, model)

                # Perform logging.
                logger.debug("Finished unrolling validation trajectory, logging trajectory")
                with self.step_context("write_logs"):
                    state.phase = "valid"
                    logger.debug("Logging state timers")
                    self.log_state_timers(state)
                    logger.debug("Logging trajectory")
                    self.log_trajectory(env, trajectory)
                    logger.debug("Logging trajectory stats")
                    self.log_trajectory_stats(env, trajectory)
                    logger.debug("Writing state")
                    self.logger.write(state)
                    state.num_valid_samples += 1

            logger.debug("Starting on_step_start")
            with self.step_context("on_step_start"):
                state = self.on_step_start(state)

            # Unrolls a trajectory.
            logger.debug("Unrolling training trajectory")
            train_rng, step_rng = jax.random.split(train_rng)
            trajectories = self._vmapped_unroll(step_rng, env, model, self.config.num_envs)

            # Updates the model on the collected trajectories.
            # logger.debug("Finished unrolling training trajectory, running model update")
            # with self.step_context("update_state"):
            #     model, opt_state = self.model_update(model, optimizer, opt_state, trajectories)

            # Logs the trajectory statistics.
            logger.debug("Finished model update, logging trajectory stats")
            with self.step_context("write_logs"):
                state.phase = "train"
                logger.debug("Logging state timers")
                self.log_state_timers(state)
                logger.debug("Logging trajectory stats")
                self.log_trajectory_stats(env, trajectories)
                logger.debug("Writing state")
                self.logger.write(state)
                state.num_steps += 1

            logger.debug("Starting on_step_end")
            with self.step_context("on_step_end"):
                state = self.on_step_end(state)

            if self.should_checkpoint(state):
                logger.debug("Saving checkpoint")
                self.save_checkpoint(model, optimizer, opt_state, state)

    def run_training(self) -> None:
        """Runs the main PPO training loop."""
        with self:
            key = self.prng_key()

            self.set_loggers()

            env = self.get_environment()

            if xax.is_master():
                Thread(target=self.log_state, daemon=True, args=(env,)).start()

            key, model_key = jax.random.split(key)
            model, optimizer, opt_state, state = self.load_initial_state(model_key)
            state = self.on_training_start(state)

            def on_exit() -> None:
                self.save_checkpoint(model, optimizer, opt_state, state)

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

            try:
                self.train_loop(
                    model=model,
                    env=env,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    state=state,
                )

            except xax.TrainingFinishedError:
                if xax.is_master():
                    xax.show_info(
                        f"Finished training after {state.num_steps} steps, {state.num_samples} samples",
                        important=True,
                    )
                self.save_checkpoint(model, optimizer, opt_state, state)

            except (KeyboardInterrupt, bdb.BdbQuit):
                if xax.is_master():
                    xax.show_info("Interrupted training", important=True)

            except BaseException:
                exception_tb = textwrap.indent(xax.highlight_exception_message(traceback.format_exc()), "  ")
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()
                self.save_checkpoint(model, optimizer, opt_state, state)

            finally:
                state = self.on_training_end(state)

    def run(self) -> None:
        match self.config.action:
            case "train":
                self.run_training()

            case "env":
                self.run_environment()

            case _:
                raise ValueError(f"Invalid action: {self.config.action}. Should be one of `train` or `env`.")
