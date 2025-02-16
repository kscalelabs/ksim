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
    def get_actor_output(
        self,
        model: PyTree,
        sys: System,
        state: BraxState,
        rng: PRNGKeyArray,
        carry: T | None,
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
                val_rng, step_val_rng = jax.random.split(val_rng)
                trajectory = env.unroll_trajectory(
                    num_steps=self.max_trajectory_steps,
                    rng=step_val_rng,
                    model=functools.partial(self.get_actor_output, model=model),
                )

                breakpoint()

                # Perform logging.
                with self.step_context("write_logs"):
                    state.phase = "valid"
                    self.log_step(model, valid_batch, output, loss, state)
                    state.num_valid_samples += 1

            with self.step_context("on_step_start"):
                state = self.on_step_start(state)

            train_rng, step_rng = jax.random.split(train_rng)
            step_rngs = jax.random.split(step_rng, self.config.num_envs)
            trajectories = jax.vmap(
                functools.partial(
                    env.unroll_trajectory,
                    num_steps=self.max_trajectory_steps,
                    model=functools.partial(self.get_actor_output, model=model),
                )
            )(step_rngs)

            breakpoint()

            # Perform logging.
            with self.step_context("write_logs"):
                state.phase = "train"
                self.log_step(model, train_batch, output, loss, state)
                state.num_steps += 1
                state.num_samples += self.get_size_of_batch(train_batch) or 0

            with self.step_context("on_step_end"):
                state = self.on_step_end(state)

            if self.should_checkpoint(state):
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
