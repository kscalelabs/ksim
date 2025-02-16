"""Defines a standard task interface for training reinforcement learning agents."""

import contextlib
import datetime
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

import jax
import xax
from dpshdl.dataset import Dataset
from omegaconf import MISSING

from ksim.env.brax import ActionModel, KScaleEnv, KScaleEnvConfig, cast_action_type

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
    default_action_model: str = xax.field(
        value="zero",
        help="The default action model to use if `actions` is not specified.",
    )


Config = TypeVar("Config", bound=RLConfig)


class RLTask(xax.Task[Config], Generic[Config], ABC):
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
        actions: Literal["random", "zero", "midpoint"] | ActionModel | None = None,
    ) -> None:
        env = self.get_environment()
        render_name = self.get_render_name(state)
        render_dir = self.exp_dir / "renders" / render_name
        logger.log(xax.LOG_STATUS, "Rendering to %s", render_dir)
        if actions is None:
            actions = cast_action_type(self.config.default_action_model)
        env.unroll_trajectory_and_render(
            self.max_steps_per_trajectory,
            render_dir,
            seed=self.config.random_seed,
            actions=actions,
        )

    def run_training(self) -> None:
        """Runs the main PPO training loop."""
        with contextlib.ExitStack() as ctx:
            self.set_loggers()

            if xax.is_master():
                Thread(target=self.log_state, daemon=True).start()

            with self.step_context("model_to_device"):
                mod = self.get_trainable_module(self)
                self.configure_model_(mod)
                mod = self.get_wrapped_model(mod)

            with self.step_context("create_optimizers"):
                opt = self.build_optimizer(mod)

            with self.step_context("load_checkpoint"):
                state = self.load_ckpt_(model=mod, optimizer=opt, strict=self.config.init_state_strict)

            def on_exit() -> None:
                self.save_ckpt(
                    state=state,
                    model=mod,
                    optimizer=opt,
                )

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1)

            try:
                if (profile := self.get_profile()) is not None:
                    ctx.enter_context(profile)

                while True:
                    if self.is_training_over(state):
                        raise xax.TrainingFinishedError

                    if self.is_valid_step(state):
                        with self.step_context("valid_step"):
                            self.val_step(mod, valid_batch_iterator.next(state), state)

                    with self.step_context("on_step_start"):
                        self.on_step_start(state)

                    with self.step_context("train_step"):
                        loss_dict = self.train_step(mod, opt, train_batch_iterator.next(state), state)

                    if self.should_save_ckpt(state):
                        with self.step_context("save_checkpoint"):
                            self.save_ckpt(
                                state=state,
                                model=mod,
                                optimizer=opt,
                            )

                    if profile is not None:
                        profile.step()

                    with self.step_context("on_step_end"):
                        self.on_step_end(state, loss_dict)

            except xax.TrainingFinishedError:
                with self.step_context("save_checkpoint"):
                    self.save_ckpt(
                        state=state,
                        model=mod,
                        optimizer=opt,
                    )

                if xax.is_master():
                    elapsed_time = xax.format_timedelta(datetime.timedelta(seconds=time.time() - state.start_time_s))
                    xax.show_info(
                        f"Finished training after {state.num_steps} steps, {state.num_samples} samples, {elapsed_time}",
                        important=True,
                    )

            except BaseException:
                exception_tb = textwrap.indent(xax.highlight_exception_message(traceback.format_exc()), "  ")
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()

            finally:
                self.on_training_end(state)

    def run(self) -> None:
        match self.config.action:
            case "train":
                self.run_training()

            case "env":
                self.run_environment()

            case _:
                raise ValueError(f"Invalid action: {self.config.action}. Should be one of `train` or `env`.")
