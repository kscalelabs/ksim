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

import xax
from dpshdl.dataset import Dataset

from ksim.env.base import Environment


@dataclass(kw_only=True)
class RLConfig(xax.Config):
    clip_param: float = xax.field(value=0.2, help="Clipping parameter for PPO.")
    gamma: float = xax.field(value=0.998, help="Discount factor for PPO.")
    lam: float = xax.field(value=0.95, help="Lambda parameter for PPO.")
    value_loss_coef: float = xax.field(value=1.0, help="Value loss coefficient for PPO.")
    entropy_coef: float = xax.field(value=0.0, help="Entropy coefficient for PPO.")
    learning_rate: float = xax.field(value=1e-3, help="Learning rate for PPO.")
    max_grad_norm: float = xax.field(value=1.0, help="Maximum gradient norm for PPO.")
    use_clipped_value_loss: bool = xax.field(value=True, help="Whether to use clipped value loss for PPO.")
    schedule: str = xax.field(value="fixed", help="Schedule for PPO.")
    desired_kl: float = xax.field(value=0.01, help="Desired KL divergence for PPO.")


Config = TypeVar("Config", bound=RLConfig)


class RLTask(xax.Task[Config], Generic[Config], ABC):
    @abstractmethod
    def get_environment(self) -> Environment: ...

    @classmethod
    def run_environment(cls, *cfgs: xax.RawConfigType, num_steps: int, use_cli: bool | list[str] = True) -> None:
        xax.configure_logging()
        cfg = cls.get_config(*cfgs, use_cli=use_cli)
        task_obj = cls(cfg)
        env = task_obj.get_environment()
        env.test_run(num_steps)

    def get_dataset(self, phase: Literal["train", "valid"]) -> Dataset:
        raise NotImplementedError("Reinforcement learning tasks do not require datasets.")

    def run(self) -> None:
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
