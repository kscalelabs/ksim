"""Defines a standard task interface for training reinforcement learning agents.

TODO: need to add SPMD and sharding support for super fast training :)
"""

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
from pathlib import Path
from threading import Thread
from typing import Collection, Generic, Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import optax
import xax
from dpshdl.dataset import Dataset
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx
from omegaconf import MISSING
from xax.utils.transformation import scan_model

from ksim.actuators import Actuators
from ksim.commands import Command
from ksim.env.data import PhysicsModel, Transition
from ksim.env.engine import PhysicsEngine, get_physics_engine
from ksim.env.unroll import (
    UnrollNaNDetector,
    unroll_trajectory,
)
from ksim.observation import Observation
from ksim.resets import Reset
from ksim.rewards import Reward
from ksim.task.types import RLDataset, RolloutTimeStats
from ksim.terminations import Termination
from ksim.utils.named_access import get_joint_metadata

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class RLConfig(xax.Config):
    run_environment: bool = xax.field(
        value=False,
        help="Instead of dropping into the training loop, run the environment loop.",
    )
    num_learning_epochs: int = xax.field(
        value=MISSING,
        help="Number of learning epochs per dataset.",
    )
    num_env_states_per_minibatch: int = xax.field(
        value=MISSING,
        help="The number of environment states to include in each minibatch.",
    )
    num_minibatches: int = xax.field(
        value=MISSING,
        help="The number of minibatches to use for training.",
    )
    num_envs: int = xax.field(
        value=MISSING,
        help="The number of training environments to run in parallel.",
    )
    eval_rollout_length: int = xax.field(
        value=MISSING,
        help="The number of ctrl steps to rollout the model for evaluation.",
    )
    pretrained: str | None = xax.field(
        value=None,
        help="The path to a saved run to load from.",
    )
    checkpoint_num: int | None = xax.field(
        value=None,
        help="The checkpoint number to load. Otherwise the latest checkpoint is loaded.",
    )
    reshuffle_rollout: bool = xax.field(
        value=True,
        help="Whether to reshuffle the rollout dataset.",
    )
    compile_unroll: bool = xax.field(
        value=True,
        help="Whether to compile the entire unroll fn.",
    )
    render_camera: int | str = xax.field(
        value=-1,
        help="The camera to render from.",
    )
    render_height: int = xax.field(
        value=640,
        help="The height of the rendered images.",
    )
    render_width: int = xax.field(
        value=480,
        help="The width of the rendered images.",
    )
    render_dir: str = xax.field(
        value="render",
        help="The directory to save the rendered images.",
    )
    ctrl_dt: float = xax.field(
        value=0.02,
        help="The time step of the control loop.",
    )
    dt: float = xax.field(
        value=0.005,
        help="The time step of the physics loop.",
    )
    min_action_latency: float = xax.field(
        value=0.0,
        help="The minimum latency of the action.",
    )
    max_action_latency: float = xax.field(
        value=0.0,
        help="The maximum latency of the action.",
    )


Config = TypeVar("Config", bound=RLConfig)


class RLTask(xax.Task[Config], Generic[Config], ABC):
    """Base class for reinforcement learning tasks."""

    @abstractmethod
    def get_mujoco_model(self) -> mujoco.MjModel: ...

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> dict[str, JointMetadataOutput]:
        return get_joint_metadata(mj_model)

    def get_mjx_model(self, mj_model: mujoco.MjModel) -> mjx.Model:
        """Convert a mujoco model to an mjx model.

        Args:
            mj_model: The mujoco model to convert.

        Returns:
            The mjx model.
        """
        # TODO: We should perform some checks on the Mujoco model to ensure
        # that it is performant in MJX.
        return mjx.put_model(mj_model)

    def get_engine(
        self,
        physics_model: PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> PhysicsEngine:
        return get_physics_engine(
            physics_model=physics_model,
            resets=self.get_resets(physics_model),
            actuators=self.get_actuators(physics_model, metadata),
            dt=self.config.dt,
            ctrl_dt=self.config.ctrl_dt,
            min_action_latency=self.config.min_action_latency,
            max_action_latency=self.config.max_action_latency,
        )

    @abstractmethod
    def get_resets(self, physics_model: PhysicsModel) -> Collection[Reset]: ...

    @abstractmethod
    def get_actuators(
        self,
        physics_model: PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> Actuators: ...

    @abstractmethod
    def get_observations(self, physics_model: PhysicsModel) -> Collection[Observation]: ...

    @abstractmethod
    def get_commands(self, physics_model: PhysicsModel) -> Collection[Command]: ...

    @abstractmethod
    def get_rewards(self, physics_model: PhysicsModel) -> Collection[Reward]: ...

    @abstractmethod
    def get_terminations(self, physics_model: PhysicsModel) -> Collection[Termination]: ...

    @abstractmethod
    def get_optimizer(self) -> optax.GradientTransformation: ...

    @abstractmethod
    def get_rollout_time_stats(
        self,
        transitions: Transition,
        agent: PyTree,
    ) -> RolloutTimeStats:
        """E.g. calculating advantages and returns for a rollout.

        `transitions` should be (T, ...) shape, with no batch dimension, since
        this is calculated per-trajectory.

        Args:
            transitions: (T, ...) shape, with no batch dimension.
            agent: The agent that generated the transitions.

        Returns:
            RolloutTimeStats: The rollout time stats.
        """
        ...

    @abstractmethod
    def model_update(
        self,
        minibatch: RLDataset,
        *,
        agent: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, Array, FrozenDict[str, Array]]:
        """Perform a single minibatch update step.

        Args:
            minibatch: The minibatch to update.
            agent: The agent to update.
            optimizer: The optimizer to use.
            opt_state: The optimizer state.
            rng: The random number generator.

        Returns:
            The updated agent, optimizer state, loss, and metrics.
        """
        ...

    @property
    def dataset_size(self) -> int:
        """The total number of environment states in the current PPO dataset.

        E.g. if you have 32 minibatches, with num_envs_per_minibatch=8192, then
        you will have dataset size 32 * 8192 = 262144.
        """
        return self.config.num_env_states_per_minibatch * self.config.num_minibatches

    @property
    def num_rollout_steps_per_env(self) -> int:
        """Number of steps to unroll per environment during dataset creation.

        E.g. if you have 32 minibatches, with num_envs_per_minibatch=8192, with
        2048 envs in total, then you will rollout 128 steps each env:
        8192 * 32 / 2048 = 128
        """
        msg = "Dataset size must be divisible by number of envs"
        assert self.dataset_size % self.config.num_envs == 0, msg
        return self.dataset_size // self.config.num_envs

    @property
    def num_envs_per_minibatch(self) -> int:
        """Number of environments per minibatch.

        E.g. if you roll out 128 steps with 2048 envs in parallel with 32
        minibatches, then you will have 64 parallel env lines per minibatch.

        This helps us go from (2048, 128) to (64, 128) 32 times.
        """
        return self.config.num_envs // self.config.num_minibatches

    def get_dataset(self, phase: Literal["train", "valid"]) -> Dataset:
        """Get the dataset for the current task."""
        raise NotImplementedError("Reinforcement learning tasks do not require datasets.")

    def get_batch_size(self) -> int:
        """Get the batch size for the current task."""
        # TODO: this is a hack for xax... need to implement mini batching properly later.
        return 1

    def get_model(self, key: PRNGKeyArray) -> PyTree:
        """Get the model for the current task."""
        raise NotImplementedError(
            "Reinforcement learning tasks must implement this method. Instead, we create an agent using dummy data."
        )

    def run(self) -> None:
        """Highest level entry point for RL tasks, determines what to run."""
        if self.config.run_environment:
            self.run_environment()
        else:
            self.run_training()

    def get_checkpoint_number_and_path(self) -> tuple[int, Path]:
        """Get the checkpoint number and path from config or latest checkpoint."""
        error_msg = "Tried to load pretrained checkpoint but no path was provided."
        assert self.config.pretrained is not None, error_msg

        pretrained_path = Path(self.config.pretrained)
        if not pretrained_path.exists():
            raise ValueError(f"Checkpoint not found at {pretrained_path}")

        if self.config.checkpoint_num is not None:
            checkpoint_path = pretrained_path / "checkpoints" / f"ckpt.{self.config.checkpoint_num}.bin"
            error_msg = f"Checkpoint number {self.config.checkpoint_num} not found at {checkpoint_path}"
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

    def get_reward_stats(
        self,
        trajectory: Transition,
        reward_generators: Collection[Reward],
    ) -> dict[str, jnp.ndarray]:
        """Get reward statistics from the trajectoryl (D)."""
        reward_stats: dict[str, jnp.ndarray] = {}

        num_episodes = jnp.sum(trajectory.done).clip(min=1)
        terms = trajectory.reward_components
        for generator in reward_generators:
            statistic = terms[generator.reward_name]
            assert isinstance(statistic, Array)
            reward_stats[generator.reward_name] = jnp.sum(statistic) / num_episodes

        return reward_stats

    def get_termination_stats(
        self,
        trajectory: Transition,
        termination_generators: Collection[Termination],
    ) -> dict[str, jnp.ndarray]:
        """Get termination statistics from the trajectory (D)."""
        termination_stats: dict[str, jnp.ndarray] = {}

        terms = trajectory.termination_components
        for generator in termination_generators:
            statistic = terms[generator.termination_name]
            assert isinstance(statistic, Array)
            termination_stats[generator.termination_name] = jnp.mean(statistic)

        return termination_stats

    def log_trajectory_stats(
        self,
        transitions: Transition,
        reward_generators: Collection[Reward],
        termination_generators: Collection[Termination],
        eval: bool = False,
    ) -> None:
        """Logs the reward and termination stats for the trajectory."""
        for key, value in self.get_reward_stats(transitions, reward_generators).items():
            self.logger.log_scalar(
                key=key,
                value=value,
                namespace="reward" if not eval else "eval/reward",
            )
        for key, value in self.get_termination_stats(transitions, termination_generators).items():
            self.logger.log_scalar(
                key=key,
                value=value,
                namespace="termination" if not eval else "eval/termination",
            )

        # Logs the mean episode length.
        episode_num_per_env = jnp.sum(transitions.done, axis=0) + (1 - transitions.done[-1])
        episode_count = jnp.sum(episode_num_per_env)
        num_env_states = jnp.prod(jnp.array(transitions.done.shape))
        mean_episode_length_steps = num_env_states / episode_count * self.config.ctrl_dt
        self.logger.log_scalar(
            key="mean_episode_seconds",
            value=mean_episode_length_steps,
            namespace="stats" if not eval else "eval/stats",
        )

    def log_update_metrics(self, metrics: FrozenDict[str, Array]) -> None:
        """Logs the update stats for the current update."""
        for key, value in metrics.items():
            assert isinstance(value, Array)
            self.logger.log_scalar(
                key=key,
                value=value,
                namespace="update",
            )

    def log_has_nans(self, has_nans: UnrollNaNDetector) -> None:
        """Logs the number of nans in the current update."""
        for key, value in has_nans._asdict().items():
            self.logger.log_scalar(
                key=key,
                value=value,
                namespace="has_nans",
            )

    @xax.profile
    def get_minibatch(
        self,
        dataset: RLDataset,
        minibatch_idx: Array,
    ) -> RLDataset:
        """Selecting across E dim to go from (E, T) to (B, T)."""
        starting_idx = minibatch_idx * self.num_envs_per_minibatch
        minibatch = xax.slice_pytree(
            dataset,
            start=starting_idx,
            slice_length=self.num_envs_per_minibatch,
        )
        return minibatch

    @xax.profile
    def scannable_minibatch_step(
        self,
        training_state: tuple[PyTree, optax.OptState, PRNGKeyArray],
        minibatch_idx: Array,
        *,
        optimizer: optax.GradientTransformation,
        dataset_ET: RLDataset,
    ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], FrozenDict[str, Array]]:
        """Perform a single minibatch update step."""
        agent, opt_state, rng = training_state
        minibatch_BT = self.get_minibatch(dataset_ET, minibatch_idx)
        agent, opt_state, _, metrics = self.model_update(
            agent=agent,
            optimizer=optimizer,
            opt_state=opt_state,
            minibatch=minibatch_BT,
            rng=rng,
        )
        rng, _ = jax.random.split(rng)

        return (agent, opt_state, rng), metrics

    @xax.profile
    def scannable_train_epoch(
        self,
        training_state: tuple[PyTree, optax.OptState, PRNGKeyArray],
        _: None,
        *,
        optimizer: optax.GradientTransformation,
        dataset_ET: RLDataset,
    ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], FrozenDict[str, Array]]:
        """Train a minibatch, returns the updated agent, optimizer state, loss, and metrics."""
        agent = training_state[0]
        opt_state = training_state[1]
        rng = training_state[2]

        if self.config.reshuffle_rollout:  # if doing recurrence, can't shuffle across T
            dataset_ET = xax.reshuffle_pytree(  # TODO: confirm this actually reshuffles across T and E (for IID)
                dataset_ET,
                (self.config.num_envs, self.num_rollout_steps_per_env),
                rng,
            )
        rng, minibatch_rng = jax.random.split(rng)

        partial_fn = functools.partial(
            self.scannable_minibatch_step,
            optimizer=optimizer,
            dataset_ET=dataset_ET,
        )

        (agent, opt_state, _), metrics = scan_model(
            partial_fn,
            agent,
            (opt_state, minibatch_rng),
            jnp.arange(self.config.num_minibatches),
        )

        # note that variables, opt_state are just the final values
        # metrics is stacked over the minibatches
        return (agent, opt_state, rng), metrics

    @xax.profile
    @eqx.filter_jit  # TODO: implement filter-like jit in xax
    def rl_pass(
        self,
        agent: PyTree,
        opt_state: optax.OptState,
        reshuffle_rng: PRNGKeyArray,
        optimizer: optax.GradientTransformation,
        dataset_ET: RLDataset,
    ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], FrozenDict[str, Array]]:
        """Perform multiple epochs of RL training on the current dataset."""
        partial_fn = functools.partial(
            self.scannable_train_epoch,
            optimizer=optimizer,
            dataset_ET=dataset_ET,
        )

        (agent, opt_state, reshuffle_rng), metrics = scan_model(
            partial_fn,
            agent,
            (opt_state, reshuffle_rng),
            None,
            length=self.config.num_learning_epochs,
        )
        metrics = jax.tree.map(lambda x: jnp.mean(x), metrics)
        return (agent, opt_state, reshuffle_rng), metrics

    def rl_train_loop(
        self,
        agent: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        training_state: xax.State,
    ) -> None:
        """Runs the main RL training loop."""
        mj_model = self.get_mujoco_model()
        metadata = self.get_mujoco_model_metadata(mj_model)
        mjx_model = self.get_mjx_model(mj_model)
        engine = self.get_engine(mjx_model, metadata)
        observations = self.get_observations(mjx_model)
        commands = self.get_commands(mjx_model)
        rewards = self.get_rewards(mjx_model)
        terminations = self.get_terminations(mjx_model)

        rng = self.prng_key()
        rng, burn_in_rng, reset_rng, train_rng = jax.random.split(rng, 4)

        reset_rngs = jax.random.split(reset_rng, self.config.num_envs)
        state_E = jax.vmap(engine.reset, in_axes=(0))(reset_rngs)

        unroll_trajectories_fn = jax.vmap(
            unroll_trajectory,
            in_axes=(0, 0, None, None, None, None, None, None, None, None),
        )

        if self.config.compile_unroll:
            unroll_trajectories_fn = eqx.filter_jit(unroll_trajectories_fn)

        # Burn in stage
        burn_in_rng_E = jax.random.split(burn_in_rng, self.config.num_envs)

        # No positional arguments with vmap (https://github.com/jax-ml/jax/issues/7465)
        unroll_trajectories_fn(
            state_E,
            burn_in_rng_E,
            agent,
            engine,
            observations,
            commands,
            rewards,
            terminations,
            self.num_rollout_steps_per_env,
            False,
        )

        while not self.is_training_over(training_state):
            with self.step_context("on_step_start"):
                training_state = self.on_step_start(training_state)

            start_time = time.time()
            rollout_rng, reshuffle_rng, train_rng = jax.random.split(train_rng, 3)
            rollout_rng_E = jax.random.split(rollout_rng, self.config.num_envs)
            transitions_ET, state_E, has_nans_E, _ = unroll_trajectories_fn(
                state_E,
                rollout_rng_E,
                agent,
                engine,
                observations,
                commands,
                rewards,
                terminations,
                self.num_rollout_steps_per_env,
                False,
            )
            has_nans = jax.tree.map(jnp.any, has_nans_E)

            rollout_time = time.time() - start_time
            logger.info("Rollout time: %s seconds", rollout_time)

            start_time = time.time()
            rollout_stats_ET = self.get_rollout_time_stats(
                transitions=transitions_ET,
                agent=agent,
            )

            dataset_ET = RLDataset(
                transitions=transitions_ET,
                rollout_time_stats=rollout_stats_ET,
            )

            # running training on minibatches
            (agent, opt_state, reshuffle_rng), metrics = self.rl_pass(
                agent=agent,
                opt_state=opt_state,
                reshuffle_rng=reshuffle_rng,
                optimizer=optimizer,
                dataset_ET=dataset_ET,
            )
            metrics_mean = jax.tree.map(lambda x: jnp.mean(x), metrics)
            update_time = time.time() - start_time
            logger.info("Update time: %s seconds", update_time)

            with self.step_context("write_logs"):
                training_state.raw_phase = "train"
                training_state.num_steps += 1
                training_state.num_samples += self.dataset_size

                self.log_trajectory_stats(transitions_ET, rewards, terminations, eval=False)
                self.log_update_metrics(metrics_mean)
                self.log_has_nans(has_nans)
                self.logger.log_scalar("rollout_time", rollout_time, namespace="⏰")
                self.logger.log_scalar("update_time", update_time, namespace="⏰")
                self.logger.write(training_state)

            with self.step_context("on_step_end"):
                training_state = self.on_step_end(training_state)

            if self.should_checkpoint(training_state):
                self.save_checkpoint(
                    model=agent,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    state=training_state,
                )

                render_name = self.get_render_name(training_state)
                render_dir = self.exp_dir / self.config.render_dir / render_name
                logger.info("Rendering to %s", render_dir)

                raise NotImplementedError("Please rewrite this to use the new unroll_trajectory function")
                logger.info("Done rendering to %s", render_dir)

                with self.step_context("write_logs"):
                    self.log_trajectory_stats(transitions_ET, rewards, terminations, eval=True)

    def run_environment(self) -> None:
        """Wraps the environment loop.

        This provides an easy interface for developers to test out their
        models and environments before launching training jobs.
        """
        with self:
            rng = self.prng_key()
            self.set_loggers()

            rng, model_rng = jax.random.split(rng)
            agent, state = self.load_initial_state(model_rng, load_optimizer=False)

            mj_model = self.get_mujoco_model()
            metadata = self.get_mujoco_model_metadata(mj_model)
            engine = self.get_engine(mj_model, metadata)
            observations = self.get_observations(mj_model)
            commands = self.get_commands(mj_model)
            rewards = self.get_rewards(mj_model)
            terminations = self.get_terminations(mj_model)

            breakpoint()

            asdf

    def run_training(self) -> None:
        """Wraps the training loop and provides clean XAX integration."""
        with self:
            rng = self.prng_key()
            self.set_loggers()

            if xax.is_master():
                Thread(target=self.log_state, daemon=True).start()

            rng, model_rng = jax.random.split(rng)
            agent, optimizer, opt_state, training_state = self.load_initial_state(model_rng, load_optimizer=True)

            training_state = self.on_training_start(training_state)
            training_state.num_samples = 1  # prevents from checkpointing at start

            def on_exit() -> None:
                self.save_checkpoint(agent, optimizer, opt_state, training_state)

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

            try:
                self.rl_train_loop(
                    agent=agent,
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
                self.save_checkpoint(agent, optimizer, opt_state, training_state)

            except (KeyboardInterrupt, bdb.BdbQuit):
                if xax.is_master():
                    xax.show_info("Interrupted training", important=True)

            except BaseException:
                exception_tb = textwrap.indent(xax.highlight_exception_message(traceback.format_exc()), "  ")
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()
                self.save_checkpoint(agent, optimizer, opt_state, training_state)

            finally:
                training_state = self.on_training_end(training_state)
