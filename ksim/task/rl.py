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
import optax
import xax
from dpshdl.dataset import Dataset
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree
from kscale.web.gen.api import JointMetadataOutput
from omegaconf import MISSING

from ksim.commands import Command
from ksim.env.base_engine import PhysicsEngine
from ksim.env.data import PhysicsModel, Transition
from ksim.env.unroll import (
    UnrollNaNDetector,
    get_initial_commands,
    get_observation,
    unroll_trajectory,
)
from ksim.loggers import AverageRewardLog, EpisodeLengthLog, ModelUpdateLog
from ksim.model.base import Agent
from ksim.normalization import Normalizer
from ksim.observation import Observation
from ksim.rewards import Reward
from ksim.task.types import RLDataset, RolloutTimeStats
from ksim.terminations import Termination
from ksim.utils.visualization import render_and_save_trajectory

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class RLConfig(xax.Config):
    action: str = xax.field(
        value="train",
        help="The action to take; should be either `train` or `env`.",
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

    def __init__(self, config: Config) -> None:
        self.config = config
        self.log_items = [EpisodeLengthLog(), AverageRewardLog(), ModelUpdateLog()]  # TODO: fix
        super().__init__(config)

    @abstractmethod
    def get_model_and_metadata(self) -> tuple[Agent, dict[str, JointMetadataOutput]]: ...

    @abstractmethod
    def get_engine(self, physics_model: PhysicsModel, metadata: dict[str, JointMetadataOutput]) -> PhysicsEngine: ...

    @abstractmethod
    def get_obs_generators(self, physics_model: PhysicsModel) -> Collection[Observation]: ...

    @abstractmethod
    def get_command_generators(self) -> Collection[Command]: ...

    @abstractmethod
    def get_reward_generators(self, physics_model: PhysicsModel) -> Collection[Reward]: ...

    @abstractmethod
    def get_termination_generators(self, physics_model: PhysicsModel) -> Collection[Termination]: ...

    @abstractmethod
    def get_obs_normalizer(self, dummy_obs: FrozenDict[str, Array]) -> Normalizer: ...

    @abstractmethod
    def get_cmd_normalizer(self, dummy_cmd: FrozenDict[str, Array]) -> Normalizer: ...

    @abstractmethod
    def get_optimizer(self) -> optax.GradientTransformation: ...

    # The following should be implemented by the algorithmic subclass.

    @abstractmethod
    def get_rollout_time_stats(
        self,
        transitions: Transition,
        *,
        agent: Agent,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
    ) -> RolloutTimeStats:
        """E.g. calculating advantages and returns for a rollout.

        NOTE: transitions should be (T, ...) shape. No batch dimension.
        """
        ...

    @abstractmethod
    def model_update(
        self,
        minibatch: RLDataset,
        *,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
        rng: PRNGKeyArray,
    ) -> tuple[Agent, optax.OptState, Array, FrozenDict[str, Array]]: ...

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
        2048 envs in total, then you will rollout 128 steps each env.
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

    def get_model(self, key: PRNGKeyArray) -> PyTree:
        """Get the model for the current task."""
        raise NotImplementedError(
            "Reinforcement learning tasks must implement this method.Instead, we create an agent using dummy data."
        )

    def run(self) -> None:
        """Highest level entry point for RL tasks, determines what to run."""
        match self.config.action:
            case "train":
                self.run_training()

            case "env":
                agent, _, _, _ = self.load_initial_state(self.prng_key())
                self.run_environment(agent)

            case _:
                raise ValueError(f"Invalid action: {self.config.action}. Should be one of `train` or `env`.")

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

    def run_environment(
        self,
        agent: Agent,
        state: xax.State | None = None,
    ) -> None:
        """Run the environment with rendering and logging."""
        with self:
            start_time = time.time()
            rng = self.prng_key()
            physics_model, metadata = self.get_model_and_metadata()
            engine = self.get_engine(physics_model, metadata)
            render_name = self.get_render_name(state)
            render_dir = self.exp_dir / self.config.render_dir / render_name

            end_time = time.time()
            logger.info("Time taken for environment setup: %s seconds", end_time - start_time)

            logger.log(logging.INFO, "Rendering to %s", render_dir)

            self.set_loggers()

            # Unroll trajectories and collect the frames for rendering
            logger.info("Unrolling trajectories")

            render_and_save_trajectory(
                engine=engine,
                agent=agent,
                rng=rng,
                output_dir=render_dir,
                num_steps=self.num_rollout_steps_per_env,
                width=self.config.render_width,
                height=self.config.render_height,
            )

    def log_state(self) -> None:
        super().log_state()

        # self.logger.log_file("env_state.yaml", OmegaConf.to_yaml(env.get_state()))

    def get_reward_stats(
        self,
        trajectory: Transition,
        reward_generators: Collection[Reward],
    ) -> dict[str, jnp.ndarray]:
        """Get reward statistics from the trajectoryl (D)."""
        reward_stats: dict[str, jnp.ndarray] = {}

        num_episodes = jnp.sum(trajectory.done).clip(min=1)  # rough approx
        terms = trajectory.reward_components
        for generator in reward_generators:
            statistic = terms[generator.reward_name]
            assert isinstance(statistic, Array)
            reward_stats[generator.reward_name] = jnp.sum(statistic) / num_episodes

        return reward_stats

    def get_termination_stats(
        self, trajectory: Transition, termination_generators: Collection[Termination]
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
        """Selecting accross E dim to go from (E, T) to (B, T)."""
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
        training_state: tuple[Agent, optax.OptState, PRNGKeyArray],
        minibatch_idx: Array,
        *,
        optimizer: optax.GradientTransformation,
        dataset_ET: RLDataset,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
    ) -> tuple[tuple[Agent, optax.OptState, PRNGKeyArray], FrozenDict[str, Array]]:
        """Perform a single minibatch update step."""
        agent, opt_state, rng = training_state
        minibatch_BT = self.get_minibatch(dataset_ET, minibatch_idx)
        agent, opt_state, _, metrics = self.model_update(
            agent=agent,
            optimizer=optimizer,
            opt_state=opt_state,
            minibatch=minibatch_BT,
            obs_normalizer=obs_normalizer,
            cmd_normalizer=cmd_normalizer,
            rng=rng,
        )
        rng, _ = jax.random.split(rng)

        return (agent, opt_state, rng), metrics

    @xax.profile
    def scannable_train_epoch(
        self,
        training_state: tuple[Agent, optax.OptState, PRNGKeyArray],
        _: None,
        *,
        optimizer: optax.GradientTransformation,
        dataset_ET: RLDataset,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
    ) -> tuple[tuple[Agent, optax.OptState, PRNGKeyArray], FrozenDict[str, Array]]:
        """Train a minibatch, returns the updated agent, optimizer state, loss, and metrics."""
        agent = training_state[0]
        opt_state = training_state[1]
        rng = training_state[2]

        if self.config.reshuffle_rollout:  # if doing recurrence, can't shuffle accross T
            dataset_ET = xax.reshuffle_pytree(  # TODO: confirm this actually reshuffles accross T and E (for IID)
                dataset_ET,
                (self.config.num_envs, self.num_rollout_steps_per_env),
                rng,
            )
        rng, minibatch_rng = jax.random.split(rng)

        partial_fn = functools.partial(
            self.scannable_minibatch_step,
            optimizer=optimizer,
            dataset_ET=dataset_ET,
            obs_normalizer=obs_normalizer,
            cmd_normalizer=cmd_normalizer,
        )

        (agent, opt_state, _), metrics = jax.lax.scan(
            partial_fn,
            (agent, opt_state, minibatch_rng),
            jnp.arange(self.config.num_minibatches),
        )

        # note that variables, opt_state are just the final values
        # metrics is stacked over the minibatches
        return (agent, opt_state, rng), metrics

    @xax.profile
    @eqx.filter_jit  # TODO: implement filter-like jit in xax
    def rl_pass(
        self,
        agent: Agent,
        opt_state: optax.OptState,
        reshuffle_rng: PRNGKeyArray,
        optimizer: optax.GradientTransformation,
        dataset_ET: RLDataset,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
    ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], FrozenDict[str, Array]]:
        """Perform multiple epochs of RL training on the current dataset."""
        partial_fn = functools.partial(
            self.scannable_train_epoch,
            optimizer=optimizer,
            dataset_ET=dataset_ET,
            obs_normalizer=obs_normalizer,
            cmd_normalizer=cmd_normalizer,
        )

        (agent, opt_state, reshuffle_rng), metrics = jax.lax.scan(
            partial_fn,
            (agent, opt_state, reshuffle_rng),
            None,
            length=self.config.num_learning_epochs,
        )
        # metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
        return (agent, opt_state, reshuffle_rng), metrics

    def rl_train_loop(
        self,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        training_state: xax.State,
    ) -> None:
        """Runs the main RL training loop."""
        physics_model, metadata = self.get_model_and_metadata()
        engine = self.get_engine(physics_model, metadata)
        obs_generators = self.get_obs_generators(physics_model)  # these should be given necessary data for building
        command_generators = self.get_command_generators()
        reward_generators = self.get_reward_generators(physics_model)
        termination_generators = self.get_termination_generators(physics_model)

        rng = self.prng_key()
        burn_in_rng, reset_rng, train_rng = jax.random.split(rng, 3)

        reset_rngs = jax.random.split(reset_rng, self.config.num_envs)
        state_E = jax.vmap(
            engine.reset,
            in_axes=(0),
        )(reset_rngs)

        # the normalizers need dummy data to infer shapes
        dummy_obs = get_observation(state_E, burn_in_rng, obs_generators=obs_generators)
        dummy_cmd = get_initial_commands(burn_in_rng, command_generators=command_generators)
        obs_normalizer = self.get_obs_normalizer(dummy_obs)
        cmd_normalizer = self.get_cmd_normalizer(dummy_cmd)

        unroll_trajectories_fn = jax.vmap(unroll_trajectory, in_axes=(0, 0))  # only 1 pos param
        if self.config.compile_unroll:
            unroll_trajectories_fn = eqx.filter_jit(unroll_trajectories_fn)

        # Burn in stage
        burn_transitions_ET, _, _, _ = unroll_trajectories_fn(
            state_E,
            rng,
            agent=agent,
            obs_normalizer=obs_normalizer,
            cmd_normalizer=cmd_normalizer,
            engine=engine,
            obs_generators=obs_generators,
            command_generators=command_generators,
            reward_generators=reward_generators,
            termination_generators=termination_generators,
            num_steps=self.num_rollout_steps_per_env,
            return_intermediate_physics_data=False,
        )

        obs_normalizer = obs_normalizer.update(burn_transitions_ET.obs)
        cmd_normalizer = cmd_normalizer.update(burn_transitions_ET.command)

        while not self.is_training_over(training_state):
            with self.step_context("on_step_start"):
                training_state = self.on_step_start(training_state)

            start_time = time.time()
            reshuffle_rng, rollout_rng, train_rng = jax.random.split(train_rng, 3)

            transitions_ET, state_E, has_nans, _ = unroll_trajectories_fn(
                state_E,
                agent=agent,
                obs_normalizer=obs_normalizer,
                cmd_normalizer=cmd_normalizer,
                rng=rollout_rng,
                engine=engine,
                obs_generators=obs_generators,
                command_generators=command_generators,
                reward_generators=reward_generators,
                termination_generators=termination_generators,
                num_steps=self.num_rollout_steps_per_env,
                return_intermediate_physics_data=False,
            )

            rollout_time = time.time() - start_time
            logger.info("Rollout time: %s seconds", rollout_time)

            start_time = time.time()

            # getting loss components that are constant across minibatches
            # (e.g. advantages) and flattening them for efficiency, thus
            # the name "rollout_time" loss components
            rollout_stats_ET = jax.vmap(
                self.get_rollout_time_stats,
                in_axes=(0),
            )(
                transitions_ET,
                agent=agent,
                obs_normalizer=obs_normalizer,
                cmd_normalizer=cmd_normalizer,
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
                obs_normalizer=obs_normalizer,
                cmd_normalizer=cmd_normalizer,
            )
            metrics_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
            update_time = time.time() - start_time
            logger.info("Update time: %s seconds", update_time)

            # this will allow for online normalization, must be updated after
            # the update step.
            obs_normalizer = obs_normalizer.update(transitions_ET.obs)
            cmd_normalizer = cmd_normalizer.update(transitions_ET.command)

            with self.step_context("write_logs"):
                training_state.raw_phase = "train"
                training_state.num_steps += 1
                training_state.num_samples += self.dataset_size

                self.log_trajectory_stats(transitions_ET, reward_generators, termination_generators, eval=False)
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

                render_and_save_trajectory(
                    agent=agent,
                    engine=engine,
                    rng=rng,
                    output_dir=render_dir,
                    num_steps=self.config.eval_rollout_length,
                    width=self.config.render_width,
                    height=self.config.render_height,
                )
                logger.info("Done rendering to %s", render_dir)

                with self.step_context("write_logs"):
                    self.log_trajectory_stats(transitions_ET, reward_generators, termination_generators, eval=True)

    def run_training(self) -> None:
        """Wraps the training loop and provides clean XAX integration."""
        with self:
            key = self.prng_key()
            self.set_loggers()

            if xax.is_master():
                Thread(target=self.log_state, daemon=True).start()

            key, model_key = jax.random.split(key)
            agent, optimizer, opt_state, training_state = self.load_initial_state(model_key)

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
