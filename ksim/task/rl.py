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
from typing import Generic, Literal, TypeVar

import jax
import jax.numpy as jnp
import optax
import xax
from dpshdl.dataset import Dataset
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree
from omegaconf import MISSING

from ksim.env.base_env import BaseEnv, BaseEnvConfig, EnvState
from ksim.loggers import AverageRewardLog, EpisodeLengthLog, ModelUpdateLog
from ksim.model.base import Agent
from ksim.model.types import ModelInput
from ksim.normalization import Normalizer
from ksim.task.types import RolloutTimeLossComponents
from ksim.utils.visualization import render_and_save_trajectory

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass
class RLConfig(BaseEnvConfig, xax.Config):
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
    compile_unroll: bool = xax.field(
        value=True,
        help="Whether to compile the entire unroll fn.",
    )


Config = TypeVar("Config", bound=RLConfig)


class RLTask(xax.Task[Config], Generic[Config], ABC):
    """Base class for reinforcement learning tasks."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.log_items = [EpisodeLengthLog(), AverageRewardLog(), ModelUpdateLog()]  # TODO: fix
        super().__init__(config)

    ####################
    # Abstract methods #
    ####################

    # These should be implemented by the user.

    @abstractmethod
    def get_environment(self) -> BaseEnv: ...

    @abstractmethod
    def get_init_actor_carry(self) -> jnp.ndarray | None: ...

    @abstractmethod
    def get_obs_normalizer(self, dummy_obs: FrozenDict[str, Array]) -> Normalizer: ...

    @abstractmethod
    def get_cmd_normalizer(self, dummy_cmd: FrozenDict[str, Array]) -> Normalizer: ...

    # The following should be implemented by the algorithmic subclass.

    @abstractmethod
    def get_rollout_time_loss_components(
        self,
        agent: Agent,
        trajectory_dataset: EnvState,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
    ) -> RolloutTimeLossComponents: ...

    @abstractmethod
    def model_update(
        self,
        agent: Agent,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        env_state_batch: EnvState,
        rollout_time_loss_components: RolloutTimeLossComponents,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
        rng: PRNGKeyArray,
    ) -> tuple[Agent, optax.OptState, Array, FrozenDict[str, Array]]: ...

    ##############
    # Properties #
    ##############

    @property
    def dataset_size(self) -> int:
        """The total number of environment states in the current PPO dataset."""
        return self.config.num_env_states_per_minibatch * self.config.num_minibatches

    @property
    def num_rollout_steps_per_env(self) -> int:
        """Number of steps to unroll per environment during dataset creation."""
        msg = "Dataset size must be divisible by number of envs"
        assert self.dataset_size % self.config.num_envs == 0, msg
        return self.dataset_size // self.config.num_envs

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
            "Reinforcement learning tasks must implement this method."
            "Instead, we create an agent using dummy data."
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
        agent: Agent,
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
            logger.info("Time taken for environment setup: %s seconds", end_time - start_time)

            logger.log(logging.INFO, "Rendering to %s", render_dir)

            self.set_loggers()

            # Unroll trajectories and collect the frames for rendering
            logger.info("Unrolling trajectories")

            render_and_save_trajectory(
                env=env,
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

    def get_reward_stats(self, trajectory: EnvState, env: BaseEnv) -> dict[str, jnp.ndarray]:
        """Get reward statistics from the trajectoryl (D)."""
        reward_stats: dict[str, jnp.ndarray] = {}

        num_episodes = jnp.sum(trajectory.done).clip(min=1)  # rough approx
        terms = trajectory.reward_components
        for key, _ in env.rewards:
            statistic = terms[key]
            assert isinstance(statistic, Array)
            reward_stats[key] = jnp.sum(statistic) / num_episodes

        return reward_stats

    def get_termination_stats(self, trajectory: EnvState, env: BaseEnv) -> dict[str, jnp.ndarray]:
        """Get termination statistics from the trajectory (D)."""
        termination_stats: dict[str, jnp.ndarray] = {}

        terms = trajectory.termination_components
        for key, _ in env.terminations:
            statistic = terms[key]
            assert isinstance(statistic, Array)
            termination_stats[key] = jnp.mean(statistic)

        return termination_stats

    def log_trajectory_stats(
        self, env: BaseEnv, env_state_TEL: EnvState, eval: bool = False
    ) -> None:
        """Logs the reward and termination stats for the trajectory."""
        for key, value in self.get_reward_stats(env_state_TEL, env).items():
            self.logger.log_scalar(
                key=key,
                value=value,
                namespace="reward" if not eval else "eval/reward",
            )
        for key, value in self.get_termination_stats(env_state_TEL, env).items():
            self.logger.log_scalar(
                key=key,
                value=value,
                namespace="termination" if not eval else "eval/termination",
            )

        # Logs the mean episode length.
        episode_num_per_env = jnp.sum(env_state_TEL.done, axis=0) + (1 - env_state_TEL.done[-1])
        episode_count = jnp.sum(episode_num_per_env)
        num_env_states = jnp.prod(jnp.array(env_state_TEL.done.shape))
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

    @xax.profile
    def reshuffle_rollout(
        self,
        rollout_dataset_DL: EnvState,
        rollout_time_loss_components_DL: RolloutTimeLossComponents,
        rng: PRNGKeyArray,
    ) -> tuple[EnvState, RolloutTimeLossComponents]:
        """Reshuffle a rollout array (DL)."""
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
        reshuffled_rollout_dataset_DL = jax.tree_util.tree_map(permute_array, rollout_dataset_DL)
        reshuffled_rollout_time_loss_components_DL = jax.tree_util.tree_map(
            permute_array, rollout_time_loss_components_DL
        )

        return reshuffled_rollout_dataset_DL, reshuffled_rollout_time_loss_components_DL

    @xax.profile
    def get_minibatch(
        self,
        rollout_DL: EnvState,
        rollout_time_loss_components_DL: RolloutTimeLossComponents,
        minibatch_idx: Array,
    ) -> tuple[EnvState, RolloutTimeLossComponents]:
        """Get a minibatch from the rollout (B)."""
        starting_idx = minibatch_idx * self.config.num_env_states_per_minibatch
        rollout_BL = xax.slice_pytree(
            rollout_DL,
            start=starting_idx,
            slice_length=self.config.num_env_states_per_minibatch,
        )
        rollout_time_loss_components_BL = xax.slice_pytree(
            rollout_time_loss_components_DL,
            start=starting_idx,
            slice_length=self.config.num_env_states_per_minibatch,
        )
        return rollout_BL, rollout_time_loss_components_BL

    @xax.profile
    def scannable_minibatch_step(
        self,
        training_state: tuple[Agent, optax.OptState, PRNGKeyArray],
        minibatch_idx: Array,
        *,
        optimizer: optax.GradientTransformation,
        dataset_DL: EnvState,
        rollout_time_loss_components_DL: RolloutTimeLossComponents,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
    ) -> tuple[tuple[Agent, optax.OptState, PRNGKeyArray], FrozenDict[str, Array]]:
        """Perform a single minibatch update step."""
        agent, opt_state, rng = training_state
        minibatch_BL, rollout_time_minibatch_loss_components_BL = self.get_minibatch(
            dataset_DL, rollout_time_loss_components_DL, minibatch_idx
        )
        agent, opt_state, _, metrics = self.model_update(
            agent=agent,
            optimizer=optimizer,
            opt_state=opt_state,
            env_state_batch=minibatch_BL,
            rollout_time_loss_components=rollout_time_minibatch_loss_components_BL,
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
        dataset_DL: EnvState,
        rollout_time_loss_components_DL: RolloutTimeLossComponents,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
    ) -> tuple[tuple[Agent, optax.OptState, PRNGKeyArray], FrozenDict[str, Array]]:
        """Train a minibatch, returns the updated agent, optimizer state, loss, and metrics."""
        agent = training_state[0]
        opt_state = training_state[1]
        rng = training_state[2]

        dataset_DL, rollout_time_loss_components_DL = self.reshuffle_rollout(
            dataset_DL, rollout_time_loss_components_DL, rng
        )
        rng, minibatch_rng = jax.random.split(rng)

        partial_fn = functools.partial(
            self.scannable_minibatch_step,
            optimizer=optimizer,
            dataset_DL=dataset_DL,
            rollout_time_loss_components_DL=rollout_time_loss_components_DL,
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
    @xax.jit(static_argnames=["self", "optimizer"])
    def rl_pass(
        self,
        agent: Agent,
        opt_state: optax.OptState,
        reshuffle_rng: PRNGKeyArray,
        optimizer: optax.GradientTransformation,
        dataset_DL: EnvState,
        rollout_time_loss_components_DL: RolloutTimeLossComponents,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
    ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], FrozenDict[str, Array]]:
        """Perform multiple epochs of RL training on the current dataset."""
        partial_fn = functools.partial(
            self.scannable_train_epoch,
            optimizer=optimizer,
            dataset_DL=dataset_DL,
            rollout_time_loss_components_DL=rollout_time_loss_components_DL,
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
        env: BaseEnv,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        training_state: xax.State,
    ) -> None:
        """Runs the main RL training loop."""
        rng = self.prng_key()
        burn_in_rng, reset_rng, train_rng = jax.random.split(rng, 3)

        # Gets the initial physics data.
        physics_model_L = env.get_init_physics_model()
        reset_rngs = jax.random.split(reset_rng, self.config.num_envs)
        dummy_env_states = env.get_dummy_env_states(self.config.num_envs)

        obs_normalizer = self.get_obs_normalizer(dummy_env_states.obs)
        cmd_normalizer = self.get_cmd_normalizer(dummy_env_states.command)

        # NOTE: env_state_EL_t and physics_data_EL_t_plus_1 are the carry data
        # structures used to efficiently unroll and resume trajectory rollouts
        env_state_EL_t, physics_data_EL_t_plus_1 = jax.vmap(
            env.reset,
            in_axes=(None, 0, None, None, None),
        )(agent, reset_rngs, physics_model_L, obs_normalizer, cmd_normalizer)
        breakpoint()

        if self.config.compile_unroll:
            static_args = ["num_steps", "num_envs", "return_intermediate_data"]
            env_rollout_fn = xax.jit(static_argnames=static_args)(env.unroll_trajectories)

        #################
        # Burn in Stage #
        #################

        burn_in_env_state_TEL, _, _ = env_rollout_fn(
            agent=agent,
            rng=burn_in_rng,
            num_steps=self.num_rollout_steps_per_env,
            num_envs=self.config.num_envs,
            env_state_EL_t_minus_1=env_state_EL_t,
            physics_data_EL_t=physics_data_EL_t_plus_1,
            physics_model_L=physics_model_L,
            obs_normalizer=obs_normalizer,
            cmd_normalizer=cmd_normalizer,
            return_intermediate_data=False,
        )

        obs_normalizer = obs_normalizer.update(burn_in_env_state_TEL.obs)
        cmd_normalizer = cmd_normalizer.update(burn_in_env_state_TEL.command)

        ##################
        # Training Stage #
        ##################
        while not self.is_training_over(training_state):
            with self.step_context("on_step_start"):
                training_state = self.on_step_start(training_state)

            #############################
            # Rollout and Normalization #
            #############################

            start_time = time.time()
            reshuffle_rng, rollout_rng, train_rng = jax.random.split(train_rng, 3)

            env_state_TEL, physics_data_EL_t_plus_1, has_nans = env_rollout_fn(
                agent=agent,
                rng=rollout_rng,
                num_steps=self.num_rollout_steps_per_env,
                num_envs=self.config.num_envs,
                env_state_EL_t_minus_1=env_state_EL_t,
                physics_data_EL_t=physics_data_EL_t_plus_1,
                physics_model_L=physics_model_L,
                obs_normalizer=obs_normalizer,
                cmd_normalizer=cmd_normalizer,
                return_intermediate_data=False,
            )
            env_state_EL_t = jax.tree_util.tree_map(lambda x: x[-1], env_state_TEL)

            rollout_time = time.time() - start_time
            logger.info("Rollout time: %s seconds", rollout_time)

            env_state_DL = xax.flatten_pytree(
                env_state_TEL,
                flatten_size=self.dataset_size,
            )

            ###########################
            # Minibatch Training Loop #
            ###########################
            start_time = time.time()

            # getting loss components that are constant across minibatches
            # (e.g. advantages) and flattening them for efficiency, thus
            # the name "rollout_time" loss components
            rollout_loss_components_TEL = self.get_rollout_time_loss_components(
                agent=agent,
                trajectory_dataset=env_state_TEL,
                obs_normalizer=obs_normalizer,
                cmd_normalizer=cmd_normalizer,
            )
            rollout_loss_components_DL = xax.flatten_pytree(
                rollout_loss_components_TEL,
                flatten_size=self.dataset_size,
            )

            # running training on minibatches
            (agent, opt_state, reshuffle_rng), metrics = self.rl_pass(
                agent=agent,
                opt_state=opt_state,
                reshuffle_rng=reshuffle_rng,
                optimizer=optimizer,
                dataset_DL=env_state_DL,
                rollout_time_loss_components_DL=rollout_loss_components_DL,
                obs_normalizer=obs_normalizer,
                cmd_normalizer=cmd_normalizer,
            )
            metrics_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x), metrics)
            update_time = time.time() - start_time
            logger.info("Update time: %s seconds", update_time)

            # this will allow for online normalization, just know that we will
            # be normalizing the already normalized observations and commands.
            obs_normalizer = obs_normalizer.update(env_state_TEL.obs)
            cmd_normalizer = cmd_normalizer.update(env_state_TEL.command)

            with self.step_context("write_logs"):
                training_state.raw_phase = "train"
                training_state.num_steps += 1
                training_state.num_samples += self.dataset_size

                self.log_trajectory_stats(env, env_state_TEL, eval=False)
                self.log_update_metrics(metrics_mean)
                self.logger.log_scalar("has_nans", has_nans, namespace="stats")
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

                eval_env_state_T1L = render_and_save_trajectory(
                    env=env,
                    agent=agent,
                    rng=rng,
                    output_dir=render_dir,
                    num_steps=self.config.eval_rollout_length,
                    width=self.config.render_width,
                    height=self.config.render_height,
                )
                logger.info("Done rendering to %s", render_dir)

                with self.step_context("write_logs"):
                    self.log_trajectory_stats(env, eval_env_state_T1L, eval=True)

    def run_training(self) -> None:
        """Wraps the training loop and provides clean XAX integration."""
        with self:
            key = self.prng_key()
            self.set_loggers()
            env = self.get_environment()

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
                self.save_checkpoint(agent, optimizer, opt_state, training_state)

            except (KeyboardInterrupt, bdb.BdbQuit):
                if xax.is_master():
                    xax.show_info("Interrupted training", important=True)

            except BaseException:
                exception_tb = textwrap.indent(
                    xax.highlight_exception_message(traceback.format_exc()), "  "
                )
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()
                self.save_checkpoint(agent, optimizer, opt_state, training_state)

            finally:
                training_state = self.on_training_end(training_state)
