"""Defines a standard task interface for training reinforcement learning agents.

TODO: need to add SPMD and sharding support for super fast training :)
"""

import bdb
import functools
import itertools
import logging
import signal
import sys
import textwrap
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Thread
from typing import Collection, Generic, TypeVar

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import mujoco_viewer
import numpy as np
import optax
import tqdm
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
from ksim.env.data import PhysicsData, PhysicsModel, PhysicsState, Transition
from ksim.env.engine import (
    EngineConstants,
    EngineVariables,
    PhysicsEngine,
    engine_type_from_physics_model,
    get_physics_engine,
)
from ksim.observation import Observation
from ksim.resets import Reset
from ksim.rewards import Reward
from ksim.task.types import RLDataset
from ksim.terminations import Termination
from ksim.utils.named_access import get_joint_metadata

logger = logging.getLogger(__name__)


def get_observation(
    physics_state: PhysicsState,
    rng: PRNGKeyArray,
    *,
    obs_generators: Collection[Observation],
) -> FrozenDict[str, Array]:
    """Get the observation from the physics state."""
    observations = {}
    for observation in obs_generators:
        rng, obs_rng = jax.random.split(rng)
        observation_value = observation(physics_state.data, obs_rng)
        observations[observation.observation_name] = observation_value
    return FrozenDict(observations)


def get_rewards(
    physics_state: PhysicsState,
    command: FrozenDict[str, Array],
    action: Array,
    next_physics_state: PhysicsState,  # TODO - rewards only process data
    next_state_terminates: Array,
    *,
    reward_generators: Collection[Reward],
) -> FrozenDict[str, Array]:
    """Get the rewards from the physics state."""
    rewards = {}
    for reward_generator in reward_generators:
        reward_val = (
            reward_generator(
                prev_action=physics_state.most_recent_action,
                physics_state=physics_state.data,
                command=command,
                action=action,
                next_physics_state=next_physics_state.data,
                next_state_terminates=next_state_terminates,
            )
            * reward_generator.scale
        )
        name = reward_generator.reward_name
        chex.assert_shape(reward_val, (), custom_message=f"Reward {name} must be a scalar")
        rewards[name] = reward_val
    return FrozenDict(rewards)


def post_accumulate_rewards(
    reward_components: FrozenDict[str, Array],
    done: Array,
    *,
    reward_generators: Collection[Reward],
) -> FrozenDict[str, Array]:
    """Post-accumulate rewards."""
    post_accumulated_reward_components = dict(reward_components)
    for reward_generator in reward_generators:
        original_reward = reward_components[reward_generator.reward_name]
        assert isinstance(original_reward, Array)
        reward_val = reward_generator.post_accumulate(original_reward, done)
        post_accumulated_reward_components[reward_generator.reward_name] = reward_val

    return FrozenDict(post_accumulated_reward_components)


def get_terminations(
    physics_state: PhysicsState,
    *,
    termination_generators: Collection[Termination],
) -> FrozenDict[str, Array]:
    """Get the terminations from the physics state."""
    terminations = {}
    for termination in termination_generators:
        termination_val = termination(physics_state.data)
        chex.assert_type(termination_val, bool)
        name = termination.termination_name
        terminations[name] = termination_val
    return FrozenDict(terminations)


def get_commands(
    prev_commands: FrozenDict[str, Array],
    physics_state: PhysicsState,
    rng: PRNGKeyArray,
    *,
    command_generators: Collection[Command],
) -> FrozenDict[str, Array]:
    """Get the commands from the physics state."""
    commands = {}
    for command_generator in command_generators:
        rng, cmd_rng = jax.random.split(rng)
        command_name = command_generator.command_name
        prev_command = prev_commands[command_name]
        assert isinstance(prev_command, Array)
        command_val = command_generator(prev_command, physics_state.data.time, cmd_rng)
        commands[command_name] = command_val
    return FrozenDict(commands)


def get_initial_commands(
    rng: PRNGKeyArray,
    *,
    command_generators: Collection[Command],
) -> FrozenDict[str, Array]:
    """Get the initial commands from the physics state."""
    commands = {}
    for command_generator in command_generators:
        rng, cmd_rng = jax.random.split(rng)
        command_name = command_generator.command_name
        command_val = command_generator.initial_command(cmd_rng)
        commands[command_name] = command_val
    return FrozenDict(commands)


def render_data_to_frames(
    data: PhysicsData,
    default_mj_model: mujoco.MjModel,
    camera: int | str | mujoco.MjvCamera = -1,
    height: int = 240,
    width: int = 320,
) -> list[np.ndarray]:
    """Render the data to a sequence of Numpy arrays."""
    for leaf in jax.tree.leaves(data):
        if isinstance(leaf, Array):
            num_steps = leaf.shape[0]
            break
    else:
        raise ValueError("No array found in data")

    mjx_data_list = [jax.tree.map(lambda x: x[i], data) for i in range(num_steps)]
    scene_option = mujoco.MjvOption()

    renderer = mujoco.Renderer(default_mj_model, height=height, width=width)
    frames = []
    for mjx_data in mjx_data_list:
        renderer.update_scene(mjx_data, camera=camera, scene_option=scene_option)
        frames.append(renderer.render())

    return frames


@jax.tree_util.register_dataclass
@dataclass
class StepInput:
    prev_command: FrozenDict[str, Array]
    physics_state: PhysicsState
    rng: PRNGKeyArray


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
        value=480,
        help="The height of the rendered images.",
    )
    render_width: int = xax.field(
        value=640,
        help="The width of the rendered images.",
    )
    render_track_body_id: int | None = xax.field(
        value=None,
        help="If set, the render camera will track the body with this ID.",
    )
    render_distance: float = xax.field(
        value=5.0,
        help="The distance of the camera from the target.",
    )
    render_azimuth: float = xax.field(
        value=90.0,
        help="The azimuth of the render camera.",
    )
    render_elevation: float = xax.field(
        value=-30.0,
        help="The elevation of the render camera.",
    )
    render_lookat: list[float] = xax.field(
        value=[0.0, 0.0, 0.5],
        help="The lookat point of the render camera.",
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
            engine_type=engine_type_from_physics_model(physics_model),
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
    def get_observations(self, physics_model: PhysicsModel) -> Collection[Observation]:
        """Returns the observation generators for the current task.

        Args:
            physics_model: The physics model to get the observations for.

        Returns:
            A collection of observation generators.
        """

    @abstractmethod
    def get_commands(self, physics_model: PhysicsModel) -> Collection[Command]:
        """Returns the command generators for the current task.

        Args:
            physics_model: The physics model to get the commands for.

        Returns:
            A collection of command generators.
        """

    @abstractmethod
    def get_rewards(self, physics_model: PhysicsModel) -> Collection[Reward]:
        """Returns the reward generators for the current task.

        Args:
            physics_model: The physics model to get the rewards for.

        Returns:
            A collection of reward generators.
        """

    @abstractmethod
    def get_terminations(self, physics_model: PhysicsModel) -> Collection[Termination]:
        """Returns the termination generators for the current task.

        Args:
            physics_model: The physics model to get the terminations for.

        Returns:
            A collection of termination generators.
        """

    @abstractmethod
    def get_initial_carry(self) -> PyTree:
        """Returns the initial carry for the model.

        Returns:
            An arbitrary PyTree, representing any carry parameters that the
            model needs.
        """

    @abstractmethod
    def sample_action(
        self,
        model: PyTree,
        carry: PyTree,
        physics_model: PhysicsModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, PyTree]:
        """Takes in the current model, the physics model, and a random key, and returns an action.

        Args:
            model: The current model.
            physics_model: The physics model.
            observations: The current observations.
            commands: The current commands.
            carry: The model carry from the previous step.
            rng: The random key.

        Returns:
            The action to take.
        """

    def step_engine(
        self,
        model: PyTree,
        engine: PhysicsEngine,
        engine_constants: EngineConstants,
        engine_variables: EngineVariables,
        rng: PRNGKeyArray,
    ) -> tuple[Transition, EngineVariables]:
        """Runs a single step of the physics engine.

        Args:
            model: The model, with parameters to be updated.
            carry: The carry from the previous step.
            engine: The physics engine.
            engine_constants: The constants for the engine.
            engine_variables: The variables for the engine.
            rng: The random key.

        Returns:
            A tuple containing the transition and the next engine variables.
        """
        obs_rng, cmd_rng, act_rng, reset_rng, physics_rng = jax.random.split(rng, 5)

        # Gets the observations from the physics state.
        observations = get_observation(
            physics_state=engine_variables.physics_state,
            rng=obs_rng,
            obs_generators=engine_constants.obs_generators,
        )

        # Gets the commmands from the previous commands and the physics state.
        commands = get_commands(
            prev_commands=engine_variables.commands,
            physics_state=engine_variables.physics_state,
            rng=cmd_rng,
            command_generators=engine_constants.command_generators,
        )

        # Samples an action from the model.
        action, next_carry = self.sample_action(
            model=model,
            carry=engine_variables.carry,
            physics_model=engine_constants.physics_model,
            observations=observations,
            commands=commands,
            rng=act_rng,
        )

        # Steps the physics engine.
        next_physics_state: PhysicsState = engine.step(
            action=action,
            physics_model=engine_constants.physics_model,
            physics_state=engine_variables.physics_state,
            rng=physics_rng,
        )

        # Gets termination components and a single termination boolean.
        terminations = get_terminations(
            physics_state=engine_variables.physics_state,
            termination_generators=engine_constants.termination_generators,
        )
        terminated = jax.tree.reduce(jnp.logical_or, list(terminations.values()))

        # Gets reward components and a single reward.
        rewards = get_rewards(
            physics_state=engine_variables.physics_state,
            command=commands,
            action=action,
            next_physics_state=next_physics_state,
            next_state_terminates=terminated,
            reward_generators=engine_constants.reward_generators,
        )
        reward = jax.tree.reduce(jnp.add, list(rewards.values()))

        # Conditionally reset on termination.
        next_physics_state = jax.lax.cond(
            terminated,
            lambda: engine.reset(engine_constants.physics_model, reset_rng),
            lambda: next_physics_state,
        )
        next_carry = jax.lax.cond(
            terminated,
            lambda: engine_constants.initial_carry,
            lambda: next_carry,
        )
        commands = jax.lax.cond(
            terminated,
            lambda: engine_constants.initial_command,
            lambda: commands,
        )

        # Combines all the relevant data into a single object.
        transition = Transition(
            obs=observations,
            command=commands,
            action=action,
            reward=reward,
            done=terminated,
            timestep=next_physics_state.data.time,
            termination_components=terminations,
            reward_components=rewards,
        )

        # Gets the variables for the next step.
        next_variables = EngineVariables(
            carry=next_carry,
            commands=commands,
            physics_state=next_physics_state,
        )

        return transition, next_variables

    def get_dataset(self, phase: xax.Phase) -> Dataset:
        raise NotImplementedError("RL tasks do not require datasets, since trajectory histories are stored in-memory.")

    def run(self) -> None:
        """Highest level entry point for RL tasks, determines what to run."""
        if self.config.run_environment:
            self.run_environment()
        else:
            self.run_training()

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
        raise NotImplementedError("RL training loop not implemented")

    def run_environment(
        self,
        num_steps: int | None = None,
        render_visualization: bool = True,
    ) -> None:
        """Provides an easy-to-use interface for debugging environments.

        This function runs the environment for `num_steps`, rendering using
        MujocoViewer while simultaneously plotting the reward and termination
        information.

        Args:
            num_steps: The number of steps to run the environment for. If not
                provided, run until the user manually terminates the
                environment visualizer.
            render_visualization: If set, render the Mujoco visualizer.
        """
        with self, jax.disable_jit():
            rng = self.prng_key()
            self.set_loggers()

            rng, model_rng = jax.random.split(rng)
            model, _ = self.load_initial_state(model_rng, load_optimizer=False)

            mj_model: PhysicsModel = self.get_mujoco_model()
            metadata = self.get_mujoco_model_metadata(mj_model)
            engine = self.get_engine(mj_model, metadata)
            observations = self.get_observations(mj_model)
            commands = self.get_commands(mj_model)
            rewards = self.get_rewards(mj_model)
            terminations = self.get_terminations(mj_model)

            # Gets initial variables.
            initial_carry = self.get_initial_carry()
            initial_commands = get_initial_commands(rng, command_generators=commands)

            # Resets the physics state.
            rng, reset_rng = jax.random.split(rng)
            physics_state = engine.reset(mj_model, reset_rng)

            viewer: mujoco_viewer.MujocoViewer | None = None

            if render_visualization:
                viewer = mujoco_viewer.MujocoViewer(
                    mj_model,
                    physics_state.data,
                    mode="window",
                    height=self.config.render_height,
                    width=self.config.render_width,
                    hide_menus=True,
                )

                viewer.cam.distance = self.config.render_distance
                viewer.cam.azimuth = self.config.render_azimuth
                viewer.cam.elevation = self.config.render_elevation
                viewer.cam.lookat[:] = self.config.render_lookat

                if self.config.render_track_body_id is not None:
                    viewer.cam.trackbodyid = self.config.render_track_body_id
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

            # These components remain constant across the entire episode.
            engine_constants = EngineConstants(
                physics_model=mj_model,
                initial_carry=initial_carry,
                initial_command=initial_commands,
                obs_generators=observations,
                command_generators=commands,
                reward_generators=rewards,
                termination_generators=terminations,
            )

            # These components are updated each step.
            engine_variables = EngineVariables(
                carry=initial_carry,
                commands=initial_commands,
                physics_state=physics_state,
            )

            iterator = tqdm.trange(num_steps) if num_steps is not None else tqdm.tqdm(itertools.count())

            step_id = 0
            try:
                for step_id in iterator:
                    rng, step_rng = jax.random.split(rng)
                    transition, engine_variables = self.step_engine(
                        model=model,
                        engine=engine,
                        engine_constants=engine_constants,
                        engine_variables=engine_variables,
                        rng=step_rng,
                    )

                    if viewer is not None:
                        # We need to manually update the viewer data field, because
                        # resetting the environment creates a new data object rather
                        # than happening in-place, as Mujoco expects.
                        viewer.data = engine_variables.physics_state.data
                        viewer.render()

            except (KeyboardInterrupt, bdb.BdbQuit):
                logger.info("Keyboard interrupt, exiting environment loop")

            except Exception:
                # Raise on the first step for debugging purposes.
                if step_id == 0:
                    raise

                logger.info("Keyboard interrupt, exiting environment loop")

            finally:
                if viewer is not None:
                    viewer.close()

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
