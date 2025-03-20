"""Defines a standard task interface for training reinforcement learning agents."""

import bdb
import io
import itertools
import logging
import signal
import sys
import textwrap
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Collection, Generic, TypeVar

import chex
import equinox as eqx
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import optax
import PIL.Image
import tqdm
import xax
from dpshdl.dataset import Dataset
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx
from omegaconf import II, MISSING
from PIL import Image, ImageDraw

from ksim.actuators import Actuators
from ksim.commands import Command
from ksim.env.data import PhysicsModel, PhysicsState, Transition
from ksim.env.engine import (
    EngineConstants,
    EngineVariables,
    PhysicsEngine,
    engine_type_from_physics_model,
    get_physics_engine,
)
from ksim.observation import Observation
from ksim.randomization import Randomization
from ksim.resets import Reset
from ksim.rewards import Reward
from ksim.terminations import Termination
from ksim.utils.mujoco import get_joint_metadata
from ksim.viewer import MujocoViewer

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


def apply_randomizations(
    physics_model: PhysicsModel,
    randomizations: Collection[Randomization],
    rng: PRNGKeyArray,
) -> PhysicsModel:
    """Apply randomizations to the physics model."""
    for randomization in randomizations:
        rng, randomization_rng = jax.random.split(rng)
        physics_model = randomization(physics_model, randomization_rng)
    return physics_model


@jax.tree_util.register_dataclass
@dataclass
class RLConfig(xax.Config):
    # Debugging parameters.
    run_environment: bool = xax.field(
        value=False,
        help="Instead of dropping into the training loop, run the environment loop.",
    )
    run_environment_num_seconds: float | None = xax.field(
        value=None,
        help="If provided, run the environment loop for the given number of seconds.",
    )
    run_environment_save_path: str | None = xax.field(
        value=None,
        help="If provided, save the rendered video to the given path.",
    )

    # Training parameters.
    num_envs: int = xax.field(
        value=MISSING,
        help="The number of training environments to run in parallel.",
    )
    rollout_length_seconds: float = xax.field(
        value=MISSING,
        help="The number of seconds to rollout each environment during training.",
    )
    eval_rollout_length_seconds: float = xax.field(
        value=II("rollout_length_seconds"),
        help="The number of seconds to rollout the model for evaluation.",
    )

    # Rendering parameters.
    max_values_per_plot: int = xax.field(
        value=8,
        help="The maximum number of values to plot for each key.",
    )
    plot_figsize: tuple[float, float] = xax.field(
        value=(12, 6),
        help="The size of the figure for each plot.",
    )
    render_height: int = xax.field(
        value=240,
        help="The height of the rendered images.",
    )
    render_width: int = xax.field(
        value=320,
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

    # Engine parameters.
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
    def get_randomization(self, physics_model: PhysicsModel) -> Collection[Randomization]:
        """Returns randomizers, for randomizing each environment.

        Args:
            physics_model: The physics model to get the randomization for.

        Returns:
            A collection of randomization generators.
        """

    @abstractmethod
    def get_resets(self, physics_model: PhysicsModel) -> Collection[Reset]:
        """Returns the reset generators for the current task.

        Args:
            physics_model: The physics model to get the resets for.

        Returns:
            A collection of reset generators.
        """

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
    def get_initial_carry(self) -> PyTree | None:
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
    ) -> tuple[Array, PyTree | None, PyTree | None]:
        """Gets an action for the current observation.

        This function returns the action to take, the next carry (for models
        which look at multiple steps), and any auxiliary outputs. The auxiliary
        outputs get stored in the final transition object and can be used to
        compute metrics like log probabilities, values, etc.

        Args:
            model: The current model.
            physics_model: The physics model.
            observations: The current observations.
            commands: The current commands.
            carry: The model carry from the previous step.
            rng: The random key.

        Returns:
            The action to take, the next carry, and any auxiliary outputs.
        """

    @property
    def rollout_length_steps(self) -> int:
        return round(self.config.rollout_length_seconds / self.config.ctrl_dt)

    @property
    def eval_rollout_length_steps(self) -> int:
        return round(self.config.eval_rollout_length_seconds / self.config.ctrl_dt)

    def step_engine(
        self,
        physics_model: PhysicsModel,
        model: PyTree,
        engine: PhysicsEngine,
        engine_constants: EngineConstants,
        engine_variables: EngineVariables,
    ) -> tuple[Transition, EngineVariables]:
        """Runs a single step of the physics engine.

        Args:
            physics_model: The physics model.
            model: The model, with parameters to be updated.
            engine: The physics engine.
            engine_constants: The constants for the engine.
            engine_variables: The variables for the engine.

        Returns:
            A tuple containing the transition and the next engine variables.
        """
        rng, obs_rng, cmd_rng, act_rng, reset_rng, physics_rng = jax.random.split(engine_variables.rng, 6)

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
        action, next_carry, aux_outputs = self.sample_action(
            model=model,
            carry=engine_variables.carry,
            physics_model=physics_model,
            observations=observations,
            commands=commands,
            rng=act_rng,
        )

        # Steps the physics engine.
        next_physics_state: PhysicsState = engine.step(
            action=action,
            physics_model=physics_model,
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
            lambda: engine.reset(physics_model, reset_rng),
            lambda: next_physics_state,
        )
        next_carry = jax.lax.cond(
            terminated,
            lambda: self.get_initial_carry(),
            lambda: next_carry,
        )
        commands = jax.lax.cond(
            terminated,
            lambda: get_initial_commands(cmd_rng, command_generators=engine_constants.command_generators),
            lambda: commands,
        )

        # Combines all the relevant data into a single object.
        transition = Transition(
            qpos=next_physics_state.data.qpos,
            qvel=next_physics_state.data.qvel,
            obs=observations,
            command=commands,
            action=action,
            reward=reward,
            done=terminated,
            timestep=next_physics_state.data.time,
            termination_components=terminations,
            reward_components=rewards,
            aux_outputs=aux_outputs,
        )

        # Gets the variables for the next step.
        next_variables = EngineVariables(
            carry=next_carry,
            commands=commands,
            physics_state=next_physics_state,
            rng=rng,
        )

        return transition, next_variables

    def get_dataset(self, phase: xax.Phase) -> Dataset:
        raise NotImplementedError("RL tasks do not require datasets, since trajectory histories are stored in-memory.")

    def compute_loss(self, model: PyTree, batch: Any, output: Any) -> Array:  # noqa: ANN401
        raise NotImplementedError(
            "Direct compute_loss from TrainMixin is not expected to be called in RL tasks. "
            "PPO tasks use model_update and loss_metrics_grads instead."
        )

    def run(self) -> None:
        """Highest level entry point for RL tasks, determines what to run."""
        if self.config.run_environment:
            self.run_environment(
                num_steps=(
                    None
                    if self.config.run_environment_num_seconds is None
                    else round(self.config.run_environment_num_seconds / self.config.ctrl_dt)
                ),
                save_path=self.config.run_environment_save_path,
            )
        else:
            self.run_training()

    def log_reward_stats(
        self,
        transitions: Transition,
        reward_generators: Collection[Reward],
        namespace: str = "reward",
    ) -> None:
        """Log reward statistics from the trajectory or trajectories.

        Args:
            transitions: The transitions to log the reward statistics for.
            reward_generators: The reward generators to log the statistics for.
            namespace: The namespace to log the statistics to.
        """
        reward_stats: dict[str, jnp.ndarray] = {}

        num_episodes = jnp.sum(transitions.done).clip(min=1)
        terms = transitions.reward_components
        for generator in reward_generators:
            statistic = terms[generator.reward_name]
            assert isinstance(statistic, Array)
            reward_stats[generator.reward_name] = jnp.sum(statistic) / num_episodes

        for key, value in reward_stats.items():
            self.logger.log_scalar(key=key, value=value, namespace=namespace)

    def log_termination_stats(
        self,
        transitions: Transition,
        termination_generators: Collection[Termination],
        namespace: str = "termination",
    ) -> None:
        """Log termination statistics from the trajectory or trajectories.

        Args:
            transitions: The transitions to log the termination statistics for.
            termination_generators: The termination generators to log the statistics for.
            namespace: The namespace to log the statistics to.
        """
        termination_stats: dict[str, jnp.ndarray] = {}

        terms = transitions.termination_components
        for generator in termination_generators:
            statistic = terms[generator.termination_name]
            assert isinstance(statistic, Array)
            termination_stats[generator.termination_name] = jnp.mean(statistic)

        for key, value in termination_stats.items():
            self.logger.log_scalar(key=key, value=value, namespace=namespace)

        # Logs the mean episode length.
        episode_num_per_env = jnp.sum(transitions.done, axis=0) + (1 - transitions.done[-1])
        episode_count = jnp.sum(episode_num_per_env)
        num_env_states = jnp.prod(jnp.array(transitions.done.shape))
        mean_episode_length_steps = num_env_states / episode_count * self.config.ctrl_dt
        self.logger.log_scalar(key="mean_episode_seconds", value=mean_episode_length_steps, namespace=namespace)

    def log_train_metrics(self, train_metrics: dict[str, Array]) -> None:
        """Logs the train metrics.

        Args:
            train_metrics: The train metrics to log.
        """
        for key, value in train_metrics.items():
            self.logger.log_scalar(key=key, value=value, namespace="train")

    def render_trajectory_video(
        self,
        transitions: Transition,
        commands: Collection[Command],
        mj_model: mujoco.MjModel,
    ) -> tuple[np.ndarray, int]:
        """Render trajectory as video frames with computed FPS."""
        fps = round(1 / self.config.ctrl_dt)

        chex.assert_shape(transitions.done, (None,))
        num_steps = transitions.done.shape[0]
        transition_list: list[Transition] = [jax.tree.map(lambda arr: arr[i], transitions) for i in range(num_steps)]

        # Holds the current data.
        mj_data = mujoco.MjData(mj_model)

        # Builds the camera for viewing the scene.
        mj_camera = mujoco.MjvCamera()
        mj_camera.distance = self.config.render_distance
        mj_camera.azimuth = self.config.render_azimuth
        mj_camera.elevation = self.config.render_elevation
        mj_camera.lookat[:] = self.config.render_lookat
        if self.config.render_track_body_id is not None:
            mj_camera.trackbodyid = self.config.render_track_body_id
            mj_camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING

        renderer = mujoco.Renderer(
            mj_model,
            height=self.config.render_height,
            width=self.config.render_width,
        )

        frame_list: list[np.ndarray] = []
        for frame_id, transition in enumerate(transition_list):
            mj_data.qpos = np.array(transition.qpos)
            mj_data.qvel = np.array(transition.qvel)

            # Renders the current frame.
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=mj_camera)

            # Adds command elements to the scene.
            for command in commands:
                command.update_scene(renderer.scene, transition.command[command.command_name])

            # Renders the frame to a Numpy array.
            frame = renderer.render()

            # Overlays the frame number on the frame.
            frame_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame_img)
            draw.text((10, 10), f"Frame {frame_id}", fill=(255, 255, 255))
            frame = np.array(frame_img)

            frame_list.append(frame)

        return np.stack(frame_list, axis=0), fps

    def log_single_trajectory(
        self,
        transitions: Transition,
        commands: Collection[Command],
        mj_model: mujoco.MjModel,
    ) -> None:
        """Visualizes a single trajectory.

        Args:
            transitions: The transitions to visualize.
            commands: The commands to visualize.
            mj_model: The Mujoco model to render the scene with.
        """
        # Logs plots of the observations, commands, actions, rewards, and terminations.
        # Emojis are used in order to prevent conflicts with user-specified namespaces.
        for namespace, arr_dict in (
            ("👀 obs images", transitions.obs),
            ("🕹️ command images", transitions.command),
            ("🏃 action images", {"action": transitions.action}),
            ("💀 termination images", transitions.termination_components),
            ("🎁 reward images", transitions.reward_components),
        ):
            for key, value in arr_dict.items():
                plt.figure(figsize=self.config.plot_figsize)

                # Ensures a consistent shape and truncates if necessary.
                value = value.reshape(value.shape[0], -1)
                if value.shape[-1] > self.config.max_values_per_plot:
                    logger.warning("Truncating %s to %d values per plot.", key, self.config.max_values_per_plot)
                    value = value[..., : self.config.max_values_per_plot]

                for i in range(value.shape[1]):
                    plt.plot(value[:, i], label=f"{i}")

                if value.shape[1] > 1:
                    plt.legend()
                plt.title(key)

                # Converts to PIL image.
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                plt.close()
                buf.seek(0)
                img = Image.open(buf)

                # Logs the image.
                self.logger.log_image(key=key, value=img, namespace=namespace)

        # Logs the video of the trajectory.
        frames, fps = self.render_trajectory_video(transitions, commands, mj_model)
        self.logger.log_video(key="trajectory", value=frames, fps=fps, namespace="➡️ trajectory images")

    @eqx.filter_jit
    def _single_unroll(
        self,
        rng: PRNGKeyArray,
        physics_model: PhysicsModel,
        model: PyTree,
        engine: PhysicsEngine,
        engine_constants: EngineConstants,
        num_steps: int,
    ) -> Transition:
        initial_carry = self.get_initial_carry()
        rng, cmd_rng = jax.random.split(rng)
        initial_commands = get_initial_commands(cmd_rng, command_generators=engine_constants.command_generators)

        # Apply randomizations to the environment.
        rng, randomization_rng = jax.random.split(rng)
        physics_model = apply_randomizations(
            physics_model,
            engine_constants.randomization_generators,
            randomization_rng,
        )

        # Reset the physics state.
        rng, reset_rng = jax.random.split(rng)
        physics_state = engine.reset(physics_model, reset_rng)

        engine_variables = EngineVariables(
            carry=initial_carry,
            commands=initial_commands,
            physics_state=physics_state,
            rng=rng,
        )

        def scan_fn(carry: EngineVariables, _: None) -> tuple[EngineVariables, Transition]:
            transition, next_engine_variables = self.step_engine(
                physics_model=physics_model,
                model=model,
                engine=engine,
                engine_constants=engine_constants,
                engine_variables=carry,
            )
            return next_engine_variables, transition

        # Scans the engine for the desired number of steps.
        _, transitions = jax.lax.scan(scan_fn, engine_variables, length=num_steps)

        # Apply post_accumulate and scale to the fully unrolled rewards.
        reward_components = {
            reward_fn.reward_name: reward_fn.post_accumulate(
                transitions.reward_components[reward_fn.reward_name], transitions.done
            )
            * reward_fn.scale
            for reward_fn in engine_constants.reward_generators
        }
        rewards = jnp.stack(list(reward_components.values()), axis=1).sum(axis=-1)

        kwargs = transitions.__dict__
        kwargs["reward"] = rewards
        kwargs["reward_components"] = reward_components
        return Transition(**kwargs)

    @eqx.filter_jit
    def _vmapped_unroll(
        self,
        rng: PRNGKeyArray,
        physics_model: PhysicsModel,
        model: PyTree,
        engine: PhysicsEngine,
        engine_constants: EngineConstants,
        num_steps: int,
        num_envs: int,
    ) -> Transition:
        rngs = jax.random.split(rng, num_envs)
        vmapped_unroll = jax.vmap(self._single_unroll, in_axes=(0, None, None, None, None, None))
        return vmapped_unroll(rngs, physics_model, model, engine, engine_constants, num_steps)

    @abstractmethod
    def update_model(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        transitions: Transition,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.GradientTransformation, optax.OptState, FrozenDict[str, Array]]:
        """Updates the model on the given trajectory.

        This function should be implemented according to the specific RL method
        that we are using.

        Args:
            model: The model to update.
            optimizer: The optimizer to use.
            opt_state: The optimizer state.
            transitions: The trajectory to update the model on.
            rng: The random seed.

        Returns:
            A tuple containing the updated model, optimizer, optimizer state
            and metrics to log.
        """

    def rl_train_loop(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        state: xax.State,
        rng: PRNGKeyArray,
    ) -> None:
        mj_model: PhysicsModel = self.get_mujoco_model()
        mjx_model = self.get_mjx_model(mj_model)
        metadata = self.get_mujoco_model_metadata(mjx_model)
        engine = self.get_engine(mjx_model, metadata)
        observations = self.get_observations(mjx_model)
        commands = self.get_commands(mjx_model)
        rewards = self.get_rewards(mjx_model)
        terminations = self.get_terminations(mjx_model)
        randomizations = self.get_randomization(mjx_model)

        # These remain constant across the entire episode.
        engine_constants = EngineConstants(
            obs_generators=observations,
            command_generators=commands,
            reward_generators=rewards,
            termination_generators=terminations,
            randomization_generators=randomizations,
        )

        while not self.is_training_over(state):
            # Validate by sampling and visualizing a single trajectory.
            if self.valid_step_timer.is_valid_step(state):
                state.raw_phase = "valid"
                state.num_valid_steps += 1
                state.num_valid_samples += self.eval_rollout_length_steps

                rng, rollout_rng = jax.random.split(rng)
                transitions = self._single_unroll(
                    rng=rollout_rng,
                    physics_model=mjx_model,
                    model=model,
                    engine=engine,
                    engine_constants=engine_constants,
                    num_steps=self.eval_rollout_length_steps,
                )

                # Logs statistics from the trajectory.
                with self.step_context("write_logs"):
                    self.log_single_trajectory(transitions, commands, mj_model)
                    self.log_reward_stats(transitions, rewards)
                    self.log_termination_stats(transitions, terminations)
                    self.log_state_timers(state)
                    self.write_logs(state)

            with self.step_context("on_step_start"):
                state = self.on_step_start(state)

            # Samples N trajectories in parallel.
            with xax.ContextTimer() as timer:
                rng, rollout_rng = jax.random.split(rng)
                transitions = self._vmapped_unroll(
                    rng=rollout_rng,
                    physics_model=mjx_model,
                    model=model,
                    engine=engine,
                    engine_constants=engine_constants,
                    num_steps=self.rollout_length_steps,
                    num_envs=self.config.num_envs,
                )
            self.logger.log_scalar("rollout_dt", timer.elapsed_time, namespace="⏰")

            # Optimizes the model on that trajectory.
            with xax.ContextTimer() as timer:
                rng, update_rng = jax.random.split(rng)
                model, opt_state, train_metrics = self.update_model(
                    model=model,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    transitions=transitions,
                    rng=update_rng,
                )
            self.logger.log_scalar("update_dt", timer.elapsed_time, namespace="⏰")

            with self.step_context("write_logs"):
                state.raw_phase = "train"
                state.num_steps += 1
                state.num_samples += self.rollout_length_steps * self.config.num_envs

                # Logs statistics from the trajectory.
                with self.step_context("write_logs"):
                    self.log_reward_stats(transitions, rewards)
                    self.log_termination_stats(transitions, terminations)
                    self.log_train_metrics(train_metrics)
                    self.log_state_timers(state)
                    self.write_logs(state)

            with self.step_context("on_step_end"):
                state = self.on_step_end(state)

            if self.should_checkpoint(state):
                self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    state=state,
                )

    def run_environment(
        self,
        num_steps: int | None = None,
        save_path: str | Path | None = None,
    ) -> None:
        """Provides an easy-to-use interface for debugging environments.

        This function runs the environment for `num_steps`, rendering using
        MujocoViewer while simultaneously plotting the reward and termination
        information.

        Args:
            num_steps: The number of steps to run the environment for. If not
                provided, run until the user manually terminates the
                environment visualizer.
            save_path: If provided, save the rendered video to the given path.
        """
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

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
            randomizations = self.get_randomization(mj_model)

            # Gets initial variables.
            initial_carry = self.get_initial_carry()
            rng, cmd_rng = jax.random.split(rng)
            initial_commands = get_initial_commands(cmd_rng, command_generators=commands)

            # Resets the physics state.
            rng, reset_rng = jax.random.split(rng)
            physics_state = engine.reset(mj_model, reset_rng)

            viewer = MujocoViewer(
                mj_model,
                physics_state.data,
                mode="window" if save_path is None else "offscreen",
                height=self.config.render_height,
                width=self.config.render_width,
            )

            # Sets the viewer camera.
            viewer.cam.distance = self.config.render_distance
            viewer.cam.azimuth = self.config.render_azimuth
            viewer.cam.elevation = self.config.render_elevation
            viewer.cam.lookat[:] = self.config.render_lookat
            if self.config.render_track_body_id is not None:
                viewer.cam.trackbodyid = self.config.render_track_body_id
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

            # These components remain constant across the entire episode.
            engine_constants = EngineConstants(
                obs_generators=observations,
                command_generators=commands,
                reward_generators=rewards,
                termination_generators=terminations,
                randomization_generators=randomizations,
            )

            # These components are updated each step.
            engine_variables = EngineVariables(
                carry=initial_carry,
                commands=initial_commands,
                physics_state=physics_state,
                rng=rng,
            )

            iterator = tqdm.trange(num_steps) if num_steps is not None else tqdm.tqdm(itertools.count())

            step_id = 0
            frames: list[np.ndarray] = []
            try:
                for step_id in iterator:
                    transition, engine_variables = self.step_engine(
                        physics_model=mj_model,
                        model=model,
                        engine=engine,
                        engine_constants=engine_constants,
                        engine_variables=engine_variables,
                    )

                    # We manually trigger randomizations on termination,
                    # whereas during training the randomization is only applied
                    # once per rollout for efficiency.
                    rng, randomization_rng = jax.random.split(rng)
                    mj_model = jax.lax.cond(
                        transition.done,
                        lambda: apply_randomizations(mj_model, randomizations, randomization_rng),
                        lambda: mj_model,
                    )

                    # We need to manually update the viewer data field, because
                    # resetting the environment creates a new data object rather
                    # than happening in-place, as Mujoco expects.
                    viewer.data = engine_variables.physics_state.data

                    # Adds command elements to the scene.
                    for command in commands:
                        command.update_scene(viewer.scn, engine_variables.commands[command.command_name])

                    # Logs the frames to render.
                    if save_path is None:
                        viewer.render()
                    else:
                        frames.append(viewer.read_pixels(depth=False))

            except (KeyboardInterrupt, bdb.BdbQuit):
                logger.info("Keyboard interrupt, exiting environment loop")

            except Exception:
                # Raise on the first step for debugging purposes.
                if step_id <= 1:
                    raise

                logger.info("Keyboard interrupt, exiting environment loop")

            finally:
                if viewer is not None:
                    viewer.close()

                if save_path is not None:
                    if len(frames) == 0:
                        raise ValueError("No frames to save")

                    fps = round(1 / self.config.ctrl_dt)

                    match save_path.suffix.lower():
                        case ".mp4":
                            try:
                                with imageio.get_writer(save_path, mode="I", fps=fps) as writer:
                                    for frame in frames:
                                        writer.append_data(frame)

                            except Exception as e:
                                raise RuntimeError(
                                    "Failed to save video - note that saving .mp4 videos with imageio usually "
                                    "requires the FFMPEG backend, which can be installed using `pip install "
                                    "'imageio[ffmpeg]'`. Note that this also requires FFMPEG to be installed in "
                                    "your system."
                                ) from e

                        case ".gif":
                            images = [PIL.Image.fromarray(frame) for frame in frames]
                            images[0].save(
                                save_path,
                                save_all=True,
                                append_images=images[1:],
                                duration=int(1000 / fps),
                                loop=0,
                            )

                        case _:
                            raise ValueError(f"Unsupported file extension: {save_path.suffix}. Expected .mp4 or .gif")

    def run_training(self) -> None:
        """Wraps the training loop and provides clean XAX integration."""
        with self:
            rng = self.prng_key()
            self.set_loggers()

            if xax.is_master():
                Thread(target=self.log_state, daemon=True).start()

            rng, model_rng = jax.random.split(rng)
            model, optimizer, opt_state, training_state = self.load_initial_state(model_rng, load_optimizer=True)

            training_state = self.on_training_start(training_state)
            training_state.num_samples = 1  # prevents from checkpointing at start

            def on_exit() -> None:
                self.save_checkpoint(model, optimizer, opt_state, training_state)

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

            try:
                self.rl_train_loop(
                    model=model,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    state=training_state,
                    rng=rng,
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
                exception_tb = textwrap.indent(xax.highlight_exception_message(traceback.format_exc()), "  ")
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()
                self.save_checkpoint(model, optimizer, opt_state, training_state)

            finally:
                training_state = self.on_training_end(training_state)
