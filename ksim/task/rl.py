"""Defines a standard task interface for training reinforcement learning agents."""

__all__ = [
    "RLConfig",
    "RLTask",
]

import bdb
import datetime
import functools
import io
import itertools
import logging
import signal
import sys
import textwrap
import time
import traceback
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any, Collection, Generic, TypeVar

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import optax
import tqdm
import xax
from dpshdl.dataset import Dataset
from jaxtyping import Array, PRNGKeyArray, PyTree
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx
from omegaconf import MISSING
from PIL import Image, ImageDraw

from ksim.actuators import Actuators
from ksim.commands import Command
from ksim.engine import (
    PhysicsEngine,
    engine_type_from_physics_model,
    get_physics_engine,
)
from ksim.events import Event
from ksim.observation import Observation
from ksim.randomization import Randomization
from ksim.resets import Reset
from ksim.rewards import Reward
from ksim.terminations import Termination
from ksim.types import (
    Histogram,
    Metrics,
    PhysicsModel,
    PhysicsState,
    Rewards,
    RolloutVariables,
    Trajectory,
)
from ksim.utils.mujoco import get_joint_metadata, load_model
from ksim.vis import Marker, configure_scene

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RolloutConstants:
    obs_generators: Collection[Observation]
    command_generators: Collection[Command]
    reward_generators: Collection[Reward]
    termination_generators: Collection[Termination]


def get_observation(
    rollout_state: RolloutVariables,
    rng: PRNGKeyArray,
    *,
    obs_generators: Collection[Observation],
) -> xax.FrozenDict[str, Array]:
    """Get the observation from the physics state."""
    observations = {}
    for observation in obs_generators:
        rng, obs_rng = jax.random.split(rng)
        observation_value = observation(rollout_state, obs_rng)
        observations[observation.observation_name] = observation_value
    return xax.FrozenDict(observations)


def get_rewards(
    trajectory: Trajectory,
    reward_generators: Collection[Reward],
    ctrl_dt: float,
    clip_max: float | None = None,
) -> Rewards:
    """Get the rewards from the physics state."""
    rewards = {}
    target_shape = trajectory.done.shape
    for reward_generator in reward_generators:
        reward_name = reward_generator.reward_name
        reward_val = reward_generator(trajectory) * reward_generator.scale * ctrl_dt
        if reward_val.shape != trajectory.done.shape:
            raise AssertionError(f"Reward {reward_name} shape {reward_val.shape} does not match {target_shape}")
        if clip_max is not None:
            reward_val = jnp.clip(reward_val, -clip_max, clip_max)
        rewards[reward_generator.reward_name] = reward_val
    total_reward = jax.tree.reduce(jnp.add, list(rewards.values()))
    return Rewards(total=total_reward, components=xax.FrozenDict(rewards))


def get_terminations(
    physics_state: PhysicsState,
    termination_generators: Collection[Termination],
) -> xax.FrozenDict[str, Array]:
    """Get the terminations from the physics state."""
    terminations = {}
    for termination in termination_generators:
        termination_val = termination(physics_state.data)
        chex.assert_type(termination_val, bool)
        name = termination.termination_name
        terminations[name] = termination_val
    return xax.FrozenDict(terminations)


def get_commands(
    prev_commands: xax.FrozenDict[str, Array],
    physics_state: PhysicsState,
    rng: PRNGKeyArray,
    command_generators: Collection[Command],
) -> xax.FrozenDict[str, Array]:
    """Get the commands from the physics state."""
    commands = {}
    for command_generator in command_generators:
        rng, cmd_rng = jax.random.split(rng)
        command_name = command_generator.command_name
        prev_command = prev_commands[command_name]
        assert isinstance(prev_command, Array)
        command_val = command_generator(prev_command, physics_state.data.time, cmd_rng)
        commands[command_name] = command_val
    return xax.FrozenDict(commands)


def get_initial_commands(
    rng: PRNGKeyArray,
    command_generators: Collection[Command],
) -> xax.FrozenDict[str, Array]:
    """Get the initial commands from the physics state."""
    commands = {}
    for command_generator in command_generators:
        rng, cmd_rng = jax.random.split(rng)
        command_name = command_generator.command_name
        command_val = command_generator.initial_command(cmd_rng)
        commands[command_name] = command_val
    return xax.FrozenDict(commands)


def get_randomizations(
    physics_model: PhysicsModel,
    randomizations: Collection[Randomization],
    rng: PRNGKeyArray,
) -> xax.FrozenDict[str, Array]:
    all_randomizations: dict[str, dict[str, Array]] = {}
    for randomization in randomizations:
        rng, randomization_rng = jax.random.split(rng)
        all_randomizations[randomization.randomization_name] = randomization(physics_model, randomization_rng)
    for name, count in Counter([k for d in all_randomizations.values() for k in d.keys()]).items():
        if count > 1:
            name_to_keys = {k: set(v.keys()) for k, v in all_randomizations.items()}
            raise ValueError(f"Found duplicate randomization keys: {name}. Randomizations: {name_to_keys}")
    return xax.FrozenDict({k: v for d in all_randomizations.values() for k, v in d.items()})


def apply_randomizations(
    physics_model: mjx.Model,
    engine: PhysicsEngine,
    randomizations: Collection[Randomization],
    rng: PRNGKeyArray,
) -> tuple[xax.FrozenDict[str, Array], PhysicsState]:
    rand_rng, reset_rng = jax.random.split(rng)
    randomizations = get_randomizations(physics_model, randomizations, rand_rng)
    physics_state = engine.reset(physics_model.tree_replace(randomizations), reset_rng)
    return randomizations, physics_state


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

    # Logging parameters.
    log_train_metrics: bool = xax.field(
        value=True,
        help="If true, log train metrics.",
    )
    epochs_per_log_step: int = xax.field(
        value=1,
        help="The number of epochs between logging steps.",
    )

    # Training parameters.
    num_envs: int = xax.field(
        value=MISSING,
        help="The number of training environments to run in parallel.",
    )
    batch_size: int = xax.field(
        value=1,
        help="The number of model update batches per trajectory batch. ",
    )
    rollout_length_seconds: float = xax.field(
        value=MISSING,
        help="The number of seconds to rollout each environment during training.",
    )

    # Override validation parameters.
    log_full_trajectory_on_first_step: bool = xax.field(
        value=False,
        help="If true, log the full trajectory on the first step.",
    )
    log_full_trajectory_every_n_steps: int | None = xax.field(
        None,
        help="Log the full trajectory every N steps.",
    )
    log_full_trajectory_every_n_seconds: float | None = xax.field(
        60.0 * 10.0,
        help="Log the full trajectory every N seconds.",
    )

    # Rendering parameters.
    max_values_per_plot: int = xax.field(
        value=8,
        help="The maximum number of values to plot for each key.",
    )
    plot_figsize: tuple[float, float] = xax.field(
        value=(8, 4),
        help="The size of the figure for each plot.",
    )
    render_shadow: bool = xax.field(
        value=False,
        help="If true, render shadows.",
    )
    render_reflection: bool = xax.field(
        value=False,
        help="If true, render reflections.",
    )
    render_contact_force: bool = xax.field(
        value=False,
        help="If true, render contact forces.",
    )
    render_contact_point: bool = xax.field(
        value=False,
        help="If true, render contact points.",
    )
    render_inertia: bool = xax.field(
        value=False,
        help="If true, render inertia.",
    )
    render_height: int = xax.field(
        value=360,
        help="The height of the rendered images.",
    )
    render_width: int = xax.field(
        value=480,
        help="The width of the rendered images.",
    )
    render_length_seconds: float | None = xax.field(
        value=5.0,
        help="The number of seconds to rollout each environment during evaluation.",
    )
    render_fps: int | None = xax.field(
        value=24,
        help="The target FPS for the renderered video.",
    )
    render_slowdown: float = xax.field(
        value=2.0,
        help="The slowdown factor for the rendered video.",
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
    reward_clip_max: float | None = xax.field(
        value=None,
        help="The maximum value of the reward.",
    )


Config = TypeVar("Config", bound=RLConfig)


class RLTask(xax.Task[Config], Generic[Config], ABC):
    """Base class for reinforcement learning tasks."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        if self.config.num_envs % self.config.batch_size != 0:
            raise ValueError(
                f"The number of environments ({self.config.num_envs}) must be divisible by "
                f"the batch size ({self.config.batch_size})"
            )

    @functools.cached_property
    def batch_size(self) -> int:
        return self.config.batch_size

    @functools.cached_property
    def num_batches(self) -> int:
        return self.config.num_envs // self.batch_size

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
        return load_model(mj_model)

    def get_engine(
        self,
        physics_model: PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> PhysicsEngine:
        return get_physics_engine(
            engine_type=engine_type_from_physics_model(physics_model),
            resets=self.get_resets(physics_model),
            events=self.get_events(physics_model),
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
    def get_events(self, physics_model: PhysicsModel) -> Collection[Event]:
        """Returns the event generators for the current task.

        Args:
            physics_model: The physics model to get the events for.
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
    def get_initial_carry(self, rng: PRNGKeyArray) -> PyTree | None:
        """Returns the initial carry for the model.

        Args:
            rng: The random key to use.

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
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, PyTree | None, PyTree | None]:
        """Gets an action for the current observation.

        This function returns the action to take, the next carry (for models
        which look at multiple steps), and any auxiliary outputs. The auxiliary
        outputs get stored in the final trajectory object and can be used to
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
    def rollout_num_samples(self) -> int:
        return self.rollout_length_steps * self.config.num_envs * self.config.epochs_per_log_step

    def step_engine(
        self,
        physics_model: PhysicsModel,
        model: PyTree,
        engine: PhysicsEngine,
        rollout_constants: RolloutConstants,
        rollout_variables: RolloutVariables,
    ) -> tuple[Trajectory, RolloutVariables]:
        """Runs a single step of the physics engine.

        Args:
            physics_model: The physics model.
            model: The model, with parameters to be updated.
            engine: The physics engine.
            rollout_constants: The constants for the engine.
            rollout_variables: The variables for the engine.

        Returns:
            A tuple containing the trajectory and the next engine variables.
        """
        rng, obs_rng, cmd_rng, act_rng, reset_rng, carry_rng, physics_rng = jax.random.split(rollout_variables.rng, 7)

        # Gets the observations from the physics state.
        observations = get_observation(
            rollout_state=rollout_variables,
            rng=obs_rng,
            obs_generators=rollout_constants.obs_generators,
        )

        # Gets the commmands from the previous commands and the physics state.
        commands = get_commands(
            prev_commands=rollout_variables.commands,
            physics_state=rollout_variables.physics_state,
            rng=cmd_rng,
            command_generators=rollout_constants.command_generators,
        )

        # Samples an action from the model.
        action, next_carry, aux_outputs = self.sample_action(
            model=model,
            carry=rollout_variables.carry,
            physics_model=physics_model,
            observations=observations,
            commands=commands,
            rng=act_rng,
        )

        # Steps the physics engine.
        next_physics_state: PhysicsState = engine.step(
            action=action,
            physics_model=physics_model,
            physics_state=rollout_variables.physics_state,
            rng=physics_rng,
        )

        # Gets termination components and a single termination boolean.
        terminations = get_terminations(
            physics_state=rollout_variables.physics_state,
            termination_generators=rollout_constants.termination_generators,
        )
        terminated = jax.tree.reduce(jnp.logical_or, list(terminations.values()))

        # Conditionally reset on termination.
        next_physics_state = jax.lax.cond(
            terminated,
            lambda: engine.reset(physics_model, reset_rng),
            lambda: next_physics_state,
        )

        next_carry = jax.lax.cond(
            terminated,
            lambda: self.get_initial_carry(carry_rng),
            lambda: next_carry,
        )
        commands = jax.lax.cond(
            terminated,
            lambda: get_initial_commands(cmd_rng, command_generators=rollout_constants.command_generators),
            lambda: commands,
        )

        # Combines all the relevant data into a single object.
        trajectory = Trajectory(
            qpos=next_physics_state.data.qpos,
            qvel=next_physics_state.data.qvel,
            obs=observations,
            command=commands,
            action=action,
            done=terminated,
            timestep=next_physics_state.data.time,
            termination_components=terminations,
            aux_outputs=aux_outputs,
        )

        # Gets the variables for the next step.
        next_variables = RolloutVariables(
            carry=next_carry,
            commands=commands,
            physics_state=next_physics_state,
            rng=rng,
        )

        return trajectory, next_variables

    def get_dataset(self, phase: xax.Phase) -> Dataset:
        raise NotImplementedError("RL tasks do not require datasets, since trajectory histories are stored in-memory.")

    def compute_loss(self, model: PyTree, batch: Any, output: Any, state: xax.State) -> Array:  # noqa: ANN401
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

    def log_train_metrics(self, metrics: Metrics) -> None:
        """Logs the train metrics.

        Args:
            metrics: The metrics to log.
        """
        if self.config.log_train_metrics:
            for namespace, metric in (
                ("🚂 train", metrics.train),
                ("🎁 reward", metrics.reward),
                ("💀 termination", metrics.termination),
            ):
                for key, value in metric.items():
                    if isinstance(value, Histogram):
                        self.logger.log_histogram_raw(
                            key,
                            counts=value.counts,
                            limits=value.limits,
                            minv=value.min,
                            maxv=value.max,
                            sumv=value.sum,
                            sum_squaresv=value.sum_squares,
                            namespace=f"{namespace} histograms",
                        )
                        self.logger.log_scalar(key, value.mean, namespace=namespace)
                    else:
                        self.logger.log_scalar(key, value.mean(), namespace=namespace)

    def render_trajectory_video(
        self,
        trajectories: Trajectory,
        markers: Collection[Marker],
        mj_model: mujoco.MjModel,
        mj_renderer: mujoco.Renderer,
        target_fps: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Render trajectory as video frames with computed FPS."""
        fps = round(1 / self.config.ctrl_dt)

        # Rerenders the trajectory at the desired FPS.
        num_frames = len(trajectories.done)
        if target_fps is not None:
            indices = jnp.arange(0, num_frames, fps / target_fps, dtype=jnp.int32).clip(max=num_frames - 1)
            trajectories = jax.tree.map(lambda arr: arr[indices], trajectories)
            fps = target_fps
        else:
            indices = jnp.arange(0, num_frames, dtype=jnp.int32)

        chex.assert_shape(trajectories.done, (None,))
        num_steps = trajectories.done.shape[0]
        trajectory_list: list[Trajectory] = [jax.tree.map(lambda arr: arr[i], trajectories) for i in range(num_steps)]

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

        frame_list: list[np.ndarray] = []
        for frame_id, trajectory in enumerate(trajectory_list):
            mj_data.qpos = np.array(trajectory.qpos)
            mj_data.qvel = np.array(trajectory.qvel)

            # Renders the current frame.
            mujoco.mj_forward(mj_model, mj_data)
            mj_renderer.update_scene(mj_data, camera=mj_camera)

            # For some reason, using markers here will sometimes cause weird
            # segfaults. Disabling for now because I can't debug this.
            # for marker in markers:
            #     marker(mj_renderer.model, mj_data, mj_renderer.scene, trajectory)

            frame = mj_renderer.render()

            # Overlays the frame number on the frame.
            frame_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame_img)
            draw.text((10, 10), f"Frame {indices[frame_id]}", fill=(255, 255, 255))
            frame = np.array(frame_img)

            # Draws an RGB patch in the bottom right corner of the frame.
            rgb = np.zeros((5, 25, 3), dtype=np.uint8)
            rgb[:, 5:10, 0] = 255
            rgb[:, 10:15, 1] = 255
            rgb[:, 15:20, 2] = 255
            rgb[:, 20:, :] = 255
            frame[-5:, -25:] = rgb

            frame_list.append(frame)

        return np.stack(frame_list, axis=0), fps

    def log_single_trajectory(
        self,
        trajectories: Trajectory,
        reward_values: Rewards,
        markers: Collection[Marker],
        mj_model: mujoco.MjModel,
        mj_renderer: mujoco.Renderer,
    ) -> None:
        """Visualizes a single trajectory.

        Args:
            trajectories: The trajectories to visualize.
            reward_values: The collected rewards to visualize.
            markers: The markers to visualize.
            mj_model: The Mujoco model to render the scene with.
            mj_renderer: The Mujoco renderer to render the scene with.
            name: The name of the trajectory being logged.
        """
        # Clips the trajectory to the desired length.
        if self.config.render_length_seconds is not None:
            render_frames = round(self.config.render_length_seconds / self.config.ctrl_dt)
            trajectories = jax.tree.map(lambda arr: arr[:render_frames], trajectories)
            reward_values = jax.tree.map(lambda arr: arr[:render_frames], reward_values)

        # Logs plots of the observations, commands, actions, rewards, and terminations.
        # Logs plots of the observations, commands, actions, rewards, and terminations.
        # Emojis are used in order to prevent conflicts with user-specified namespaces.
        for namespace, arr_dict in (
            ("👀 obs images", trajectories.obs),
            ("🕹️ command images", trajectories.command),
            ("🏃 action images", {"action": trajectories.action}),
            ("💀 termination images", trajectories.termination_components),
            ("🎁 reward images", reward_values.components),
            ("🎁 reward images", {"total": reward_values.total}),
        ):
            for key, value in arr_dict.items():
                plt.figure(figsize=self.config.plot_figsize)

                # Ensures a consistent shape and truncates if necessary.
                value = value.reshape(value.shape[0], -1)
                if value.shape[-1] > self.config.max_values_per_plot:
                    logger.debug("Truncating %s to %d values per plot.", key, self.config.max_values_per_plot)
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
        frames, fps = self.render_trajectory_video(
            trajectories=trajectories,
            markers=markers,
            mj_model=mj_model,
            mj_renderer=mj_renderer,
            target_fps=self.config.render_fps,
        )
        self.logger.log_video(
            key="trajectory",
            value=frames,
            fps=round(fps / self.config.render_slowdown),
            namespace="➡️ trajectory images",
        )

    @abstractmethod
    def update_model(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectories: Trajectory,
        rewards: Rewards,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, xax.FrozenDict[str, Array]]:
        """Updates the model on the given trajectory.

        This function should be implemented according to the specific RL method
        that we are using.

        Args:
            model_arr: The mutable part of the model to update.
            model_static: The static part of the model to update.
            optimizer: The optimizer to use.
            opt_state: The optimizer state.
            trajectories: The trajectories to update the model on.
            rewards: The rewards to update the model on.
            rng: The random seed.

        Returns:
            A tuple containing the updated model, optimizer state
            and metrics to log. If a metric has a single element it is logged
            as a scalar, otherwise it is logged as a histogram.
        """

    def get_histogram(self, arr: Array, bins: int = 100) -> Histogram:
        """Gets the histogram of the given array.

        Args:
            arr: The array to get the histogram for.
            bins: The number of bins to use for the histogram.

        Returns:
            The histogram of the given array.
        """
        arr = arr.flatten()
        counts, limits = jnp.histogram(arr, bins=bins)
        return Histogram(
            counts=counts,
            limits=limits[..., 1:],
            min=arr.min(),
            max=arr.max(),
            sum=arr.sum(),
            sum_squares=arr.dot(arr),
            mean=arr.mean(),
        )

    def get_reward_metrics(self, trajectories: Trajectory, rewards: Rewards) -> dict[str, Array]:
        """Gets the reward metrics.

        Args:
            trajectories: The trajectories to get the reward metrics for.
            rewards: The rewards to get the metrics for.
        """
        return {
            "total": rewards.total,
            **{key: value for key, value in rewards.components.items()},
        }

    def get_termination_metrics(self, trajectories: Trajectory) -> dict[str, Array]:
        """Gets the termination metrics.

        Args:
            trajectories: The trajectories to get the termination metrics for.
        """
        kvs = list(trajectories.termination_components.items())
        all_terminations = jnp.stack([v for _, v in kvs], axis=-1)
        has_termination = (all_terminations.any(axis=-1)).sum(axis=-1)
        num_terminations = has_termination.sum().clip(min=1)
        num_timesteps = trajectories.done.shape[-1]

        return {
            "episode_length": (num_timesteps / (has_termination + 1).mean()) * self.config.ctrl_dt,
            **{key: (value.sum() / num_terminations) for key, value in kvs},
        }

    def get_markers(
        self,
        commands: Collection[Command],
        observations: Collection[Observation],
        randomizations: Collection[Randomization],
    ) -> Collection[Marker]:
        markers: list[Marker] = []
        for command in commands:
            markers.extend(command.get_markers())
        for observation in observations:
            markers.extend(observation.get_markers())
        for randomization in randomizations:
            markers.extend(randomization.get_markers())
        return markers

    @xax.jit(
        static_argnames=[
            "self",
            "model_static",
            "optimizer",
            "engine",
            "rollout_constants",
        ]
    )
    def _rl_train_loop_step(
        self,
        physics_model: PhysicsModel,
        randomization_dict: xax.FrozenDict[str, Array],
        model_arr: PyTree,
        model_static: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        engine: PhysicsEngine,
        rollout_constants: RolloutConstants,
        rollout_variables: RolloutVariables,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, Metrics, RolloutVariables, Trajectory, Rewards]:
        def single_unroll(
            physics_model: PhysicsModel,
            randomization_dict: xax.FrozenDict[str, Array],
            model_arr: PyTree,
            model_static: PyTree,
            engine: PhysicsEngine,
            rollout_constants: RolloutConstants,
            rollout_variables: RolloutVariables,
            num_steps: int,
        ) -> tuple[Trajectory, Rewards, RolloutVariables]:
            # Recombines the mutable and static parts of the model.
            model = eqx.combine(model_arr, model_static)

            # Applies randomizations to the model.
            physics_model = physics_model.tree_replace(randomization_dict)

            def scan_fn(carry: RolloutVariables, _: None) -> tuple[RolloutVariables, Trajectory]:
                trajectory, next_rollout_variables = self.step_engine(
                    physics_model=physics_model,
                    model=model,
                    engine=engine,
                    rollout_constants=rollout_constants,
                    rollout_variables=carry,
                )
                return next_rollout_variables, trajectory

            # Scans the engine for the desired number of steps.
            next_rollout_variables, trajectory = jax.lax.scan(scan_fn, rollout_variables, length=num_steps)

            # Gets the rewards.
            reward = get_rewards(
                trajectory=trajectory,
                reward_generators=rollout_constants.reward_generators,
                ctrl_dt=self.config.ctrl_dt,
                clip_max=self.config.reward_clip_max,
            )

            return trajectory, reward, next_rollout_variables

        def single_step_fn(
            carry: tuple[PyTree, optax.OptState, RolloutVariables],
            rng: PRNGKeyArray,
        ) -> tuple[tuple[PyTree, optax.OptState, RolloutVariables], tuple[Metrics, Trajectory, Rewards]]:
            model_arr, opt_state, rollout_variables = carry

            # Rolls out a new trajectory.
            vmapped_unroll = jax.vmap(
                single_unroll,
                in_axes=(
                    None,
                    0,
                    None,
                    None,
                    None,
                    None,
                    0,
                    None,
                ),
            )
            trajectories, rewards, rollout_variables = vmapped_unroll(
                physics_model,
                randomization_dict,
                model_arr,
                model_static,
                engine,
                rollout_constants,
                rollout_variables,
                self.rollout_length_steps,
            )

            # Runs update on the previous trajectory.
            model_arr, opt_state, train_metrics = self.update_model(
                model_arr=model_arr,
                model_static=model_static,
                optimizer=optimizer,
                opt_state=opt_state,
                trajectories=trajectories,
                rewards=rewards,
                rng=rng,
            )

            metrics = Metrics(
                train=train_metrics,
                reward=xax.FrozenDict(self.get_reward_metrics(trajectories, rewards)),
                termination=xax.FrozenDict(self.get_termination_metrics(trajectories)),
            )

            # Saving the last trajectory for visualization.
            final_trajectory = jax.tree.map(lambda arr: arr[-1], trajectories)
            final_rewards = jax.tree.map(lambda arr: arr[-1], rewards)

            return (model_arr, opt_state, rollout_variables), (metrics, final_trajectory, final_rewards)

        ((model_arr, opt_state, rollout_variables), (metrics, final_trajectories, final_rewards)) = jax.lax.scan(
            single_step_fn,
            (model_arr, opt_state, rollout_variables),
            jax.random.split(rng, self.config.epochs_per_log_step),
        )

        # Convert any array with more than one element to a histogram.
        metrics = jax.tree.map(lambda x: self.get_histogram(x) if isinstance(x, Array) and x.size > 1 else x, metrics)

        # Only get final trajectory and rewards.
        final_trajectory = jax.tree.map(lambda arr: arr[-1], final_trajectories)
        final_reward = jax.tree.map(lambda arr: arr[-1], final_rewards)

        # Metrics, final_trajectories, final_rewards batch dim of epochs.
        # Rollout variables has batch dim of num_envs and are used next rollout.
        return model_arr, opt_state, metrics, rollout_variables, final_trajectory, final_reward

    def on_context_stop(self, step: str, elapsed_time: float) -> None:
        super().on_context_stop(step, elapsed_time)

        self.logger.log_scalar(key=step, value=elapsed_time, namespace="⌛ dt")

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
            terminations = self.get_terminations(mj_model)
            rewards = self.get_rewards(mj_model)
            randomizations = self.get_randomization(mj_model)

            # Creates the markers.
            markers = self.get_markers(
                commands=commands,
                observations=observations,
                randomizations=randomizations,
            )

            # Gets initial variables.
            rng, carry_rng, cmd_rng = jax.random.split(rng, 3)
            initial_carry = self.get_initial_carry(carry_rng)
            initial_commands = get_initial_commands(cmd_rng, command_generators=commands)

            def apply_randomizations_to_mujoco(rng: PRNGKeyArray) -> PhysicsState:
                rand_rng, reset_rng = jax.random.split(rng)
                rand_dict = get_randomizations(mj_model, randomizations, rand_rng)
                for k, v in rand_dict.items():
                    setattr(mj_model, k, v)
                return engine.reset(mj_model, reset_rng)

            # Resets the physics state.
            rng, reset_rng = jax.random.split(rng)
            physics_state = apply_randomizations_to_mujoco(reset_rng)

            try:
                from ksim.viewer import MujocoViewer

            except ModuleNotFoundError:
                raise ModuleNotFoundError("glfw not installed - install with `pip install glfw`")

            viewer = MujocoViewer(
                mj_model,
                physics_state.data,
                mode="window" if save_path is None else "offscreen",
                height=self.config.render_height,
                width=self.config.render_width,
                shadow=self.config.render_shadow,
                reflection=self.config.render_reflection,
                contact_force=self.config.render_contact_force,
                contact_point=self.config.render_contact_point,
                inertia=self.config.render_inertia,
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
            rollout_constants = RolloutConstants(
                obs_generators=observations,
                command_generators=commands,
                reward_generators=rewards,
                termination_generators=terminations,
            )

            # These components are updated each step.
            rollout_variables = RolloutVariables(
                carry=initial_carry,
                commands=initial_commands,
                physics_state=physics_state,
                rng=rng,
            )

            iterator = tqdm.trange(num_steps) if num_steps is not None else tqdm.tqdm(itertools.count())
            frames: list[np.ndarray] = []

            def reset_mujoco_model(
                physics_model: mujoco.MjModel,
                rollout_variables: RolloutVariables,
                rng: PRNGKeyArray,
            ) -> tuple[mujoco.MjModel, RolloutVariables]:
                rng, rand_rng, carry_rng = jax.random.split(rng, 3)
                physics_state = apply_randomizations_to_mujoco(rand_rng)
                rollout_variables = RolloutVariables(
                    carry=self.get_initial_carry(carry_rng),
                    commands=rollout_variables.commands,
                    physics_state=physics_state,
                    rng=rng,
                )
                return physics_model, rollout_variables

            try:
                transitions = []
                for _ in iterator:
                    transition, rollout_variables = self.step_engine(
                        physics_model=mj_model,
                        model=model,
                        engine=engine,
                        rollout_constants=rollout_constants,
                        rollout_variables=rollout_variables,
                    )
                    transitions.append(transition)
                    rng, rand_rng = jax.random.split(rng)
                    mj_model, rollout_variables = jax.lax.cond(
                        transition.done,
                        lambda: reset_mujoco_model(mj_model, rollout_variables, rand_rng),
                        lambda: (mj_model, rollout_variables),
                    )

                    def render_callback(model: mujoco.MjModel, data: mujoco.MjData, scene: mujoco.MjvScene) -> None:
                        for marker in markers:
                            marker(model, data, scene, transition)

                    # Logs the frames to render.
                    viewer.data = rollout_variables.physics_state.data
                    if save_path is None:
                        viewer.render(callback=render_callback)
                    else:
                        frames.append(viewer.read_pixels(depth=False, callback=render_callback))

                trajectory = jax.tree_map(lambda *xs: jnp.stack(xs), *transitions)

                get_rewards(
                    trajectory=trajectory,
                    reward_generators=rollout_constants.reward_generators,
                    ctrl_dt=self.config.ctrl_dt,
                    clip_max=self.config.reward_clip_max,
                )

                # TODO: some nice visualizer of rewards...

            except (KeyboardInterrupt, bdb.BdbQuit):
                logger.info("Keyboard interrupt, exiting environment loop")

            if save_path is not None:
                fps = round(1 / self.config.ctrl_dt)

                match save_path.suffix.lower():
                    case ".mp4":
                        try:
                            import imageio.v2 as imageio

                        except ImportError:
                            raise RuntimeError(
                                "Failed to save video - note that saving .mp4 videos with imageio usually "
                                "requires the FFMPEG backend, which can be installed using `pip install "
                                "'imageio[ffmpeg]'`. Note that this also requires FFMPEG to be installed in "
                                "your system."
                            )

                        try:
                            with imageio.get_writer(save_path, mode="I", fps=fps) as writer:
                                for frame in frames:
                                    writer.append_data(frame)  # type: ignore[attr-defined]

                        except Exception as e:
                            raise RuntimeError(
                                "Failed to save video - note that saving .mp4 videos with imageio usually "
                                "requires the FFMPEG backend, which can be installed using `pip install "
                                "'imageio[ffmpeg]'`. Note that this also requires FFMPEG to be installed in "
                                "your system."
                            ) from e

                    case ".gif":
                        images = [Image.fromarray(frame) for frame in frames]
                        images[0].save(
                            save_path,
                            save_all=True,
                            append_images=images[1:],
                            duration=int(1000 / fps),
                            loop=0,
                        )

                    case _:
                        raise ValueError(f"Unsupported file extension: {save_path.suffix}. Expected .mp4 or .gif")

    def log_full_trajectory(self, state: xax.State, is_first_step: bool, last_log_time: float) -> bool:
        if is_first_step and self.config.log_full_trajectory_on_first_step:
            return True

        if (n_steps := self.config.log_full_trajectory_every_n_steps) is not None and state.num_steps % n_steps == 0:
            return True

        if (n_secs := self.config.log_full_trajectory_every_n_seconds) is not None:
            elapsed = time.time() - last_log_time
            if elapsed > n_secs:
                return True

        return False

    def run_training(self) -> None:
        """Wraps the training loop and provides clean XAX integration."""
        with self:
            rng = self.prng_key()
            self.set_loggers()

            if xax.is_master():
                Thread(target=self.log_state, daemon=True).start()

            rng, model_rng = jax.random.split(rng)
            model, optimizer, opt_state, state = self.load_initial_state(model_rng, load_optimizer=True)

            mj_model: PhysicsModel = self.get_mujoco_model()
            mjx_model = self.get_mjx_model(mj_model)
            metadata = self.get_mujoco_model_metadata(mjx_model)
            engine = self.get_engine(mjx_model, metadata)
            observations = self.get_observations(mjx_model)
            commands = self.get_commands(mjx_model)
            rewards_terms = self.get_rewards(mjx_model)
            terminations = self.get_terminations(mjx_model)
            randomizations = self.get_randomization(mjx_model)

            # Creates the renderer.
            mj_renderer = mujoco.Renderer(
                mj_model,
                height=self.config.render_height,
                width=self.config.render_width,
            )
            configure_scene(
                mj_renderer._scene,
                mj_renderer._scene_option,
                shadow=self.config.render_shadow,
                contact_force=self.config.render_contact_force,
                contact_point=self.config.render_contact_point,
                inertia=self.config.render_inertia,
            )

            # Creates the markers.
            markers = self.get_markers(
                commands=commands,
                observations=observations,
                randomizations=randomizations,
            )

            # These remain constant across the entire episode.
            rollout_constants = RolloutConstants(
                obs_generators=tuple(observations),
                command_generators=tuple(commands),
                reward_generators=tuple(rewards_terms),
                termination_generators=tuple(terminations),
            )

            # JAX requires that we partition the model into mutable and static
            # parts in order to use lax.scan, so that `arr` can be a PyTree.`
            model_arr, model_static = eqx.partition(model, self.model_partition_fn)

            # Builds the variables for the training environment.
            train_rngs = jax.random.split(rng, self.config.num_envs)

            # Applies randomizations to the physics model.
            randomization_fn = jax.vmap(apply_randomizations, in_axes=(None, None, None, 0))
            randomization_dict, physics_state = randomization_fn(mjx_model, engine, randomizations, train_rngs)

            # Defines the vectorized initialization functions.
            carry_fn = jax.vmap(self.get_initial_carry, in_axes=0)
            command_fn = jax.vmap(get_initial_commands, in_axes=(0, None))

            rollout_variables = RolloutVariables(
                carry=carry_fn(train_rngs),
                commands=command_fn(train_rngs, rollout_constants.command_generators),
                physics_state=physics_state,
                rng=train_rngs,
            )

            state = self.on_training_start(state)

            def on_exit() -> None:
                model = eqx.combine(model_arr, model_static)
                self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    state=state,
                )

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

            is_first_step = True
            last_log_time = time.time()

            try:
                while not self.is_training_over(state):
                    state = self.on_step_start(state)

                    # Using validation phase to log full trajectories.
                    if self.log_full_trajectory(state, is_first_step, last_log_time):
                        state = state.replace(phase="valid")
                        last_log_time = time.time()
                    else:
                        state = state.replace(phase="train")

                    # Runs the training loop.
                    rng, update_rng = jax.random.split(rng)
                    with xax.ContextTimer() as timer:
                        (
                            model_arr,
                            opt_state,
                            metrics,
                            rollout_variables,
                            final_trajectory,
                            final_reward,
                        ) = self._rl_train_loop_step(
                            physics_model=mjx_model,
                            randomization_dict=randomization_dict,
                            model_arr=model_arr,
                            model_static=model_static,
                            optimizer=optimizer,
                            opt_state=opt_state,
                            engine=engine,
                            rollout_constants=rollout_constants,
                            rollout_variables=rollout_variables,
                            rng=update_rng,
                        )

                    if is_first_step:
                        is_first_step = False
                        elapsed_time = xax.format_timedelta(datetime.timedelta(seconds=timer.elapsed_time), short=True)
                        logger.log(xax.LOG_STATUS, "First step time: %s", elapsed_time)

                    state = state.replace(
                        num_steps=state.num_steps + self.config.epochs_per_log_step,
                        num_samples=state.num_samples + self.rollout_num_samples,
                    )

                    # Only log trajectory information on validation steps.
                    if state.phase == "valid":
                        self.log_single_trajectory(
                            trajectories=final_trajectory,
                            reward_values=final_reward,
                            markers=markers,
                            mj_model=mj_model,
                            mj_renderer=mj_renderer,
                        )

                    self.log_train_metrics(metrics)
                    self.log_state_timers(state)
                    self.write_logs(state)

                    if self.should_checkpoint(state):
                        model = eqx.combine(model_arr, model_static)
                        self.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            opt_state=opt_state,
                            state=state,
                        )

                    state = self.on_step_end(state)

            except xax.TrainingFinishedError:
                if xax.is_master():
                    msg = f"Finished training after {state.num_steps}steps and {state.num_samples} samples"
                    xax.show_info(msg, important=True)

                model = eqx.combine(model_arr, model_static)
                self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    state=state,
                )

            except (KeyboardInterrupt, bdb.BdbQuit):
                if xax.is_master():
                    xax.show_info("Interrupted training", important=True)

            except BaseException:
                exception_tb = textwrap.indent(xax.highlight_exception_message(traceback.format_exc()), "  ")
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()

                model = eqx.combine(model_arr, model_static)
                self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    state=state,
                )

            finally:
                state = self.on_training_end(state)
