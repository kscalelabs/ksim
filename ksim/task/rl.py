"""Defines a standard task interface for training reinforcement learning agents."""

__all__ = [
    "RLConfig",
    "RLTask",
]

import bdb
import io
import itertools
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
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree
from kmv.viewer import launch_passive
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx
from omegaconf import II, MISSING
from PIL import Image, ImageDraw

from ksim.actuators import Actuators
from ksim.commands import Command
from ksim.engine import (
    EngineConstants,
    EngineVariables,
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
from ksim.types import Histogram, Metrics, PhysicsModel, PhysicsState, Rewards, Trajectory
from ksim.utils.mujoco import get_ctrl_data_idx_by_name, get_joint_metadata

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


def compute_step_rewards(
    transition: Trajectory,
    reward_generators: Collection[Reward],
    ctrl_dt: float,
    clip_max: float | None = None,
) -> Rewards:
    """Get the rewards for a single step/transition.

    Similar to get_rewards but designed for a single step rather than a full trajectory.

    Args:
        transition: A single-step trajectory
        reward_generators: Collection of reward generator functions
        ctrl_dt: Control timestep
        clip_max: Optional maximum value to clip rewards

    Returns:
        Rewards object with total reward and components
    """
    # Add batch dimension to the transition - reward functions expect this
    batched_transition = jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), transition)

    rewards = {}
    for reward_generator in reward_generators:
        # Compute reward with batch dimension
        reward_val = reward_generator(batched_transition) * reward_generator.scale * ctrl_dt
        # Remove batch dimension for the result
        reward_val = jnp.squeeze(reward_val, axis=0)
        if clip_max is not None:
            reward_val = jnp.clip(reward_val, -clip_max, clip_max)
        rewards[reward_generator.reward_name] = reward_val
    total_reward = jax.tree.reduce(jnp.add, list(rewards.values()))
    return Rewards(total=total_reward, components=FrozenDict(rewards))


@xax.jit(static_argnames=["reward_generators", "ctrl_dt"])
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
    return Rewards(total=total_reward, components=FrozenDict(rewards))


def get_terminations(
    physics_state: PhysicsState,
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

    # Validation parameters.
    valid_every_n_steps: int | None = xax.field(
        None,
        help="Number of training steps to run per validation step",
    )
    valid_first_n_steps: int = xax.field(
        0,
        help="Treat the first N steps as validation steps",
    )
    valid_every_n_seconds: float | None = xax.field(
        60.0,
        help="Run validation every N seconds",
    )
    valid_first_n_seconds: float | None = xax.field(
        60.0,
        help="Run first validation after N seconds",
    )
    log_single_traj_every_n_valid_steps: int = xax.field(
        value=10,
        help="The number of valid steps between logging a full trajectory video.",
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
    num_valid_envs: int = xax.field(
        value=1,
        help="The number of validation environments to run in parallel.",
    )
    num_batches: int = xax.field(
        value=1,
        help="The number of model update batches per trajectory batch. ",
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
        value=(8, 4),
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
    reward_clip_max: float | None = xax.field(
        value=None,
        help="The maximum value of the reward.",
    )


Config = TypeVar("Config", bound=RLConfig)


class RLTask(xax.Task[Config], Generic[Config], ABC):
    """Base class for reinforcement learning tasks."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        if self.config.num_envs % self.config.num_batches != 0:
            raise ValueError(
                f"The number of environments ({self.config.num_envs}) must be divisible by "
                f"the number of model update batches ({self.config.num_batches})"
            )

    @property
    def batch_size(self) -> int:
        return self.config.num_envs // self.config.num_batches

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
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
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
    def eval_rollout_length_steps(self) -> int:
        return round(self.config.eval_rollout_length_seconds / self.config.ctrl_dt)

    def step_engine(
        self,
        physics_model: PhysicsModel,
        model: PyTree,
        engine: PhysicsEngine,
        engine_constants: EngineConstants,
        engine_variables: EngineVariables,
    ) -> tuple[Trajectory, EngineVariables]:
        """Runs a single step of the physics engine.

        Args:
            physics_model: The physics model.
            model: The model, with parameters to be updated.
            engine: The physics engine.
            engine_constants: The constants for the engine.
            engine_variables: The variables for the engine.

        Returns:
            A tuple containing the trajectory and the next engine variables.
        """
        rng, obs_rng, cmd_rng, act_rng, reset_rng, carry_rng, physics_rng = jax.random.split(engine_variables.rng, 7)

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
            lambda: get_initial_commands(cmd_rng, command_generators=engine_constants.command_generators),
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
        next_variables = EngineVariables(
            carry=next_carry,
            commands=commands,
            physics_state=next_physics_state,
            rng=rng,
        )

        return trajectory, next_variables

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
        commands: Collection[Command],
        mj_model: mujoco.MjModel,
    ) -> tuple[np.ndarray, int]:
        """Render trajectory as video frames with computed FPS."""
        fps = round(1 / self.config.ctrl_dt)

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

        renderer = mujoco.Renderer(
            mj_model,
            height=self.config.render_height,
            width=self.config.render_width,
        )

        frame_list: list[np.ndarray] = []
        for frame_id, trajectory in enumerate(trajectory_list):
            mj_data.qpos = np.array(trajectory.qpos)
            mj_data.qvel = np.array(trajectory.qvel)

            # Renders the current frame.
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=mj_camera)

            # Adds command elements to the scene.
            for command in commands:
                command.update_scene(renderer.scene, trajectory.command[command.command_name])

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
        trajectories: Trajectory,
        commands: Collection[Command],
        rewards: Rewards,
        mj_model: mujoco.MjModel,
    ) -> None:
        """Visualizes a single trajectory.

        Args:
            trajectories: The trajectories to visualize.
            commands: The commands to visualize.
            rewards: The rewards to visualize.
            mj_model: The Mujoco model to render the scene with.
            name: The name of the trajectory being logged.
        """
        # Logs plots of the observations, commands, actions, rewards, and terminations.
        # Emojis are used in order to prevent conflicts with user-specified namespaces.
        for namespace, arr_dict in (
            ("👀 obs images", trajectories.obs),
            ("🕹️ command images", trajectories.command),
            ("🏃 action images", {"action": trajectories.action}),
            ("💀 termination images", trajectories.termination_components),
            ("🎁 reward images", rewards.components),
            ("🎁 reward images", {"total": rewards.total}),
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
        frames, fps = self.render_trajectory_video(trajectories, commands, mj_model)
        self.logger.log_video(key="trajectory", value=frames, fps=fps, namespace="➡️ trajectory images")

    @xax.jit(static_argnames=["self", "model_static", "engine", "engine_constants", "num_steps"])
    def _single_unroll(
        self,
        physics_model: PhysicsModel,
        model_arr: PyTree,
        model_static: PyTree,
        engine: PhysicsEngine,
        engine_constants: EngineConstants,
        engine_variables: EngineVariables,
        num_steps: int,
    ) -> tuple[Trajectory, Rewards, EngineVariables]:
        # Recombines the mutable and static parts of the model.
        model = eqx.combine(model_arr, model_static)

        def scan_fn(carry: EngineVariables, _: None) -> tuple[EngineVariables, Trajectory]:
            trajectory, next_engine_variables = self.step_engine(
                physics_model=physics_model,
                model=model,
                engine=engine,
                engine_constants=engine_constants,
                engine_variables=carry,
            )
            return next_engine_variables, trajectory

        # Scans the engine for the desired number of steps.
        next_engine_variables, trajectory = jax.lax.scan(scan_fn, engine_variables, length=num_steps)

        # Gets the rewards.
        reward = get_rewards(
            trajectory=trajectory,
            reward_generators=engine_constants.reward_generators,
            ctrl_dt=self.config.ctrl_dt,
            clip_max=self.config.reward_clip_max,
        )

        return trajectory, reward, next_engine_variables

    @xax.jit(static_argnames=["self", "model_static", "engine", "engine_constants", "num_steps"])
    def _vmapped_unroll(
        self,
        physics_models: PhysicsModel,
        model_arr: PyTree,
        model_static: PyTree,
        engine: PhysicsEngine,
        engine_constants: EngineConstants,
        engine_variables: EngineVariables,
        num_steps: int,
    ) -> tuple[Trajectory, Rewards, EngineVariables]:
        vmapped_unroll = jax.vmap(self._single_unroll, in_axes=(0, None, None, None, None, 0, None))
        return vmapped_unroll(
            physics_models,
            model_arr,
            model_static,
            engine,
            engine_constants,
            engine_variables,
            num_steps,
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
    ) -> tuple[PyTree, optax.OptState, FrozenDict[str, Array]]:
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
        num_terms = trajectories.done.sum(-1, dtype=trajectories.action.dtype) + 1
        traj_lens = (trajectories.done.shape[-1] / num_terms) * self.config.ctrl_dt

        return {
            "episode_length": traj_lens.mean(),
            **{key: (value / num_terms[:, None]).mean() for key, value in trajectories.termination_components.items()},
        }

    @xax.jit(static_argnames=["self", "model_static", "optimizer", "engine", "engine_constants"])
    def _rl_train_loop_step(
        self,
        rng: PRNGKeyArray,
        physics_models: PhysicsModel,
        model_arr: PyTree,
        model_static: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        engine: PhysicsEngine,
        engine_constants: EngineConstants,
        engine_variables: EngineVariables,
    ) -> tuple[PyTree, optax.OptState, Metrics, EngineVariables]:
        def single_step_fn(
            carry: tuple[PyTree, optax.OptState, EngineVariables],
            rng: PRNGKeyArray,
        ) -> tuple[tuple[PyTree, optax.OptState, EngineVariables], Metrics]:
            model_arr, opt_state, engine_variables = carry
            rng, update_rng = jax.random.split(rng)

            # Rolls out a new trajectory.
            trajectories, rewards, engine_variables = self._vmapped_unroll(
                physics_models=physics_models,
                model_arr=model_arr,
                model_static=model_static,
                engine=engine,
                engine_constants=engine_constants,
                engine_variables=engine_variables,
                num_steps=self.rollout_length_steps,
            )

            # Runs update on the previous trajectory.
            model_arr, opt_state, train_metrics = self.update_model(
                model_arr=model_arr,
                model_static=model_static,
                optimizer=optimizer,
                opt_state=opt_state,
                trajectories=trajectories,
                rewards=rewards,
                rng=update_rng,
            )

            metrics = Metrics(
                train=train_metrics,
                reward=FrozenDict(self.get_reward_metrics(trajectories, rewards)),
                termination=FrozenDict(self.get_termination_metrics(trajectories)),
            )

            return (model_arr, opt_state, engine_variables), metrics

        ((model_arr, opt_state, engine_variables), metrics) = jax.lax.scan(
            single_step_fn,
            (model_arr, opt_state, engine_variables),
            jax.random.split(rng, self.config.epochs_per_log_step),
        )

        # Convert any array with more than one element to a histogram.
        metrics = jax.tree.map(lambda x: self.get_histogram(x) if isinstance(x, Array) and x.size > 1 else x, metrics)

        return model_arr, opt_state, metrics, engine_variables

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

        try:
            from pynput import keyboard

        except ImportError:
            raise ImportError("pynput is not installed. Please install it using 'pip install pynput'.")

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
            rng, carry_rng, cmd_rng = jax.random.split(rng, 3)
            initial_carry = self.get_initial_carry(carry_rng)
            initial_commands = get_initial_commands(cmd_rng, command_generators=commands)

            # Resets the physics state.
            rng, reset_rng = jax.random.split(rng)
            physics_state = engine.reset(mj_model, reset_rng)

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

            paused = False

            def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
                nonlocal paused
                if key == keyboard.Key.space:
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == keyboard.Key.esc:
                    # Stop listener using a different approach
                    listener.stop()

            # Start the listener in a separate thread.
            listener = keyboard.Listener(on_press=on_press)
            listener.start()

            try:
                viewer_model = mj_model
                viewer_data = physics_state.data
                with launch_passive(
                    viewer_model,
                    viewer_data,
                    save_path=save_path,
                    render_width=self.config.render_width,
                    render_height=self.config.render_height,
                    ctrl_dt=self.config.ctrl_dt,
                    make_plots=True,
                ) as viewer:
                    viewer.setup_camera(
                        render_distance=self.config.render_distance,
                        render_azimuth=self.config.render_azimuth,
                        render_elevation=self.config.render_elevation,
                        render_lookat=self.config.render_lookat,
                    )

                    ctrl_idx_to_name = {v: k for k, v in get_ctrl_data_idx_by_name(mj_model).items()}
                    viewer.add_plot_group(title="Actions", index_mapping=ctrl_idx_to_name, y_axis_min=-2, y_axis_max=2)

                    # We'll set up reward component plots in the first iteration
                    reward_component_mapping = None

                    for step_id in iterator:
                        # We need to manually sync the data back and forth between
                        # the viewer and the engine, because the resetting the
                        # environment creates a new data object rather than
                        # happening in-place, as Mujoco expects.
                        viewer.copy_data(dst=engine_variables.physics_state.data, src=viewer_data)

                        # If paused, only update the viewer
                        if paused:
                            viewer.update_and_sync()
                            time.sleep(0.005)
                            continue

                        transition, engine_variables = self.step_engine(
                            physics_model=mj_model,
                            model=model,
                            engine=engine,
                            engine_constants=engine_constants,
                            engine_variables=engine_variables,
                        )

                        step_rewards = compute_step_rewards(
                            transition=transition,
                            reward_generators=rewards,
                            ctrl_dt=self.config.ctrl_dt,
                            clip_max=self.config.reward_clip_max,
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

                        # Sync data again
                        viewer.copy_data(dst=viewer_data, src=engine_variables.physics_state.data)
                        mujoco.mj_forward(viewer_model, viewer_data)
                        viewer.add_commands(dict(engine_variables.commands))
                        viewer.update_and_sync()

                        # Create rewards group on first step
                        if step_id == 0:
                            # First element is the total reward
                            # Remaining elements are the individual reward components
                            component_names = list(step_rewards.components.keys())
                            reward_component_mapping = {0: "total_reward"}
                            for i, name in enumerate(component_names):
                                reward_component_mapping[i + 1] = name
                            viewer.add_plot_group(title="Reward Components", index_mapping=reward_component_mapping)

                        # Update reward components
                        reward_values = [float(step_rewards.total)]
                        reward_values.extend([float(val) for val in step_rewards.components.values()])
                        viewer.update_plot_group("Reward Components", reward_values)

                        # Update actions
                        # Convert actions to a flat list of floats
                        action_values = np.array(engine_variables.physics_state.most_recent_action, dtype=float)
                        action_values_flat = [float(x) for x in action_values.flatten().tolist()]
                        viewer.update_plot_group("Actions", action_values_flat)
            except (KeyboardInterrupt, bdb.BdbQuit):
                logger.info("Keyboard interrupt, exiting environment loop")

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

            # These remain constant across the entire episode.
            engine_constants = EngineConstants(
                obs_generators=tuple(observations),
                command_generators=tuple(commands),
                reward_generators=tuple(rewards_terms),
                termination_generators=tuple(terminations),
                randomization_generators=tuple(randomizations),
            )

            # JAX requires that we partition the model into mutable and static
            # parts in order to use lax.scan, so that `arr` can be a PyTree.`
            model_arr, model_static = eqx.partition(model, eqx.is_inexact_array)

            # Defines the vectorized initialization functions.
            randomization_fn = jax.vmap(apply_randomizations, in_axes=(None, None, 0))
            carry_fn = jax.vmap(self.get_initial_carry, in_axes=0)
            command_fn = jax.vmap(get_initial_commands, in_axes=(0, None))
            state_fn = jax.vmap(engine.reset, in_axes=(0, 0))

            # Builds the variables for the training environment.
            train_rngs = jax.random.split(rng, self.config.num_envs)
            mjx_models = randomization_fn(mjx_model, engine_constants.randomization_generators, train_rngs)
            engine_variables = EngineVariables(
                carry=carry_fn(train_rngs),
                commands=command_fn(train_rngs, engine_constants.command_generators),
                physics_state=state_fn(mjx_models, train_rngs),
                rng=train_rngs,
            )

            # Builds the variables for the validation environment.
            valid_rngs = jax.random.split(rng, self.config.num_valid_envs)
            valid_mjx_models = randomization_fn(mjx_model, engine_constants.randomization_generators, valid_rngs)
            valid_engine_variables = EngineVariables(
                carry=carry_fn(valid_rngs),
                commands=command_fn(valid_rngs, engine_constants.command_generators),
                physics_state=state_fn(valid_mjx_models, valid_rngs),
                rng=valid_rngs,
            )

            state = self.on_training_start(state)
            state.num_samples = 1  # prevents from checkpointing at start

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

            try:
                while not self.is_training_over(state):
                    # Validate by sampling and visualizing a single trajectory.
                    if self.valid_step_timer.is_valid_step(state):
                        state.raw_phase = "valid"
                        state.num_valid_steps += 1
                        state.num_valid_samples += self.eval_rollout_length_steps

                        # Rolls out a new trajectory.
                        trajectories, rewards, valid_engine_variables = self._vmapped_unroll(
                            physics_models=valid_mjx_models,
                            model_arr=model_arr,
                            model_static=model_static,
                            engine=engine,
                            engine_constants=engine_constants,
                            num_steps=self.eval_rollout_length_steps,
                            engine_variables=valid_engine_variables,
                        )

                        # Logs statistics from the trajectory.
                        trajectory = jax.tree.map(lambda arr: arr[0], trajectories)
                        reward = jax.tree.map(lambda arr: arr[0], rewards)

                        if state.num_valid_steps % self.config.log_single_traj_every_n_valid_steps == 0:
                            self.log_single_trajectory(trajectory, commands, reward, mj_model)
                        self.log_state_timers(state)
                        self.write_logs(state)

                    state = self.on_step_start(state)

                    # Optimizes the model on that trajectory.
                    rng, update_rng = jax.random.split(rng)
                    model_arr, opt_state, metrics, engine_variables = self._rl_train_loop_step(
                        rng=update_rng,
                        physics_models=mjx_models,
                        model_arr=model_arr,
                        model_static=model_static,
                        optimizer=optimizer,
                        opt_state=opt_state,
                        engine=engine,
                        engine_constants=engine_constants,
                        engine_variables=engine_variables,
                    )

                    state.raw_phase = "train"
                    state.num_steps += self.config.epochs_per_log_step
                    state.num_samples += (
                        self.rollout_length_steps * self.config.num_envs * self.config.epochs_per_log_step
                    )

                    self.log_train_metrics(metrics)
                    self.log_state_timers(state)
                    self.write_logs(state)

                    state = self.on_step_end(state)

                    if self.should_checkpoint(state):
                        model = eqx.combine(model_arr, model_static)
                        self.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            opt_state=opt_state,
                            state=state,
                        )

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
