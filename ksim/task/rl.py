"""Defines a standard task interface for training reinforcement learning agents."""

__all__ = [
    "RLConfig",
    "RLTask",
    "RolloutConstants",
    "RolloutSharedState",
    "RolloutEnvState",
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
from collections import Counter, deque
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Thread
from types import FrameType
from typing import Any, Callable, Collection, Generic, TypeVar

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import optax
import tqdm
import xax
from dpshdl.dataset import Dataset
from jax.core import get_aval
from jaxtyping import Array, PRNGKeyArray, PyTree
from kmv.app.viewer import DefaultMujocoViewer, QtViewer
from kmv.core.types import RenderMode
from mujoco import mjx
from omegaconf import MISSING
from PIL import Image, ImageDraw

from ksim.actuators import Actuators
from ksim.commands import Command
from ksim.curriculum import Curriculum, CurriculumState
from ksim.dataset import TrajectoryDataset
from ksim.debugging import JitLevel
from ksim.engine import PhysicsEngine, engine_type_from_physics_model, get_physics_engine
from ksim.events import Event
from ksim.observation import Observation, ObservationInput, StatefulObservation
from ksim.randomization import PhysicsRandomizer
from ksim.resets import Reset
from ksim.rewards import Reward, StatefulReward
from ksim.terminations import Termination
from ksim.types import (
    Action,
    Histogram,
    LoggedTrajectory,
    Metadata,
    Metrics,
    PhysicsData,
    PhysicsModel,
    PhysicsState,
    RewardState,
    Trajectory,
)
from ksim.utils.mujoco import (
    get_joint_names_in_order,
    get_position_limits,
    get_torque_limits,
    load_model,
    log_joint_config_table,
)
from ksim.vis import Marker, configure_scene

logger = logging.getLogger(__name__)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RolloutEnvState:
    """Per-environment variables for the rollout loop."""

    commands: xax.FrozenDict[str, Array]
    physics_state: PhysicsState
    randomization_dict: xax.FrozenDict[str, Array]
    model_carry: PyTree
    reward_carry: xax.FrozenDict[str, Array]
    obs_carry: xax.FrozenDict[str, PyTree]
    curriculum_state: CurriculumState
    rng: PRNGKeyArray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RolloutSharedState:
    """Variables used across all environments."""

    physics_model: PhysicsModel
    model_arrs: tuple[PyTree, ...]
    aux_values: xax.FrozenDict[str, PyTree]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RolloutConstants:
    """Constants for the rollout loop."""

    model_statics: tuple[PyTree, ...]
    engine: PhysicsEngine
    observations: Collection[Observation]
    commands: Collection[Command]
    rewards: Collection[Reward]
    terminations: Collection[Termination]
    curriculum: Curriculum
    argmax_action: bool
    aux_constants: xax.FrozenDict[str, PyTree]


def get_observation(
    rollout_env_state: RolloutEnvState,
    observations: Collection[Observation],
    obs_carry: PyTree,
    curriculum_level: Array,
    rng: PRNGKeyArray,
) -> tuple[xax.FrozenDict[str, Array], xax.FrozenDict[str, PyTree]]:
    """Get the observation and carry from the physics state."""
    observation_dict: dict[str, Array] = {}
    next_obs_carry: dict[str, PyTree] = {}
    for observation in observations:
        rng, obs_rng, noise_rng = jax.random.split(rng, 3)
        observation_state = ObservationInput(
            commands=rollout_env_state.commands,
            physics_state=rollout_env_state.physics_state,
            obs_carry=obs_carry[observation.observation_name],
        )

        # Calls the observation function.
        if isinstance(observation, StatefulObservation):
            observation_val, new_carry = observation.observe_stateful(observation_state, curriculum_level, obs_rng)
        else:
            observation_val = observation.observe(observation_state, curriculum_level, obs_rng)
            new_carry = observation_state.obs_carry
        observation_val = observation.add_noise(observation_val, curriculum_level, noise_rng)

        observation_dict[observation.observation_name] = observation_val
        next_obs_carry[observation.observation_name] = new_carry
        rng = jax.random.split(rng)[1]

    return xax.FrozenDict(observation_dict), xax.FrozenDict(next_obs_carry)


def get_rewards(
    trajectory: Trajectory,
    rewards: Collection[Reward],
    rewards_carry: xax.FrozenDict[str, PyTree],
    curriculum_level: Array,
    rng: PRNGKeyArray,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> RewardState:
    """Get the rewards from the physics state."""
    reward_dict: dict[str, Array] = {}
    next_reward_carry: dict[str, PyTree] = {}
    target_shape = trajectory.done.shape

    for reward in rewards:
        reward_name = reward.reward_name
        reward_carry = rewards_carry[reward_name]

        if isinstance(reward, StatefulReward):
            reward_carry = jax.tree.map(
                lambda new, old: jnp.where(trajectory.done[..., -1], new, old),
                reward.initial_carry(rng),
                reward_carry,
            )
            reward_val, reward_carry = reward.get_reward_stateful(trajectory, reward_carry)
        else:
            reward_val = reward.get_reward(trajectory)
        reward_val = reward_val * reward.scale
        if reward.scale_by_curriculum:
            reward_val = reward_val * curriculum_level

        if reward_val.shape != trajectory.done.shape:
            raise AssertionError(f"Reward {reward_name} shape {reward_val.shape} does not match {target_shape}")

        reward_dict[reward_name] = reward_val
        next_reward_carry[reward_name] = reward_carry

    total_reward = jax.tree.reduce(jnp.add, list(reward_dict.values()))
    if clip_min is not None:
        total_reward = jnp.maximum(total_reward, clip_min)
    if clip_max is not None:
        total_reward = jnp.minimum(total_reward, clip_max)

    return RewardState(
        total=total_reward,
        components=xax.FrozenDict(reward_dict),
        carry=xax.FrozenDict(next_reward_carry),
    )


def get_initial_obs_carry(
    rng: PRNGKeyArray,
    physics_state: PhysicsState,
    observations: Collection[Observation],
) -> xax.FrozenDict[str, PyTree]:
    """Get the initial observation carry."""
    rngs = jax.random.split(rng, len(observations))
    return xax.FrozenDict(
        {
            obs.observation_name: (
                obs.initial_carry(physics_state, rng) if isinstance(obs, StatefulObservation) else None
            )
            for obs, rng in zip(observations, rngs, strict=True)
        }
    )


def get_initial_reward_carry(
    rng: PRNGKeyArray,
    rewards: Collection[Reward],
) -> xax.FrozenDict[str, PyTree]:
    """Get the initial reward carry."""
    rngs = jax.random.split(rng, len(rewards))
    return xax.FrozenDict(
        {
            reward.reward_name: reward.initial_carry(rng) if isinstance(reward, StatefulReward) else None
            for reward, rng in zip(rewards, rngs, strict=True)
        }
    )


def get_terminations(
    physics_state: PhysicsState,
    terminations: Collection[Termination],
    curriculum_level: Array,
) -> xax.FrozenDict[str, Array]:
    """Get the terminations from the physics state."""
    termination_dict = {}
    for termination in terminations:
        termination_val = termination(physics_state.data, curriculum_level)
        chex.assert_type(termination_val, int)
        name = termination.termination_name
        termination_dict[name] = termination_val
    return xax.FrozenDict(termination_dict)


def get_commands(
    prev_commands: xax.FrozenDict[str, Array],
    physics_state: PhysicsState,
    rng: PRNGKeyArray,
    commands: Collection[Command],
    curriculum_level: Array,
) -> xax.FrozenDict[str, Array]:
    """Get the commands from the physics state."""
    command_dict = {}
    for command_generator in commands:
        rng, cmd_rng = jax.random.split(rng)
        command_name = command_generator.command_name
        prev_command = prev_commands[command_name]
        assert isinstance(prev_command, Array)
        command_val = command_generator(prev_command, physics_state.data, curriculum_level, cmd_rng)
        command_dict[command_name] = command_val
    return xax.FrozenDict(command_dict)


def get_initial_commands(
    rng: PRNGKeyArray,
    physics_data: PhysicsData,
    commands: Collection[Command],
    curriculum_level: Array,
) -> xax.FrozenDict[str, Array]:
    """Get the initial commands from the physics state."""
    command_dict = {}
    for command_generator in commands:
        rng, cmd_rng = jax.random.split(rng)
        command_name = command_generator.command_name
        command_val = command_generator.initial_command(physics_data, curriculum_level, cmd_rng)
        command_dict[command_name] = command_val
    return xax.FrozenDict(command_dict)


def get_physics_randomizers(
    physics_model: PhysicsModel,
    randomizers: Collection[PhysicsRandomizer],
    rng: PRNGKeyArray,
) -> xax.FrozenDict[str, Array]:
    all_randomizations: dict[str, dict[str, Array]] = {}
    for randomizer in randomizers:
        rng, randomization_rng = jax.random.split(rng)
        all_randomizations[randomizer.randomization_name] = randomizer(physics_model, randomization_rng)
    for name, count in Counter([k for d in all_randomizations.values() for k in d.keys()]).items():
        if count > 1:
            name_to_keys = {k: set(v.keys()) for k, v in all_randomizations.items()}
            raise ValueError(f"Found duplicate randomization keys: {name}. Randomizations: {name_to_keys}")
    return xax.FrozenDict({k: v for d in all_randomizations.values() for k, v in d.items()})


def apply_randomizations(
    physics_model: PhysicsModel,
    engine: PhysicsEngine,
    randomizers: Collection[PhysicsRandomizer],
    curriculum_level: Array,
    rng: PRNGKeyArray,
) -> tuple[xax.FrozenDict[str, Array], PhysicsState]:
    rand_rng, reset_rng = jax.random.split(rng)
    randomizations = get_physics_randomizers(physics_model, randomizers, rand_rng)

    # Applies the randomizations to the model.
    if isinstance(physics_model, mjx.Model):
        physics_model = physics_model.tree_replace(randomizations)
    elif isinstance(physics_model, mujoco.MjModel):
        for k, v in randomizations.items():
            setattr(physics_model, k, v)
    else:
        raise ValueError(f"Unknown physics model type: {type(physics_model)}")

    physics_state = engine.reset(physics_model, curriculum_level, reset_rng)
    return randomizations, physics_state


@jax.tree_util.register_dataclass
@dataclass
class RLConfig(xax.Config):
    # Toggle this to run the environment viewer loop.
    run_mode: str = xax.field(
        value="train",
        help="Mode to run the task in - either 'train' or 'view'",
    )
    viewer_argmax_action: bool = xax.field(
        value=True,
        help="If set, take the argmax action instead of sampling from the action distribution.",
    )
    viewer_num_seconds: float | None = xax.field(
        value=None,
        help="If provided, run the environment loop for the given number of seconds.",
    )
    viewer_save_renders: bool = xax.field(
        value=False,
        help="If set, save the renders to the experiment directory.",
    )
    viewer_save_video: bool = xax.field(
        value=False,
        help="If set, render the environment as a video instead of a GIF.",
    )

    # Toggle this to collect a dataset.
    collect_dataset: bool = xax.field(
        value=False,
        help="If true, collect a dataset.",
    )
    dataset_num_batches: int = xax.field(
        1,
        help="The number of trajectories to collect at a time.",
    )
    dataset_save_path: str | None = xax.field(
        value=None,
        help="If provided, save the dataset to the given path.",
    )
    collect_dataset_argmax_action: bool = xax.field(
        value=False,
        help="If set, get the argmax action, otherwise sample randomly from the model.",
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
    profile_memory: bool = xax.field(
        value=False,
        help="If true, profile memory usage.",
    )
    exclude_combined_reward_components: list[str] = xax.field(
        value=[],
        help="If provided, exclude these components from the combined reward plot.",
    )
    # Training parameters.
    num_envs: int = xax.field(
        value=MISSING,
        help="The number of training environments to run in parallel.",
    )
    batch_size: int = xax.field(
        value=1,
        help="The number of trajectories to process in each minibatch during gradient updates.",
    )
    rollout_length_seconds: float = xax.field(
        value=MISSING,
        help="The number of seconds to rollout each environment during training.",
    )

    # Validation timing parameters.
    valid_every_n_seconds: float | None = xax.field(
        value=150.0,
        help="Run full validation (render trajectory and all graphs) every N seconds",
    )
    valid_first_n_seconds: float | None = xax.field(
        value=None,
        help="Run first validation after N seconds",
    )
    render_full_every_n_seconds: float = xax.field(
        value=60.0 * 30.0,
        help="Render the trajectory (without associated graphs) every N seconds",
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
    render_with_glfw: bool | None = xax.field(
        value=None,
        help="Explicitly toggle GLFW rendering; if not specified, use GLFW when rendering on-screen",
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
        value=480,
        help="The height of the rendered images during the validation phase.",
    )
    render_width: int = xax.field(
        value=640,
        help="The height of the rendered images during the validation phase.",
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
        value=1.0,
        help="The slowdown factor for the rendered video.",
    )
    render_track_body_id: int | None = xax.field(
        value=None,
        help="If set, the render camera will track the body with this ID.",
    )
    render_distance: float = xax.field(
        value=3.5,
        help="The distance of the camera from the target.",
    )
    render_azimuth: float = xax.field(
        value=90.0,
        help="The azimuth of the render camera.",
    )
    render_elevation: float = xax.field(
        value=-10.0,
        help="The elevation of the render camera.",
    )
    render_lookat: tuple[float, float, float] = xax.field(
        value=[0.0, 0.0, 0.5],
        help="The lookat point of the render camera.",
    )

    # Engine parameters.
    ctrl_dt: float = xax.field(
        value=0.02,
        help="The time step of the control loop.",
    )
    dt: float = xax.field(
        value=0.002,
        help="The time step of the physics loop.",
    )
    tolerance: float = xax.field(
        value=1e-10,
        help="The tolerance of the solver.",
    )
    iterations: int = xax.field(
        value=MISSING,
        help="Number of main solver iterations",
    )
    ls_iterations: int = xax.field(
        value=MISSING,
        help="Maximum number of CG / Newton linesearch iterations",
    )
    solver: str = xax.field(
        value="newton",
        help="The constraint solver algorithm to use",
    )
    integrator: str = xax.field(
        value="implicitfast",
        help="The integrator algorithm to use",
    )
    disable_euler_damping: bool = xax.field(
        value=True,
        help="If set, disable Euler damping - this is a performance improvement",
    )
    action_latency_range: tuple[float, float] = xax.field(
        value=(0.0, 0.0),
        help="The range of action latencies to use.",
    )
    actuator_update_dt: float | None = xax.field(
        value=None,
        help="The time step of the actuator update.",
    )
    drop_action_prob: float = xax.field(
        value=0.0,
        help="The probability of dropping an action.",
    )
    reward_clip_min: float | None = xax.field(
        value=None,
        help="The minimum value of the reward.",
    )
    reward_clip_max: float | None = xax.field(
        value=None,
        help="The maximum value of the reward.",
    )
    render_markers: bool = xax.field(
        value=False,
        help="If true, render markers.",
    )
    render_camera_name: str | int | None = xax.field(
        value=None,
        help="The name or id of the camera to use in rendering.",
    )
    live_reward_buffer_size: int = xax.field(
        value=4,
        help="Size of the rolling buffer for computing live rewards",
    )
    viewer_timeout_secs: float = xax.field(
        value=10.0,
        help="The timeout for the QT viewer.",
    )


Config = TypeVar("Config", bound=RLConfig)


def get_qt_viewer(
    *,
    mj_model: mujoco.MjModel,
    config: Config,
    mj_data: mujoco.MjData | None = None,
    save_path: str | Path | None = None,
    mode: RenderMode | None = None,
) -> QtViewer:
    return QtViewer(
        mj_model,
        mode=mode if mode is not None else "window" if save_path is None else "offscreen",
        width=config.render_width,
        height=config.render_height,
        shadow=config.render_shadow,
        reflection=config.render_reflection,
        contact_force=config.render_contact_force,
        contact_point=config.render_contact_point,
        inertia=config.render_inertia,
        camera_distance=config.render_distance,
        camera_azimuth=config.render_azimuth,
        camera_elevation=config.render_elevation,
        camera_lookat=config.render_lookat,
        track_body_id=config.render_track_body_id,
        timeout_secs=config.viewer_timeout_secs,
    )


def get_default_viewer(
    *,
    mj_model: mujoco.MjModel,
    config: Config,
    width: int | None = None,
    height: int | None = None,
) -> DefaultMujocoViewer:
    viewer = DefaultMujocoViewer(
        mj_model,
        width=width or config.render_width,
        height=height or config.render_height,
    )

    viewer.cam.distance = config.render_distance
    viewer.cam.azimuth = config.render_azimuth
    viewer.cam.elevation = config.render_elevation
    viewer.cam.lookat[:] = config.render_lookat
    if config.render_track_body_id is not None:
        viewer.cam.trackbodyid = config.render_track_body_id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    if config.render_camera_name is not None:
        viewer.set_camera(config.render_camera_name)

    configure_scene(
        viewer.scn,
        viewer.vopt,
        shadow=config.render_shadow,
        contact_force=config.render_contact_force,
        contact_point=config.render_contact_point,
        inertia=config.render_inertia,
    )

    return viewer


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RLLoopConstants:
    optimizer: tuple[optax.GradientTransformation, ...]
    constants: RolloutConstants


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RLLoopCarry:
    opt_state: tuple[optax.OptState, ...]
    env_states: RolloutEnvState
    shared_state: RolloutSharedState


class RLTask(xax.Task[Config], Generic[Config], ABC):
    """Base class for reinforcement learning tasks."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._is_running = True

        # Using this Matplotlib backend since it is non-interactive.
        matplotlib.use("agg")

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

    def set_mujoco_model_opts(self, mj_model: mujoco.MjModel) -> mujoco.MjModel:
        def _set_opt(name: str, value: Any) -> None:  # noqa: ANN401
            model_val = getattr(mj_model.opt, name)
            if model_val != value:
                logger.debug("User-specified %s %s is different from model %s %s", name, value, name, model_val)
                setattr(mj_model.opt, name, value)

        solver = getattr(mjx.SolverType, self.config.solver.upper(), None)
        if solver is None:
            raise ValueError(f"Invalid solver type: {self.config.solver}")

        integrator = getattr(mjx.IntegratorType, self.config.integrator.upper(), None)
        if integrator is None:
            raise ValueError(f"Invalid integrator type: {self.config.integrator}")

        _set_opt("timestep", self.config.dt)
        _set_opt("iterations", self.config.iterations)
        _set_opt("ls_iterations", self.config.ls_iterations)
        _set_opt("integrator", integrator)
        _set_opt("tolerance", self.config.tolerance)
        _set_opt("solver", solver)

        if self.config.disable_euler_damping:
            mj_model.opt.disableflags = mj_model.opt.disableflags | mjx.DisableBit.EULERDAMP

        return mj_model

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> Metadata:
        return Metadata.from_model(mj_model)

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
        metadata: Metadata | None = None,
    ) -> PhysicsEngine:
        return get_physics_engine(
            engine_type=engine_type_from_physics_model(physics_model),
            resets=self.get_resets(physics_model),
            events=self.get_events(physics_model),
            actuators=self.get_actuators(physics_model, metadata),
            dt=float(physics_model.opt.timestep),
            ctrl_dt=self.config.ctrl_dt,
            action_latency_range=self.config.action_latency_range,
            drop_action_prob=self.config.drop_action_prob,
            actuator_update_dt=self.config.actuator_update_dt,
        )

    @abstractmethod
    def get_physics_randomizers(self, physics_model: PhysicsModel) -> Collection[PhysicsRandomizer]:
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
        metadata: Metadata | None = None,
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
    def get_initial_model_carry(self, rng: PRNGKeyArray) -> PyTree | None:
        """Returns the initial carry for the model.

        Args:
            rng: The random key to use.

        Returns:
            An arbitrary PyTree, representing any carry parameters that the
            model needs.
        """

    @abstractmethod
    def get_curriculum(self, physics_model: PhysicsModel) -> Curriculum:
        """Returns the curriculum for the current task.

        Args:
            physics_model: The physics model to get the curriculum for.
        """

    @abstractmethod
    def sample_action(
        self,
        model: PyTree,
        model_carry: PyTree,
        physics_model: PhysicsModel,
        physics_state: PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> Action:
        """Gets an action for the current observation.

        This function returns the action to take, the next carry (for models
        which look at multiple steps), and any auxiliary outputs. The auxiliary
        outputs get stored in the final trajectory object and can be used to
        compute metrics like log probabilities, values, etc.

        Args:
            model: The current model.
            physics_model: The physics model.
            physics_state: The current physics state.
            observations: The current observations.
            commands: The current commands.
            model_carry: The model carry from the previous step.
            rng: The random key.
            argmax: If set, get the argmax action, otherwise sample randomly
                from the model.

        Returns:
            The action to take, the next carry, and any auxiliary outputs.
        """

    @property
    def rollout_length_steps(self) -> int:
        return round(self.config.rollout_length_seconds / self.config.ctrl_dt)

    @property
    def rollout_num_samples(self) -> int:
        return self.rollout_length_steps * self.config.num_envs

    def get_mujoco_model_info(self, mj_model: mujoco.MjModel) -> dict:
        return {
            "joint_names": get_joint_names_in_order(mj_model),
            "position_limits": get_position_limits(mj_model),
            "torque_limits": get_torque_limits(mj_model),
        }

    @xax.jit(static_argnames=["self", "constants"], jit_level=JitLevel.ENGINE)
    def step_engine(
        self,
        constants: RolloutConstants,
        env_states: RolloutEnvState,
        shared_state: RolloutSharedState,
    ) -> tuple[Trajectory, RolloutEnvState]:
        """Runs a single step of the physics engine.

        Args:
            constants: The constants for the engine.
            env_states: The environment variables for the engine.
            shared_state: The control variables for the engine.

        Returns:
            A tuple containing the trajectory and the next engine variables.
        """
        rng = env_states.rng
        rng, obs_rng, cmd_rng, act_rng, reset_rng, carry_rng, physics_rng = jax.random.split(rng, 7)

        # Recombines the mutable and static parts of the model.
        policy_model_arr = shared_state.model_arrs[0]
        policy_model_static = constants.model_statics[0]
        policy_model = eqx.combine(policy_model_arr, policy_model_static)

        # Gets the observations from the physics state.
        observations, next_obs_carry = get_observation(
            rollout_env_state=env_states,
            observations=constants.observations,
            obs_carry=env_states.obs_carry,
            curriculum_level=env_states.curriculum_state.level,
            rng=obs_rng,
        )

        # Samples an action from the model.
        action = self.sample_action(
            model=policy_model,
            model_carry=env_states.model_carry,
            physics_model=shared_state.physics_model,
            physics_state=env_states.physics_state,
            observations=observations,
            commands=env_states.commands,
            rng=act_rng,
            argmax=constants.argmax_action,
        )

        # Steps the physics engine.
        next_physics_state: PhysicsState = constants.engine.step(
            action=action.action,
            physics_model=shared_state.physics_model,
            physics_state=env_states.physics_state,
            curriculum_level=env_states.curriculum_state.level,
            rng=physics_rng,
        )

        # Gets termination components and a single termination boolean.
        terminations = get_terminations(
            physics_state=next_physics_state,
            terminations=constants.terminations,
            curriculum_level=env_states.curriculum_state.level,
        )

        # Convert ternary terminations to binary arrays.
        terminated = jax.tree.reduce(jnp.logical_or, [t != 0 for t in terminations.values()])
        success = jax.tree.reduce(jnp.logical_and, [t != -1 for t in terminations.values()]) & terminated

        # Combines all the relevant data into a single object. Lives up here to
        # avoid accidentally incorporating information it shouldn't access to.
        transition = Trajectory(
            qpos=jnp.array(next_physics_state.data.qpos),
            qvel=jnp.array(next_physics_state.data.qvel),
            xpos=jnp.array(next_physics_state.data.xpos),
            xquat=jnp.array(next_physics_state.data.xquat),
            ctrl=jnp.array(next_physics_state.data.ctrl),
            obs=observations,
            command=env_states.commands,
            event_state=next_physics_state.event_states,
            action=action.action,
            done=terminated,
            success=success,
            timestep=next_physics_state.data.time,
            termination_components=terminations,
            aux_outputs=action.aux_outputs,
        )

        next_physics_state = jax.lax.cond(
            terminated,
            lambda: constants.engine.reset(
                shared_state.physics_model,
                env_states.curriculum_state.level,
                reset_rng,
            ),
            lambda: next_physics_state,
        )

        # Conditionally reset on termination.
        next_commands = jax.lax.cond(
            terminated,
            lambda: get_initial_commands(
                rng=cmd_rng,
                physics_data=next_physics_state.data,
                commands=constants.commands,
                curriculum_level=env_states.curriculum_state.level,
            ),
            lambda: get_commands(
                prev_commands=env_states.commands,
                physics_state=next_physics_state,
                rng=cmd_rng,
                commands=constants.commands,
                curriculum_level=env_states.curriculum_state.level,
            ),
        )

        next_obs_carry = jax.lax.cond(
            terminated,
            lambda: get_initial_obs_carry(
                rng=carry_rng, physics_state=next_physics_state, observations=constants.observations
            ),
            lambda: next_obs_carry,
        )

        next_model_carry = jax.lax.cond(
            terminated,
            lambda: self.get_initial_model_carry(carry_rng),
            lambda: action.carry,
        )

        # Gets the variables for the next step.
        next_env_state = replace(
            env_states,
            commands=next_commands,
            physics_state=next_physics_state,
            model_carry=next_model_carry,
            obs_carry=next_obs_carry,
            rng=rng,
        )

        return transition, next_env_state

    def get_dataset(self, phase: xax.Phase) -> Dataset:
        raise NotImplementedError("RL tasks do not require datasets, since trajectory histories are stored in-memory.")

    def compute_loss(self, model: PyTree, batch: Any, output: Any, state: xax.State) -> Array:  # noqa: ANN401
        raise NotImplementedError(
            "Direct compute_loss from TrainMixin is not expected to be called in RL tasks. "
            "PPO tasks use model_update and loss_metrics_grads instead."
        )

    def run(self) -> None:
        """Highest level entry point for RL tasks, determines what to run."""
        match self.config.run_mode.lower():
            case "train":
                self.run_training()

            case "view":
                self.run_model_viewer(
                    num_steps=(
                        None
                        if self.config.viewer_num_seconds is None
                        else round(self.config.viewer_num_seconds / self.config.ctrl_dt)
                    ),
                    save_renders=self.config.viewer_save_renders,
                    argmax_action=self.config.viewer_argmax_action,
                )

            case "collect_dataset":
                self.collect_dataset(
                    num_batches=self.config.dataset_num_batches,
                    save_path=self.config.dataset_save_path,
                    argmax_action=self.config.collect_dataset_argmax_action,
                )

            case _:
                raise ValueError(f"Invalid run mode: {self.config.run_mode}")

    def log_train_metrics(self, metrics: Metrics) -> None:
        """Logs the train metrics.

        Args:
            metrics: The metrics to log.
            rollout_length: The length of the rollout.
        """
        if self.config.log_train_metrics:
            for namespace, metric, secondary in (
                ("🚂 train", metrics.train, True),
                ("🎁 reward", metrics.reward, False),
                ("💀 termination", metrics.termination, True),
                ("🔄 curriculum", {"level": metrics.curriculum_level}, True),
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
                        self.logger.log_scalar(key, value.mean, namespace=namespace, secondary=secondary)
                    else:
                        self.logger.log_scalar(key, value.mean(), namespace=namespace, secondary=secondary)

    def render_trajectory_video(
        self,
        trajectory: Trajectory,
        markers: Collection[Marker],
        viewer: DefaultMujocoViewer,
        target_fps: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """Render trajectory as video frames with computed FPS."""
        fps = round(1 / self.config.ctrl_dt)

        # Rerenders the trajectory at the desired FPS.
        num_frames = len(trajectory.done)
        if target_fps is not None:
            indices = jnp.arange(0, num_frames, fps / target_fps, dtype=jnp.int32).clip(max=num_frames - 1)
            trajectory = jax.tree.map(lambda arr: arr[indices], trajectory)
            fps = target_fps
        else:
            indices = jnp.arange(0, num_frames, dtype=jnp.int32)

        chex.assert_shape(trajectory.done, (None,))
        num_steps = trajectory.done.shape[0]

        trajectory_list: list[Trajectory] = [
            jax.tree.map(lambda arr, i=i: arr[i], trajectory) for i in range(num_steps)
        ]

        frame_list: list[np.ndarray] = []

        for frame_id, sub_trajectory in enumerate(trajectory_list):
            # Updates the model with the latest data.
            viewer.data.qpos[:] = np.array(sub_trajectory.qpos)
            viewer.data.qvel[:] = np.array(sub_trajectory.qvel)
            mujoco.mj_forward(viewer.model, viewer.data)

            def render_callback(
                model: mujoco.MjModel,
                data: mujoco.MjData,
                scene: mujoco.MjvScene,
                traj: Trajectory = sub_trajectory,
            ) -> None:
                if self.config.render_markers:
                    for marker in markers:
                        marker(model, data, scene, traj)

            frame = viewer.read_pixels(callback=render_callback)

            # Overlays the frame number on the frame.
            frame_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(frame_img)

            text = f"Frame {indices[frame_id]}"
            bbox = draw.textbbox((0, 0), text)
            draw.rectangle([8, 8, 12 + bbox[2] - bbox[0], 12 + bbox[3] - bbox[1]], fill="white")
            draw.text((10, 10), text, fill="black")
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

    def _crop_to_length(self, logged_traj: LoggedTrajectory, length: float) -> LoggedTrajectory:
        render_frames = round(length / self.config.ctrl_dt)
        return jax.tree.map(lambda arr: arr[:render_frames] if arr.ndim > 0 else arr, logged_traj)

    def _log_logged_trajectory_graphs(
        self,
        logged_traj: LoggedTrajectory,
        log_callback: Callable[[str, Image.Image, str], None],
    ) -> None:
        """Visualizes a single trajectory.

        Args:
            logged_traj: The single trajectory to log.
            log_callback: A callable function to run to log a given image.
        """
        # Clips the trajectory to the desired length.
        if self.config.render_length_seconds is not None:
            logged_traj = self._crop_to_length(logged_traj, self.config.render_length_seconds)

        def create_plot_image(
            fig_size: tuple[float, float], plot_fn: Callable[[plt.Figure, plt.Axes], None]
        ) -> Image.Image:
            """Create a plot image using the provided plotting function."""
            plt.figure(figsize=fig_size)
            ax = plt.gca()
            plot_fn(plt.gcf(), ax)

            # Convert to PIL image
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            return Image.open(buf)

        # Logs plots of the observations, commands, actions, rewards, and terminations.
        # Emojis are used in order to prevent conflicts with user-specified namespaces.
        for namespace, arr_dict in (
            ("👀 obs images", logged_traj.trajectory.obs),
            ("🕹️ command images", logged_traj.trajectory.command),
            ("🏃 action images", {"action": logged_traj.trajectory.action}),
            ("💀 termination images", logged_traj.trajectory.termination_components),
            ("🗓️ event images", logged_traj.trajectory.event_state),
            ("🎁 reward images", logged_traj.rewards.components),
            ("🎁 reward images", {"total": logged_traj.rewards.total}),
            ("📈 metrics images", logged_traj.metrics),
        ):
            for key, value in arr_dict.items():

                def plot_individual_component(
                    fig: plt.Figure, ax: plt.Axes, key: str = key, value: Array = value
                ) -> None:
                    # Ensures a consistent shape and truncates if necessary.
                    processed_value = value.reshape(value.shape[0], -1)
                    if processed_value.shape[-1] > self.config.max_values_per_plot:
                        logger.debug("Truncating %s to %d values per plot.", key, self.config.max_values_per_plot)
                        processed_value = processed_value[..., : self.config.max_values_per_plot]

                    for i in range(processed_value.shape[1]):
                        ax.plot(processed_value[:, i], label=f"{i}")

                    if processed_value.shape[1] > 1:
                        ax.legend()
                    ax.set_title(key)

                # Create and log the image
                img = create_plot_image(self.config.plot_figsize, plot_individual_component)
                log_callback(key, img, namespace)

        # Add a combined plot with all reward components for easy comparison
        def plot_combined_rewards(fig: plt.Figure, ax: plt.Axes) -> None:
            for key, value in logged_traj.rewards.components.items():
                if key in self.config.exclude_combined_reward_components:
                    continue
                processed_value = value.reshape(value.shape[0], -1)
                if processed_value.shape[1] > 1:
                    processed_value = processed_value.mean(axis=1)

                ax.plot(processed_value, label=key, alpha=0.8)

            ax.set_title("All Rewards")
            ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5))
            fig.tight_layout()
            # Add extra space to the right for the legend
            fig.subplots_adjust(right=0.7)

        # Make the plot wider so the legend fits
        combined_rewards_figsize = (self.config.plot_figsize[0] * 1.3, self.config.plot_figsize[1])
        img = create_plot_image(combined_rewards_figsize, plot_combined_rewards)
        log_callback("all_components_comparison", img, "🎁 reward images")

    def _log_logged_trajectory_video(
        self,
        logged_traj: LoggedTrajectory,
        markers: Collection[Marker],
        viewer: DefaultMujocoViewer,
        key: str,
    ) -> None:
        """Visualizes a single trajectory.

        Args:
            logged_traj: The single trajectory to log.
            markers: The markers to visualize.
            viewer: The Mujoco viewer to render the scene with.
            key: The logging key to use.
        """
        # Clips the trajectory to the desired length.
        if self.config.render_length_seconds is not None:
            logged_traj = self._crop_to_length(logged_traj, self.config.render_length_seconds)

        # Logs the video of the trajectory.
        frames, fps = self.render_trajectory_video(
            trajectory=logged_traj.trajectory,
            markers=markers,
            viewer=viewer,
            target_fps=self.config.render_fps,
        )

        self.logger.log_video(
            key=key,
            value=frames,
            fps=round(fps / self.config.render_slowdown),
            namespace=f"➡️ {key} images",
        )

    @abstractmethod
    def update_model(
        self,
        *,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        trajectories: Trajectory,
        rewards: RewardState,
        rng: PRNGKeyArray,
    ) -> tuple[
        RLLoopCarry,
        xax.FrozenDict[str, Array],
        LoggedTrajectory,
    ]:
        """Updates the model on the given trajectory.

        This function should be implemented according to the specific RL method
        that we are using.

        Args:
            constants: The constants to use for the rollout.
            carry: The carry to use for the rollout.
            trajectories: The trajectories to update the model on.
            rewards: The rewards to update the model on.
            rng: The random seed.

        Returns:
            A tuple containing the next carry, metrics to log, and the single
            trajectory to log. If a metric has a single element it is logged as
            a scalar, otherwise it is logged as a histogram.
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

    def get_reward_metrics(self, trajectories: Trajectory, rewards: RewardState) -> dict[str, Array]:
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
        # Compute the mean number of terminations per episode, broken down by
        # the type of termination.
        kvs = list(trajectories.termination_components.items())
        all_terminations = jnp.stack([v for _, v in kvs], axis=-1)
        has_termination = (all_terminations.any(axis=-1)).sum(axis=-1)
        num_terminations = has_termination.sum().clip(min=1)
        mean_terminations = trajectories.done.sum(-1).mean()

        return {
            "episode_length": trajectories.episode_length(),
            "mean_terminations": mean_terminations,
            **{f"prct/{key}": ((value != 0).sum() / num_terminations) for key, value in kvs},
        }

    def get_markers(
        self,
        commands: Collection[Command],
        observations: Collection[Observation],
        rewards: Collection[Reward],
    ) -> Collection[Marker]:
        markers: list[Marker] = []
        for command in commands:
            markers.extend(command.get_markers())
        for observation in observations:
            markers.extend(observation.get_markers())
        for reward in rewards:
            markers.extend(reward.get_markers())
        return markers

    def postprocess_trajectory(
        self,
        constants: RolloutConstants,
        env_states: RolloutEnvState,
        shared_state: RolloutSharedState,
        trajectory: Trajectory,
        rng: PRNGKeyArray,
    ) -> Trajectory:
        return trajectory

    @xax.jit(static_argnames=["self", "constants"], jit_level=JitLevel.UNROLL)
    def _single_unroll(
        self,
        constants: RolloutConstants,
        env_state: RolloutEnvState,
        shared_state: RolloutSharedState,
    ) -> tuple[Trajectory, RewardState, RolloutEnvState]:
        # Applies randomizations to the model.
        shared_state = replace(
            shared_state,
            physics_model=shared_state.physics_model.tree_replace(env_state.randomization_dict),
        )

        def scan_fn(env_state: RolloutEnvState, _: None) -> tuple[RolloutEnvState, Trajectory]:
            trajectory, env_state = self.step_engine(
                constants=constants,
                env_states=env_state,
                shared_state=shared_state,
            )
            return env_state, trajectory

        # Scans the engine for the desired number of steps.
        env_state, trajectory = xax.scan(
            scan_fn,
            env_state,
            length=self.rollout_length_steps,
            jit_level=JitLevel.UNROLL,
        )

        rng, reward_rng, postprocess_rng = jax.random.split(env_state.rng, 3)

        # Post-processes the trajectory.
        trajectory = self.postprocess_trajectory(
            constants=constants,
            env_states=env_state,
            shared_state=shared_state,
            trajectory=trajectory,
            rng=postprocess_rng,
        )

        # Gets the rewards.
        reward = get_rewards(
            trajectory=trajectory,
            rewards=constants.rewards,
            rewards_carry=env_state.reward_carry,
            curriculum_level=env_state.curriculum_state.level,
            rng=reward_rng,
            clip_min=self.config.reward_clip_min,
            clip_max=self.config.reward_clip_max,
        )

        # Updates the reward carry in the environment state.
        env_state = replace(
            env_state,
            reward_carry=reward.carry,
            rng=rng,
        )

        return trajectory, reward, env_state

    @xax.jit(static_argnames=["self", "constants"], jit_level=JitLevel.OUTER_LOOP)
    def _rl_train_loop_step(
        self,
        carry: RLLoopCarry,
        constants: RLLoopConstants,
        state: xax.State,
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopCarry, Metrics, LoggedTrajectory]:
        """Runs a single step of the RL training loop."""

        def single_step_fn(
            carry_i: RLLoopCarry,
            rng: PRNGKeyArray,
        ) -> tuple[RLLoopCarry, tuple[Metrics, LoggedTrajectory]]:
            # Rolls out a new trajectory.
            vmapped_unroll = xax.vmap(
                self._single_unroll,
                in_axes=(None, 0, None),
                jit_level=JitLevel.UNROLL,
            )
            trajectories, rewards, env_state = vmapped_unroll(
                constants.constants,
                carry_i.env_states,
                carry_i.shared_state,
            )

            # Runs update on the previous trajectory.
            carry_i, train_metrics, logged_traj = self.update_model(
                constants=constants,
                carry=carry_i,
                trajectories=trajectories,
                rewards=rewards,
                rng=rng,
            )

            # Store all the metrics to log.
            metrics = Metrics(
                train=train_metrics,
                reward=xax.FrozenDict(self.get_reward_metrics(trajectories, rewards)),
                termination=xax.FrozenDict(self.get_termination_metrics(trajectories)),
                curriculum_level=carry_i.env_states.curriculum_state.level,
            )

            # Steps the curriculum.
            curriculum_state = constants.constants.curriculum(
                trajectory=trajectories,
                rewards=rewards,
                training_state=state,
                prev_state=carry_i.env_states.curriculum_state,
            )

            # Update the environment states *after* doing the model update -
            # the model needs to be updated using the same environment states
            # that were used to generate the trajectory.
            carry_i = replace(
                carry_i,
                env_states=replace(
                    env_state,
                    curriculum_state=curriculum_state,
                ),
            )

            return carry_i, (metrics, logged_traj)

        rngs = jax.random.split(rng, self.config.epochs_per_log_step)
        carry, (metrics, logged_traj) = xax.scan(single_step_fn, carry, rngs, jit_level=JitLevel.OUTER_LOOP)

        # Convert any array with more than one element to a histogram.
        metrics = jax.tree.map(self._histogram_fn, metrics)

        # Only get final trajectory and rewards.
        logged_traj = jax.tree.map(lambda arr: arr[-1], logged_traj)

        return carry, metrics, logged_traj

    @xax.jit(static_argnums=(0,), jit_level=JitLevel.HELPER_FUNCTIONS)
    def _histogram_fn(self, x: Any) -> Any:  # noqa: ANN401
        if isinstance(x, Array) and x.size > 1:
            return self.get_histogram(x)
        return x

    def run_environment_step(
        self,
        constants: RolloutConstants,
        env_states: RolloutEnvState,
        shared_state: RolloutSharedState,
    ) -> tuple[Array, Array, tuple[Trajectory, RolloutEnvState] | None]:
        """Runs a single step of the environment.

        Args:
            constants: The constants
            env_states: The environment states
            shared_state: The shared state

        Returns:
            A tuple containing the qpos, qvel, and optionally the
            transition and env_states.
        """
        transition, env_states = self.step_engine(
            constants=constants,
            env_states=env_states,
            shared_state=shared_state,
        )

        qpos = env_states.physics_state.data.qpos
        qvel = env_states.physics_state.data.qvel

        return qpos, qvel, (transition, env_states)

    def run_model_viewer(
        self,
        num_steps: int | None = None,
        save_renders: bool = False,
        argmax_action: bool = True,
    ) -> None:
        """Provides an easy-to-use interface for debugging environments.

        This function runs the environment for `num_steps`, rendering using
        MujocoViewer while simultaneously plotting the reward and termination
        information.

        Args:
            num_steps: The number of steps to run the environment for. If not
                provided, run until the user manually terminates the
                environment visualizer.
            save_renders: If provided, save the rendered video to the given path.
            argmax_action: If set, get the argmax action, otherwise sample
                randomly from the model.
        """
        save_path = self.exp_dir / "renders" / f"render_{time.monotonic()}" if save_renders else None
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)

        with self, jax.disable_jit():
            rng = self.prng_key()
            self.set_loggers()

            # Loads the Mujoco model and logs some information about it.
            mj_model = self.get_mujoco_model()
            mj_model = self.set_mujoco_model_opts(mj_model)
            metadata = self.get_mujoco_model_metadata(mj_model)
            log_joint_config_table(mj_model, metadata, self.logger)

            randomizers = self.get_physics_randomizers(mj_model)

            rng, model_rng = jax.random.split(rng)
            models, _ = self.load_initial_state(model_rng, load_optimizer=False)

            # Partitions the models into mutable and static parts.
            model_arrs, model_statics = (
                tuple(models)
                for models in zip(
                    *(eqx.partition(model, self.model_partition_fn) for model in models),
                    strict=True,
                )
            )

            constants = self._get_constants(
                mj_model=mj_model,
                physics_model=mj_model,
                model_statics=model_statics,
                argmax_action=argmax_action,
            )
            env_states = self._get_env_state(
                rng=rng,
                rollout_constants=constants,
                mj_model=mj_model,
                physics_model=mj_model,
                randomizers=randomizers,
            )
            shared_state = self._get_shared_state(
                mj_model=mj_model,
                physics_model=mj_model,
                model_arrs=model_arrs,
            )

            live_reward_transition_buffer: deque[Trajectory] = deque(maxlen=self.config.live_reward_buffer_size)
            viewer_rng = rng

            # Creates the markers.
            markers = self.get_markers(
                commands=constants.commands,
                observations=constants.observations,
                rewards=constants.rewards,
            )

            # Creates the viewer.
            viewer = get_qt_viewer(
                mj_model=mj_model,
                config=self.config,
                mj_data=env_states.physics_state.data,
                save_path=save_path,
            )

            iterator = tqdm.trange(num_steps) if num_steps is not None else tqdm.tqdm(itertools.count())
            frames: list[np.ndarray] = []

            transitions = []

            try:
                for _ in iterator:
                    # Get commands
                    new_commands = self.get_viewer_commands(
                        commands=constants.commands, prev_command_inputs=env_states.commands
                    )
                    env_states = replace(env_states, commands=new_commands)

                    transition, env_states = self.step_engine(
                        constants=constants,
                        env_states=env_states,
                        shared_state=shared_state,
                    )
                    transitions.append(transition)

                    # Build a window of transitions to compute live rewards
                    live_reward_transition_buffer.append(transition)
                    traj_small = jax.tree.map(lambda *xs: jnp.stack(xs), *live_reward_transition_buffer)
                    viewer_rng, traj_rng, step_rng = jax.random.split(viewer_rng, 3)
                    traj_small = self.postprocess_trajectory(
                        constants=constants,
                        env_states=env_states,
                        shared_state=shared_state,
                        trajectory=traj_small,
                        rng=traj_rng,
                    )
                    reward_state = get_rewards(
                        trajectory=traj_small,
                        rewards=constants.rewards,
                        rewards_carry=env_states.reward_carry,
                        curriculum_level=env_states.curriculum_state.level,
                        rng=step_rng,
                        clip_min=self.config.reward_clip_min,
                        clip_max=self.config.reward_clip_max,
                    )
                    env_states = replace(env_states, reward_carry=reward_state.carry)

                    # Send viewer the physics state
                    sim_time = float(env_states.physics_state.data.time)
                    viewer.push_state(
                        np.array(env_states.physics_state.data.qpos),
                        np.array(env_states.physics_state.data.qvel),
                        sim_time=sim_time,
                    )

                    # Send rewards
                    reward_scalars = {
                        "total": float(jax.device_get(reward_state.total[-1])),
                        **{k: float(jax.device_get(v[-1])) for k, v in reward_state.components.items()},
                    }
                    viewer.push_plot_metrics(reward_scalars, group="reward")

                    # Send observations
                    obs_dict = jax.tree_util.tree_map(
                        lambda x: np.asarray(jax.device_get(x)),
                        transition.obs,
                    )
                    for obs_name, obs_value in obs_dict.items():
                        flat_obs = obs_value.reshape(-1)
                        obs_scalars = {f"{obs_name}_{i}": float(v) for i, v in enumerate(flat_obs)}
                        viewer.push_plot_metrics(obs_scalars, group=f"Observations/{obs_name}")

                    # Send physics properties (just first 3 values of qpos for now)
                    qpos_arr = np.asarray(env_states.physics_state.data.qpos)
                    physics_scalars = {f"qpos{i}": float(qpos_arr[i]) for i in range(min(3, qpos_arr.size))}
                    viewer.push_plot_metrics(physics_scalars, group="physics")

                    # Send actions (just 3 for now)
                    ctrl_arr = np.asarray(env_states.physics_state.data.ctrl)
                    action_scalars = {f"act_{i}": float(ctrl_arr[i]) for i in range(min(ctrl_arr.size, 3))}
                    viewer.push_plot_metrics(action_scalars, group="action")

                    # Send commands
                    command_scalars = {}
                    for cmd_name, cmd_val in env_states.commands.items():
                        cmd_arr = np.asarray(jax.device_get(cmd_val))
                        command_scalars.update(
                            {f"{cmd_name}_{i}": float(val) for i, val in enumerate(cmd_arr.flatten())}
                        )
                    viewer.push_plot_metrics(command_scalars, group="command")

                    # Recieve pushes from the viewer
                    xfrc = viewer.drain_control_pipe()
                    if xfrc is not None:
                        env_states.physics_state.data.xfrc_applied[:] = xfrc

                    # TODO: Support markers in kmv
                    def render_callback(
                        model: mujoco.MjModel,
                        data: mujoco.MjData,
                        scene: mujoco.MjvScene,
                        traj: Trajectory = transition,
                    ) -> None:
                        if self.config.render_markers:
                            for marker in markers:
                                marker(model, data, scene, traj)

                    if not viewer.is_open:
                        logger.info("Viewer closed, exiting environment loop")
                        break

            except (KeyboardInterrupt, bdb.BdbQuit):
                logger.info("Keyboard interrupt, exiting environment loop")
            finally:
                viewer.close()

            if len(transitions) == 0:
                logger.warning("Trajectory is empty!")
                return

            rng, postprocess_rng, reward_rng = jax.random.split(rng, 3)

            trajectory = jax.tree.map(lambda *xs: jnp.stack(xs), *transitions)
            trajectory = self.postprocess_trajectory(
                constants=constants,
                env_states=env_states,
                shared_state=shared_state,
                trajectory=trajectory,
                rng=postprocess_rng,
            )

            reward_state = get_rewards(
                trajectory=trajectory,
                rewards=constants.rewards,
                rewards_carry=env_states.reward_carry,
                curriculum_level=env_states.curriculum_state.level,
                rng=reward_rng,
                clip_min=self.config.reward_clip_min,
                clip_max=self.config.reward_clip_max,
            )

            if save_path is not None:
                self._log_logged_trajectory_graphs(
                    logged_traj=LoggedTrajectory(
                        trajectory=trajectory,
                        rewards=reward_state,
                        metrics=xax.FrozenDict({}),
                    ),
                    log_callback=lambda name, value, _: value.save(save_path / f"{name}.png"),
                )

                self._save_viewer_video(frames, save_path)

    def get_viewer_commands(
        self, commands: Collection[Command], prev_command_inputs: xax.FrozenDict[str, Array]
    ) -> xax.FrozenDict[str, Array]:
        """Get the commands when running with run_mode == "view".

        This is a no-op by default, but can be overridden by subclasses to provide
        a custom command for the viewer. E.g. to read keyboard inputs and pass commands
        to the environment based on the inputs.
        """
        return prev_command_inputs

    def _save_viewer_video(self, frames: list[np.ndarray], save_path: Path) -> None:
        fps = round(1 / self.config.ctrl_dt)
        vid_save_path = save_path / ("render.mp4" if self.config.viewer_save_video else "render.gif")

        match vid_save_path.suffix.lower():
            case ".mp4":
                try:
                    import imageio.v2 as imageio  # noqa: PLC0415

                except ImportError as err:
                    raise RuntimeError(
                        "Failed to save video - note that saving .mp4 videos with imageio usually "
                        "requires the FFMPEG backend, which can be installed using `pip install "
                        "'imageio[ffmpeg]'`. Note that this also requires FFMPEG to be installed in "
                        "your system."
                    ) from err

                try:
                    with imageio.get_writer(vid_save_path, mode="I", fps=fps) as writer:
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
                images = [Image.fromarray(frame) for frame in frames]
                images[0].save(
                    vid_save_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=int(1000 / fps),
                    loop=0,
                )

            case _:
                raise ValueError(f"Unsupported file extension: {vid_save_path.suffix}. Expected .mp4 or .gif")

        logger.log(xax.LOG_STATUS, "Rendered trajectory visuals to %s", save_path)

    def _get_constants(
        self,
        *,
        mj_model: mujoco.MjModel,
        physics_model: PhysicsModel,
        model_statics: tuple[PyTree, ...],
        argmax_action: bool,
    ) -> RolloutConstants:
        if len(model_statics) < 1:
            raise ValueError("No models found")

        metadata = self.get_mujoco_model_metadata(mj_model)
        engine = self.get_engine(physics_model, metadata)
        observations = self.get_observations(physics_model)
        commands = self.get_commands(physics_model)
        rewards_terms = self.get_rewards(physics_model)
        if len(rewards_terms) == 0:
            raise ValueError("No rewards found! Must have at least one reward.")
        terminations = self.get_terminations(physics_model)
        if len(terminations) == 0:
            raise ValueError("No terminations found! Must have at least one termination.")
        curriculum = self.get_curriculum(physics_model)

        return RolloutConstants(
            model_statics=model_statics,
            engine=engine,
            observations=tuple(observations),
            commands=tuple(commands),
            rewards=tuple(rewards_terms),
            terminations=tuple(terminations),
            curriculum=curriculum,
            argmax_action=argmax_action,
            aux_constants=xax.FrozenDict({}),
        )

    def _get_shared_state(
        self,
        *,
        mj_model: mujoco.MjModel,
        physics_model: PhysicsModel,
        model_arrs: tuple[PyTree, ...],
    ) -> RolloutSharedState:
        if len(model_arrs) < 1:
            raise ValueError("No models found")

        return RolloutSharedState(
            physics_model=physics_model,
            model_arrs=model_arrs,
            aux_values=xax.FrozenDict({}),
        )

    def _get_env_state(
        self,
        *,
        rng: PRNGKeyArray,
        rollout_constants: RolloutConstants,
        mj_model: mujoco.MjModel,
        physics_model: PhysicsModel,
        randomizers: Collection[PhysicsRandomizer],
    ) -> RolloutEnvState:
        rng, carry_rng, command_rng, rand_rng, rollout_rng, curriculum_rng, reward_rng = jax.random.split(rng, 7)

        if isinstance(physics_model, mjx.Model):
            # Defines the vectorized initialization functions.
            carry_fn = xax.vmap(self.get_initial_model_carry, in_axes=0, jit_level=JitLevel.INITIALIZATION)
            command_fn = xax.vmap(get_initial_commands, in_axes=(0, 0, None, 0), jit_level=JitLevel.INITIALIZATION)
            reward_carry_fn = xax.vmap(get_initial_reward_carry, in_axes=(0, None), jit_level=JitLevel.INITIALIZATION)
            obs_carry_fn = xax.vmap(get_initial_obs_carry, in_axes=(0, None, None), jit_level=JitLevel.INITIALIZATION)

            # Gets the initial curriculum state.
            curriculum_fn = rollout_constants.curriculum.get_initial_state
            curriculum_fn = xax.vmap(curriculum_fn, in_axes=0, jit_level=JitLevel.INITIALIZATION)
            curriculum_state = curriculum_fn(jax.random.split(curriculum_rng, self.config.num_envs))

            # Gets the per-environment randomizations.
            randomization_fn = apply_randomizations
            randomization_fn = xax.vmap(
                randomization_fn,
                in_axes=(None, None, None, 0, 0),
                jit_level=JitLevel.INITIALIZATION,
            )
            randomization_dict, physics_state = randomization_fn(
                physics_model,
                rollout_constants.engine,
                randomizers,
                curriculum_state.level,
                jax.random.split(rand_rng, self.config.num_envs),
            )

            return RolloutEnvState(
                commands=command_fn(
                    jax.random.split(command_rng, self.config.num_envs),
                    physics_state.data,
                    rollout_constants.commands,
                    curriculum_state.level,
                ),
                physics_state=physics_state,
                randomization_dict=randomization_dict,
                model_carry=carry_fn(jax.random.split(carry_rng, self.config.num_envs)),
                reward_carry=reward_carry_fn(
                    jax.random.split(reward_rng, self.config.num_envs), rollout_constants.rewards
                ),
                obs_carry=obs_carry_fn(
                    jax.random.split(carry_rng, self.config.num_envs), physics_state, rollout_constants.observations
                ),
                curriculum_state=curriculum_state,
                rng=jax.random.split(rollout_rng, self.config.num_envs),
            )

        else:
            # Gets the initial curriculum state.
            curriculum_state = rollout_constants.curriculum.get_initial_state(curriculum_rng)

            # Gets the environment randomizations.
            randomization_dict, physics_state = apply_randomizations(
                physics_model,
                rollout_constants.engine,
                randomizers,
                curriculum_state.level,
                rand_rng,
            )

            return RolloutEnvState(
                commands=get_initial_commands(
                    command_rng,
                    physics_state.data,
                    rollout_constants.commands,
                    curriculum_state.level,
                ),
                physics_state=physics_state,
                randomization_dict=randomization_dict,
                model_carry=self.get_initial_model_carry(carry_rng),
                reward_carry=get_initial_reward_carry(reward_rng, rollout_constants.rewards),
                obs_carry=get_initial_obs_carry(
                    rng=carry_rng, physics_state=physics_state, observations=rollout_constants.observations
                ),
                curriculum_state=curriculum_state,
                rng=rollout_rng,
            )

    def collect_dataset(
        self,
        num_batches: int,
        save_path: str | Path | None = None,
        argmax_action: bool = False,
    ) -> None:
        """Collects a dataset of state-action pairs by running the environment loop.

        Args:
            num_batches: The number of batches to collect at a time.
            save_path: Where to save the dataset; if not specified, will save
                to the experimental directory.
            argmax_action: If set, get the argmax action, otherwise sample
                randomly from the model.
        """
        with self:
            rng = self.prng_key()
            self.set_loggers()

            if xax.is_master():
                Thread(target=self.log_state, daemon=True).start()

            # Loads the Mujoco model and logs some information about it.
            mj_model: PhysicsModel = self.get_mujoco_model()
            mj_model = self.set_mujoco_model_opts(mj_model)
            metadata = self.get_mujoco_model_metadata(mj_model)
            log_joint_config_table(mj_model, metadata, self.logger)

            mjx_model = self.get_mjx_model(mj_model)
            randomizations = self.get_physics_randomizers(mjx_model)

            rng, model_rng = jax.random.split(rng)
            models, state = self.load_initial_state(model_rng, load_optimizer=False)

            # Partitions the models into mutable and static parts.
            model_arrs, model_statics = (
                tuple(models)
                for models in zip(
                    *(eqx.partition(model, self.model_partition_fn) for model in models),
                    strict=True,
                )
            )

            if save_path is None:
                save_path = self.exp_dir / f"dataset_{state.num_steps}.npz"

            rollout_constants = self._get_constants(
                mj_model=mj_model,
                physics_model=mjx_model,
                model_statics=model_statics,
                argmax_action=argmax_action,
            )
            rollout_env_state = self._get_env_state(
                rng=rng,
                rollout_constants=rollout_constants,
                mj_model=mj_model,
                physics_model=mjx_model,
                randomizers=randomizations,
            )
            rollout_shared_state = self._get_shared_state(
                mj_model=mj_model,
                physics_model=mjx_model,
                model_arrs=model_arrs,
            )

            state = self.on_training_start(state)

            @xax.jit(jit_level=JitLevel.UNROLL)
            def get_batch(
                rollout_env_state: RolloutEnvState,
            ) -> tuple[Trajectory, RewardState, RolloutEnvState]:
                vmapped_unroll = xax.vmap(
                    self._single_unroll,
                    in_axes=(None, 0, None),
                    jit_level=JitLevel.UNROLL,
                )
                return vmapped_unroll(rollout_constants, rollout_env_state, rollout_shared_state)

            with TrajectoryDataset.writer(save_path, num_batches * self.batch_size) as writer:
                for _ in tqdm.trange(num_batches):
                    trajectories, rewards, rollout_env_state = get_batch(rollout_env_state)

                    # Splits trajectories and rewards into a list of `batch_size` samples.
                    for i in range(0, len(trajectories.done), self.batch_size):
                        trajectory = jax.tree.map(lambda x, index=i: x[index], trajectories)
                        reward = jax.tree.map(lambda x, index=i: x[index], rewards)
                        writer.write(trajectory, reward)

            logger.info("Saved dataset to %s", save_path)

    def initialize_rl_training(
        self,
        mj_model: PhysicsModel,
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopConstants, RLLoopCarry, xax.State]:
        # Gets the model and optimizer variables.
        rng, model_rng = jax.random.split(rng)
        models, optimizers, opt_states, state = self.load_initial_state(model_rng, load_optimizer=True)

        # Logs model and optimizer information.
        for i, (model, opt_state) in enumerate(zip(models, opt_states, strict=True), 1):
            suffix = f" {i}" if len(models) > 1 else ""
            model_size = xax.get_pytree_param_count(model)
            opt_state_size = xax.get_pytree_param_count(opt_state)
            logger.log(xax.LOG_PING, "Model%s size: %s parameters", suffix, f"{model_size:,}")
            logger.log(xax.LOG_PING, "Optimizer%s size: %s parameters", suffix, f"{opt_state_size:,}")

        # Partitions the models into mutable and static parts.
        model_arrs, model_statics = (
            tuple(models)
            for models in zip(
                *(eqx.partition(model, self.model_partition_fn) for model in models),
                strict=True,
            )
        )

        # Loads the MJX model, and initializes the loop variables.
        mjx_model = self.get_mjx_model(mj_model)
        randomizers = self.get_physics_randomizers(mjx_model)

        constants = RLLoopConstants(
            optimizer=tuple(optimizers),
            constants=self._get_constants(
                mj_model=mj_model,
                physics_model=mjx_model,
                model_statics=model_statics,
                argmax_action=False,
            ),
        )

        carry = RLLoopCarry(
            opt_state=tuple(opt_states),
            env_states=self._get_env_state(
                rng=rng,
                rollout_constants=constants.constants,
                mj_model=mj_model,
                physics_model=mjx_model,
                randomizers=randomizers,
            ),
            shared_state=self._get_shared_state(
                mj_model=mj_model,
                physics_model=mjx_model,
                model_arrs=model_arrs,
            ),
        )

        return constants, carry, state

    def _save(
        self,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        state: xax.State,
    ) -> None:
        model = eqx.combine(carry.shared_state.model_arrs, constants.constants.model_statics)
        models = list(model) if isinstance(model, tuple) else [model]
        self.save_checkpoint(models=models, optimizers=constants.optimizer, opt_states=carry.opt_state, state=state)

    def run_training(self) -> None:
        """Wraps the training loop and provides clean XAX integration."""

        def on_exit(signum: int, frame: FrameType | None) -> None:
            if self._is_running:
                self._is_running = False
                if xax.is_master():
                    xax.show_info("Gracefully shutting down...", important=True)

        signal.signal(signal.SIGINT, on_exit)
        signal.signal(signal.SIGTERM, on_exit)

        with self:
            rng = self.prng_key()
            self.set_loggers()
            self._is_running = True

            if xax.is_master():
                Thread(target=self.log_state, daemon=True).start()

            # Loads the Mujoco model and logs some information about it.
            mj_model: PhysicsModel = self.get_mujoco_model()
            mj_model = self.set_mujoco_model_opts(mj_model)
            metadata = self.get_mujoco_model_metadata(mj_model)
            log_joint_config_table(mj_model, metadata, self.logger)

            constants, carry, state = self.initialize_rl_training(mj_model, rng)

            for name, leaf in xax.get_named_leaves(carry, max_depth=3):
                aval = get_aval(leaf)
                if aval.weak_type:
                    logger.warning("Found weak type: '%s' This could slow down compilation time", name)

            # Creates the markers.
            markers = self.get_markers(
                commands=constants.constants.commands,
                observations=constants.constants.observations,
                rewards=constants.constants.rewards,
            )

            # Creates the viewer.
            viewer = get_default_viewer(
                mj_model=mj_model,
                config=self.config,
            )

            state = self.on_training_start(state)

            is_first_step = True
            last_full_render_time = 0.0

            try:
                while self._is_running and not self.is_training_over(state):
                    # Runs the training loop.
                    with xax.ContextTimer() as timer:
                        valid_step = self.valid_step_timer(state)

                        state = state.replace(
                            phase="valid" if valid_step else "train",
                        )

                        state = self.on_step_start(state)

                        rng, update_rng = jax.random.split(rng)

                        carry, metrics, logged_traj = self._rl_train_loop_step(
                            carry=carry,
                            constants=constants,
                            state=state,
                            rng=update_rng,
                        )

                        if self.config.profile_memory:
                            carry = jax.block_until_ready(carry)
                            metrics = jax.block_until_ready(metrics)
                            logged_traj = jax.block_until_ready(logged_traj)
                            jax.profiler.save_device_memory_profile(self.exp_dir / "train_loop_step.prof")

                        self.log_train_metrics(metrics)
                        self.log_state_timers(state)

                        if self.should_checkpoint(state):
                            self._save(constants=constants, carry=carry, state=state)

                        state = self.on_step_end(state)

                        if valid_step:
                            cur_time = time.monotonic()
                            full_render = cur_time - last_full_render_time > self.config.render_full_every_n_seconds
                            self._log_logged_trajectory_video(
                                logged_traj=logged_traj,
                                markers=markers,
                                viewer=viewer,
                                key="full trajectory" if full_render else "trajectory",
                            )
                            if full_render:
                                self._log_logged_trajectory_graphs(
                                    logged_traj=logged_traj,
                                    log_callback=lambda key, value, namespace: self.logger.log_image(
                                        key=key,
                                        value=value,
                                        namespace=namespace,
                                    ),
                                )
                                last_full_render_time = cur_time

                        # Updates the step and sample counts.
                        num_steps = self.config.epochs_per_log_step
                        num_samples = self.rollout_num_samples * self.config.epochs_per_log_step

                        state = state.replace(
                            num_steps=state.num_steps + num_steps,
                            num_samples=state.num_samples + num_samples,
                        )

                        self.write_logs(state)

                    # Update  state with the elapsed time.
                    elapsed_time = timer.elapsed_time
                    state = state.replace(
                        elapsed_time_s=state.elapsed_time_s + elapsed_time,
                    )

                    if is_first_step:
                        is_first_step = False
                        logger.log(
                            xax.LOG_STATUS,
                            "First step time: %s",
                            xax.format_timedelta(datetime.timedelta(seconds=elapsed_time), short=True),
                        )

                # Save the checkpoint when done.
                self._save(constants=constants, carry=carry, state=state)

            except xax.TrainingFinishedError:
                if xax.is_master():
                    msg = f"Finished training after {state.num_steps}steps and {state.num_samples} samples"
                    xax.show_info(msg, important=True)
                self._save(constants=constants, carry=carry, state=state)

            except BaseException:
                exception_tb = textwrap.indent(xax.highlight_exception_message(traceback.format_exc()), "  ")
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()
                self._save(constants=constants, carry=carry, state=state)

            finally:
                state = self.on_training_end(state)
