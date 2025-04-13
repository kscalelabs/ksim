"""Defines a standard task interface for training reinforcement learning agents."""

__all__ = [
    "RLConfig",
    "RLTask",
    "RolloutConstants",
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
import matplotlib
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
from omegaconf import MISSING, DictConfig, OmegaConf
from PIL import Image, ImageDraw

from ksim.actuators import Actuators
from ksim.commands import Command
from ksim.curriculum import Curriculum, CurriculumState
from ksim.dataset import TrajectoryDataset
from ksim.engine import (
    PhysicsEngine,
    engine_type_from_physics_model,
    get_physics_engine,
)
from ksim.events import Event
from ksim.observation import Observation, ObservationState
from ksim.randomization import PhysicsRandomizer
from ksim.resets import Reset
from ksim.rewards import Reward
from ksim.terminations import Termination
from ksim.types import (
    Action,
    Histogram,
    LoggedTrajectory,
    Metrics,
    PhysicsData,
    PhysicsModel,
    PhysicsState,
    Rewards,
    Trajectory,
)
from ksim.utils.mujoco import (
    get_joint_metadata,
    get_joint_names_in_order,
    get_position_limits,
    get_torque_limits,
    load_model,
)
from ksim.viewer import DefaultMujocoViewer, GlfwMujocoViewer, RenderMode
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
    curriculum_state: CurriculumState
    rng: PRNGKeyArray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RolloutSharedState:
    """Variables used across all environments."""

    physics_model: PhysicsModel
    model_arr: PyTree


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RolloutConstants:
    """Constants for the rollout loop."""

    model_static: PyTree
    engine: PhysicsEngine
    observations: Collection[Observation]
    commands: Collection[Command]
    rewards: Collection[Reward]
    terminations: Collection[Termination]
    curriculum: Curriculum
    argmax_action: bool


def get_observation(
    rollout_env_state: RolloutEnvState,
    observations: Collection[Observation],
    curriculum_level: Array,
    rng: PRNGKeyArray,
) -> xax.FrozenDict[str, Array]:
    """Get the observation from the physics state."""
    observation_dict: dict[str, Array] = {}
    observation_state = ObservationState(
        commands=rollout_env_state.commands,
        physics_state=rollout_env_state.physics_state,
    )
    for observation in observations:
        rng, obs_rng = jax.random.split(rng)
        observation_value = observation(observation_state, curriculum_level, obs_rng)
        observation_dict[observation.observation_name] = observation_value
    return xax.FrozenDict(observation_dict)


def get_rewards(
    trajectory: Trajectory,
    rewards: Collection[Reward],
    rewards_carry: xax.FrozenDict[str, PyTree],
    rollout_length_steps: int,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> Rewards:
    """Get the rewards from the physics state."""
    reward_dict: dict[str, Array] = {}
    next_reward_carry: dict[str, PyTree] = {}
    target_shape = trajectory.done.shape

    for reward_generator in rewards:
        reward_name = reward_generator.reward_name
        reward_carry = rewards_carry[reward_name]
        reward_val, reward_carry = reward_generator(trajectory, reward_carry)
        reward_val = reward_val * reward_generator.scale / rollout_length_steps
        if reward_val.shape != trajectory.done.shape:
            raise AssertionError(f"Reward {reward_name} shape {reward_val.shape} does not match {target_shape}")
        reward_dict[reward_name] = reward_val
        next_reward_carry[reward_name] = reward_carry
    total_reward = jax.tree.reduce(jnp.add, list(reward_dict.values()))
    if clip_min is not None:
        total_reward = jnp.maximum(total_reward, clip_min)
    if clip_max is not None:
        total_reward = jnp.minimum(total_reward, clip_max)

    return Rewards(total=total_reward, components=xax.FrozenDict(reward_dict), carry=xax.FrozenDict(next_reward_carry))


def get_initial_reward_carry(
    rng: PRNGKeyArray,
    rewards: Collection[Reward],
) -> xax.FrozenDict[str, Array]:
    """Get the initial reward carry."""
    rngs = jax.random.split(rng, len(rewards))
    return xax.FrozenDict({reward.reward_name: reward.initial_carry(rng) for reward, rng in zip(rewards, rngs)})


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
    run_environment_argmax_action: bool = xax.field(
        value=True,
        help="If set, take the argmax action instead of sampling from the action distribution.",
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

    # Validation timing parameters.
    valid_every_n_seconds: float | None = xax.field(
        150.0,
        help="Run validation every N seconds",
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
        value=320,
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


Config = TypeVar("Config", bound=RLConfig)


def get_viewer(
    mj_model: mujoco.MjModel,
    config: Config,
    mj_data: mujoco.MjData | None = None,
    save_path: str | Path | None = None,
    mode: RenderMode | None = None,
) -> GlfwMujocoViewer | DefaultMujocoViewer:
    if mode is None:
        mode = "window" if save_path is None else "offscreen"

    if (render_with_glfw := config.render_with_glfw) is None:
        render_with_glfw = mode == "window"

    viewer: GlfwMujocoViewer | DefaultMujocoViewer

    if render_with_glfw:
        viewer = GlfwMujocoViewer(
            mj_model,
            data=mj_data,
            mode=mode,
            height=config.render_height,
            width=config.render_width,
            shadow=config.render_shadow,
            reflection=config.render_reflection,
            contact_force=config.render_contact_force,
            contact_point=config.render_contact_point,
            inertia=config.render_inertia,
        )

    else:
        viewer = DefaultMujocoViewer(
            mj_model,
            width=config.render_width,
            height=config.render_height,
        )

    # Sets the viewer camera.
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


class RLTask(xax.Task[Config], Generic[Config], ABC):
    """Base class for reinforcement learning tasks."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

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

    @xax.jit(static_argnames=["self", "rollout_constants"], jit_level=3)
    def step_engine(
        self,
        rollout_constants: RolloutConstants,
        rollout_env_state: RolloutEnvState,
        rollout_shared_state: RolloutSharedState,
    ) -> tuple[Trajectory, RolloutEnvState]:
        """Runs a single step of the physics engine.

        Args:
            physics_model: The physics model.
            rollout_constants: The constants for the engine.
            rollout_env_state: The environment variables for the engine.
            rollout_shared_state: The control variables for the engine.

        Returns:
            A tuple containing the trajectory and the next engine variables.
        """
        rng, obs_rng, cmd_rng, act_rng, reset_rng, carry_rng, physics_rng = jax.random.split(rollout_env_state.rng, 7)

        # Recombines the mutable and static parts of the model.
        model = eqx.combine(rollout_shared_state.model_arr, rollout_constants.model_static)

        # Gets the observations from the physics state.
        observations = get_observation(
            rollout_env_state=rollout_env_state,
            observations=rollout_constants.observations,
            curriculum_level=rollout_env_state.curriculum_state.level,
            rng=obs_rng,
        )

        # Samples an action from the model.
        action = self.sample_action(
            model=model,
            model_carry=rollout_env_state.model_carry,
            physics_model=rollout_shared_state.physics_model,
            physics_state=rollout_env_state.physics_state,
            observations=observations,
            commands=rollout_env_state.commands,
            rng=act_rng,
            argmax=rollout_constants.argmax_action,
        )

        # Steps the physics engine.
        next_physics_state: PhysicsState = rollout_constants.engine.step(
            action=action.action,
            physics_model=rollout_shared_state.physics_model,
            physics_state=rollout_env_state.physics_state,
            curriculum_level=rollout_env_state.curriculum_state.level,
            rng=physics_rng,
        )

        # Gets termination components and a single termination boolean.
        terminations = get_terminations(
            physics_state=next_physics_state,
            terminations=rollout_constants.terminations,
            curriculum_level=rollout_env_state.curriculum_state.level,
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
            obs=observations,
            command=rollout_env_state.commands,
            event_state=next_physics_state.event_states,
            action=action.action,
            done=terminated,
            success=success,
            timestep=next_physics_state.data.time,
            termination_components=terminations,
            aux_outputs=action.aux_outputs,
        )

        # Conditionally reset on termination.
        next_commands = jax.lax.cond(
            terminated,
            lambda: get_initial_commands(
                rng=cmd_rng,
                physics_data=next_physics_state.data,
                commands=rollout_constants.commands,
                curriculum_level=rollout_env_state.curriculum_state.level,
            ),
            lambda: get_commands(
                prev_commands=rollout_env_state.commands,
                physics_state=next_physics_state,
                rng=cmd_rng,
                commands=rollout_constants.commands,
                curriculum_level=rollout_env_state.curriculum_state.level,
            ),
        )

        next_physics_state = jax.lax.cond(
            terminated,
            lambda: rollout_constants.engine.reset(
                rollout_shared_state.physics_model,
                rollout_env_state.curriculum_state.level,
                reset_rng,
            ),
            lambda: next_physics_state,
        )

        next_carry = jax.lax.cond(
            terminated,
            lambda: self.get_initial_model_carry(carry_rng),
            lambda: action.carry,
        )

        # Gets the variables for the next step.
        next_rollout_env_state = RolloutEnvState(
            commands=next_commands,
            physics_state=next_physics_state,
            randomization_dict=rollout_env_state.randomization_dict,
            model_carry=next_carry,
            reward_carry=rollout_env_state.reward_carry,
            curriculum_state=rollout_env_state.curriculum_state,
            rng=rng,
        )

        return transition, next_rollout_env_state

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
                argmax_action=self.config.run_environment_argmax_action,
            )

        elif self.config.collect_dataset:
            self.collect_dataset(
                num_batches=self.config.dataset_num_batches,
                save_path=self.config.dataset_save_path,
                argmax_action=self.config.collect_dataset_argmax_action,
            )

        else:
            self.run_training()

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
        viewer: GlfwMujocoViewer | DefaultMujocoViewer,
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
        trajectory_list: list[Trajectory] = [jax.tree.map(lambda arr: arr[i], trajectory) for i in range(num_steps)]

        frame_list: list[np.ndarray] = []

        for frame_id, trajectory in enumerate(trajectory_list):
            # Updates the model with the latest data.
            viewer.data.qpos[:] = np.array(trajectory.qpos)
            viewer.data.qvel[:] = np.array(trajectory.qvel)
            mujoco.mj_forward(viewer.model, viewer.data)

            def render_callback(model: mujoco.MjModel, data: mujoco.MjData, scene: mujoco.MjvScene) -> None:
                if self.config.render_markers:
                    for marker in markers:
                        marker(model, data, scene, trajectory)

            frame = viewer.read_pixels(callback=render_callback)

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

    def log_logged_trajectory(
        self,
        logged_traj: LoggedTrajectory,
        markers: Collection[Marker],
        viewer: GlfwMujocoViewer | DefaultMujocoViewer,
    ) -> None:
        """Visualizes a single trajectory.

        Args:
            logged_traj: The single trajectory to log.
            markers: The markers to visualize.
            viewer: The Mujoco viewer to render the scene with.
            name: The name of the trajectory being logged.
        """
        # Clips the trajectory to the desired length.
        if self.config.render_length_seconds is not None:
            render_frames = round(self.config.render_length_seconds / self.config.ctrl_dt)
            logged_traj = jax.tree.map(lambda arr: arr[:render_frames], logged_traj)

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
            trajectory=logged_traj.trajectory,
            markers=markers,
            viewer=viewer,
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
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectories: Trajectory,
        rewards: Rewards,
        rollout_env_states: RolloutEnvState,
        rollout_shared_state: RolloutSharedState,
        rollout_constants: RolloutConstants,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, PyTree, xax.FrozenDict[str, Array], LoggedTrajectory]:
        """Updates the model on the given trajectory.

        This function should be implemented according to the specific RL method
        that we are using.

        Args:
            optimizer: The optimizer to use.
            opt_state: The optimizer state.
            trajectories: The trajectories to update the model on.
            rewards: The rewards to update the model on.
            rollout_env_states: The environment variables to use for the rollout.
            rollout_shared_state: The shared state to use for the rollout.
            rollout_constants: The constants to use for the rollout.
            rng: The random seed.

        Returns:
            A tuple containing the updated model, optimizer state, next model
            carry, metrics to log, and the single trajectory to log. If a metric
            has a single element it is logged as a scalar, otherwise it is
            logged as a histogram.
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
        randomizers: Collection[PhysicsRandomizer],
    ) -> Collection[Marker]:
        markers: list[Marker] = []
        for command in commands:
            markers.extend(command.get_markers())
        for observation in observations:
            markers.extend(observation.get_markers())
        for reward in rewards:
            markers.extend(reward.get_markers())
        for randomizer in randomizers:
            markers.extend(randomizer.get_markers())
        return markers

    @xax.jit(static_argnames=["self", "rollout_constants"], jit_level=2)
    def _single_unroll(
        self,
        rollout_constants: RolloutConstants,
        rollout_env_state: RolloutEnvState,
        rollout_shared_state: RolloutSharedState,
    ) -> tuple[Trajectory, Rewards, RolloutEnvState]:
        # Applies randomizations to the model.
        rollout_shared_state = RolloutSharedState(
            physics_model=rollout_shared_state.physics_model.tree_replace(rollout_env_state.randomization_dict),
            model_arr=rollout_shared_state.model_arr,
        )

        def scan_fn(carry: RolloutEnvState, _: None) -> tuple[RolloutEnvState, Trajectory]:
            trajectory, next_rollout_variables = self.step_engine(
                rollout_constants=rollout_constants,
                rollout_env_state=carry,
                rollout_shared_state=rollout_shared_state,
            )
            return next_rollout_variables, trajectory

        # Scans the engine for the desired number of steps.
        next_rollout_variables, trajectory = xax.scan(
            scan_fn,
            rollout_env_state,
            length=self.rollout_length_steps,
        )

        # Gets the rewards.
        reward = get_rewards(
            trajectory=trajectory,
            rewards=rollout_constants.rewards,
            rewards_carry=rollout_env_state.reward_carry,
            rollout_length_steps=self.rollout_length_steps,
            clip_min=self.config.reward_clip_min,
            clip_max=self.config.reward_clip_max,
        )

        return trajectory, reward, next_rollout_variables

    @xax.jit(static_argnames=["self", "optimizer", "rollout_constants"], jit_level=1)
    def _rl_train_loop_step(
        self,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        rollout_constants: RolloutConstants,
        rollout_env_states: RolloutEnvState,
        rollout_shared_state: RolloutSharedState,
        state: xax.State,
        rng: PRNGKeyArray,
    ) -> tuple[optax.OptState, Metrics, RolloutEnvState, RolloutSharedState, LoggedTrajectory]:
        """Runs a single step of the RL training loop."""

        def single_step_fn(
            carry: tuple[optax.OptState, RolloutEnvState, RolloutSharedState],
            rng: PRNGKeyArray,
        ) -> tuple[
            tuple[optax.OptState, RolloutEnvState, RolloutSharedState],
            tuple[Metrics, LoggedTrajectory],
        ]:
            opt_state, rollout_env_states, rollout_shared_state = carry

            # Rolls out a new trajectory.
            vmapped_unroll = jax.vmap(self._single_unroll, in_axes=(None, 0, None))
            trajectories, rewards, next_rollout_env_states = vmapped_unroll(
                rollout_constants,
                rollout_env_states,
                rollout_shared_state,
            )
            # The reward carry is updated every rollout using the last episode.
            next_reward_carry = rewards.carry

            # Runs update on the previous trajectory.
            model_arr, opt_state, next_model_carry, train_metrics, logged_traj = self.update_model(
                optimizer=optimizer,
                opt_state=opt_state,
                trajectories=trajectories,
                rewards=rewards,
                rollout_env_states=rollout_env_states,
                rollout_shared_state=rollout_shared_state,
                rollout_constants=rollout_constants,
                rng=rng,
            )

            # Store all the metrics to log.
            metrics = Metrics(
                train=train_metrics,
                reward=xax.FrozenDict(self.get_reward_metrics(trajectories, rewards)),
                termination=xax.FrozenDict(self.get_termination_metrics(trajectories)),
                curriculum_level=rollout_env_states.curriculum_state.level,
            )

            # Steps the curriculum.
            curriculum_state = rollout_constants.curriculum(
                trajectory=trajectories,
                rewards=rewards,
                training_state=state,
                prev_state=rollout_env_states.curriculum_state,
            )

            # Constructs the final rollout variables.
            next_rollout_env_states = RolloutEnvState(
                # For the next rollout, we use the model carry from the output
                # of the model update instead of the output of the rollout.
                # This was shown to work slightly better in practice - for an
                # RNN model, for example, after updating the model, the model
                # carry will be new and the previous rollout's model carry will
                # be incorrect.
                commands=next_rollout_env_states.commands,
                physics_state=next_rollout_env_states.physics_state,
                randomization_dict=rollout_env_states.randomization_dict,
                model_carry=next_model_carry,
                reward_carry=next_reward_carry,
                curriculum_state=curriculum_state,
                rng=next_rollout_env_states.rng,
            )

            next_rollout_shared_state = RolloutSharedState(
                model_arr=model_arr,
                physics_model=rollout_shared_state.physics_model,
            )

            return (opt_state, next_rollout_env_states, next_rollout_shared_state), (metrics, logged_traj)

        (opt_state, rollout_env_states, rollout_shared_state), (metrics, logged_traj) = jax.lax.scan(
            single_step_fn,
            (opt_state, rollout_env_states, rollout_shared_state),
            jax.random.split(rng, self.config.epochs_per_log_step),
        )

        # Convert any array with more than one element to a histogram.
        metrics = jax.tree.map(lambda x: self.get_histogram(x) if isinstance(x, Array) and x.size > 1 else x, metrics)

        # Only get final trajectory and rewards.
        logged_traj = jax.tree.map(lambda arr: arr[-1], logged_traj)

        # Metrics, final_trajectories, final_rewards batch dim of epochs.
        # Rollout variables has batch dim of num_envs and are used next rollout.
        return opt_state, metrics, rollout_env_states, rollout_shared_state, logged_traj

    def run_environment(
        self,
        num_steps: int | None = None,
        save_path: str | Path | None = None,
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
            save_path: If provided, save the rendered video to the given path.
            argmax_action: If set, get the argmax action, otherwise sample
                randomly from the model.
        """
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        with self, jax.disable_jit():
            rng = self.prng_key()
            self.set_loggers()

            rng, model_rng = jax.random.split(rng)
            model, _ = self.load_initial_state(model_rng, load_optimizer=False)

            # Loads the Mujoco model and logs some information about it.
            mj_model = self.get_mujoco_model()
            mujoco_info = OmegaConf.to_yaml(DictConfig(self.get_mujoco_model_info(mj_model)))
            self.logger.log_file("mujoco_info.yaml", mujoco_info)

            # Initializes the control loop variables.
            randomizers = self.get_physics_randomizers(mj_model)

            # JAX requires that we partition the model into mutable and static
            # parts in order to use lax.scan, so that `arr` can be a PyTree.
            model_arr, model_static = eqx.partition(model, self.model_partition_fn)

            rollout_constants = self._get_rollout_constants(mj_model, model_static, argmax_action)
            rollout_env_state = self._get_rollout_env_state(rng, rollout_constants, mj_model, randomizers)
            rollout_shared_state = self._get_rollout_shared_state(mj_model, model_arr)

            # Creates the markers.
            markers = self.get_markers(
                commands=rollout_constants.commands,
                observations=rollout_constants.observations,
                rewards=rollout_constants.rewards,
                randomizers=randomizers,
            )

            # Creates the viewer.
            viewer = get_viewer(
                mj_model=mj_model,
                config=self.config,
                mj_data=rollout_env_state.physics_state.data,
                save_path=save_path,
            )

            iterator = tqdm.trange(num_steps) if num_steps is not None else tqdm.tqdm(itertools.count())
            frames: list[np.ndarray] = []

            transitions = []

            try:
                for _ in iterator:
                    transition, rollout_env_state = self.step_engine(
                        rollout_constants=rollout_constants,
                        rollout_env_state=rollout_env_state,
                        rollout_shared_state=rollout_shared_state,
                    )
                    transitions.append(transition)

                    # Logs the frames to render.
                    viewer.data.qpos[:] = np.array(rollout_env_state.physics_state.data.qpos)
                    viewer.data.qvel[:] = np.array(rollout_env_state.physics_state.data.qvel)
                    mujoco.mj_forward(viewer.model, viewer.data)

                    def render_callback(model: mujoco.MjModel, data: mujoco.MjData, scene: mujoco.MjvScene) -> None:
                        for marker in markers:
                            marker(model, data, scene, transition)

                    if save_path is None:
                        viewer.render(callback=render_callback)
                    else:
                        frames.append(viewer.read_pixels(callback=render_callback))

            except (KeyboardInterrupt, bdb.BdbQuit):
                logger.info("Keyboard interrupt, exiting environment loop")

            if len(transitions) > 0:
                trajectory = jax.tree_map(lambda *xs: jnp.stack(xs), *transitions)

                get_rewards(
                    trajectory=trajectory,
                    rewards=rollout_constants.rewards,
                    rewards_carry=rollout_env_state.reward_carry,
                    rollout_length_steps=self.rollout_length_steps,
                    clip_min=self.config.reward_clip_min,
                    clip_max=self.config.reward_clip_max,
                )

                # TODO: some nice visualizer of rewards...

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

    def _get_rollout_constants(
        self,
        mj_model: PhysicsModel,
        model_static: PyTree,
        argmax_action: bool,
    ) -> RolloutConstants:
        metadata = self.get_mujoco_model_metadata(mj_model)
        engine = self.get_engine(mj_model, metadata)
        observations = self.get_observations(mj_model)
        commands = self.get_commands(mj_model)
        rewards_terms = self.get_rewards(mj_model)
        terminations = self.get_terminations(mj_model)
        curriculum = self.get_curriculum(mj_model)

        return RolloutConstants(
            model_static=model_static,
            engine=engine,
            observations=tuple(observations),
            commands=tuple(commands),
            rewards=tuple(rewards_terms),
            terminations=tuple(terminations),
            curriculum=curriculum,
            argmax_action=argmax_action,
        )

    def _get_rollout_shared_state(self, mj_model: PhysicsModel, model_arr: PyTree) -> RolloutSharedState:
        return RolloutSharedState(
            physics_model=mj_model,
            model_arr=model_arr,
        )

    def _get_rollout_env_state(
        self,
        rng: PRNGKeyArray,
        rollout_constants: RolloutConstants,
        mj_model: PhysicsModel,
        randomizers: Collection[PhysicsRandomizer],
    ) -> RolloutEnvState:
        rng, carry_rng, command_rng, rand_rng, rollout_rng, curriculum_rng, reward_rng = jax.random.split(rng, 7)

        # Vectorize across N environments for MJX models, use single model for Mujoco.
        # TODO (for a later refactor): just one code path and vmap on the outside.
        if isinstance(mj_model, mjx.Model):
            # Defines the vectorized initialization functions.
            carry_fn = jax.vmap(self.get_initial_model_carry, in_axes=0)
            command_fn = jax.vmap(get_initial_commands, in_axes=(0, 0, None, 0))
            reward_carry_fn = jax.vmap(get_initial_reward_carry, in_axes=(0, None))

            # Gets the initial curriculum state.
            curriculum_state = jax.vmap(rollout_constants.curriculum.get_initial_state, in_axes=0)(
                jax.random.split(curriculum_rng, self.config.num_envs)
            )

            # Gets the per-environment randomizations.
            randomization_fn = jax.vmap(apply_randomizations, in_axes=(None, None, None, 0, 0))
            randomization_dict, physics_state = randomization_fn(
                mj_model,
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
                curriculum_state=curriculum_state,
                rng=jax.random.split(rollout_rng, self.config.num_envs),
            )

        else:
            # Gets the initial curriculum state.
            curriculum_state = rollout_constants.curriculum.get_initial_state(curriculum_rng)

            # Gets the environment randomizations.
            randomization_dict, physics_state = apply_randomizations(
                mj_model,
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

            rng, model_rng = jax.random.split(rng)
            model, state = self.load_initial_state(model_rng, load_optimizer=False)

            if save_path is None:
                save_path = self.exp_dir / f"dataset_{state.num_steps}.npz"

            # Loads the Mujoco model and logs some information about it.
            mj_model: PhysicsModel = self.get_mujoco_model()
            mujoco_info = OmegaConf.to_yaml(DictConfig(self.get_mujoco_model_info(mj_model)))
            self.logger.log_file("mujoco_info.yaml", mujoco_info)

            mjx_model = self.get_mjx_model(mj_model)
            randomizations = self.get_physics_randomizers(mjx_model)

            # JAX requires that we partition the model into mutable and static
            # parts in order to use lax.scan, so that `arr` can be a PyTree.
            model_arr, model_static = eqx.partition(model, self.model_partition_fn)

            rollout_constants = self._get_rollout_constants(mjx_model, model_static, argmax_action)
            rollout_env_state = self._get_rollout_env_state(rng, rollout_constants, mjx_model, randomizations)
            rollout_shared_state = self._get_rollout_shared_state(mjx_model, model_arr)

            state = self.on_training_start(state)

            @xax.jit()
            def get_batch(
                rollout_env_state: RolloutEnvState,
            ) -> tuple[Trajectory, Rewards, RolloutEnvState]:
                vmapped_unroll = jax.vmap(self._single_unroll, in_axes=(None, 0, None))
                return vmapped_unroll(rollout_constants, rollout_env_state, rollout_shared_state)

            with TrajectoryDataset.writer(save_path, num_batches * self.batch_size) as writer:
                for _ in tqdm.trange(num_batches):
                    trajectories, rewards, rollout_env_state = get_batch(rollout_env_state)

                    # Splits trajectories and rewards into a list of `batch_size` samples.
                    for i in range(0, len(trajectories.done), self.batch_size):
                        trajectory = jax.tree.map(lambda x: x[i], trajectories)
                        reward = jax.tree.map(lambda x: x[i], rewards)
                        writer.write(trajectory, reward)

            logger.info("Saved dataset to %s", save_path)

    def run_training(self) -> None:
        """Wraps the training loop and provides clean XAX integration."""
        with self:
            rng = self.prng_key()
            self.set_loggers()

            if xax.is_master():
                Thread(target=self.log_state, daemon=True).start()

            rng, model_rng = jax.random.split(rng)
            model, optimizer, opt_state, state = self.load_initial_state(model_rng, load_optimizer=True)

            # Loads the Mujoco model and logs some information about it.
            mj_model: PhysicsModel = self.get_mujoco_model()
            mujoco_info = OmegaConf.to_yaml(DictConfig(self.get_mujoco_model_info(mj_model)))
            self.logger.log_file("mujoco_info.yaml", mujoco_info)

            # Loads the MJX model, and initializes the loop variables.
            mjx_model = self.get_mjx_model(mj_model)
            randomizers = self.get_physics_randomizers(mjx_model)

            # JAX requires that we partition the model into mutable and static
            # parts in order to use lax.scan, so that `arr` can be a PyTree.
            model_arr, model_static = eqx.partition(model, self.model_partition_fn)

            rollout_constants = self._get_rollout_constants(mjx_model, model_static, argmax_action=False)
            rollout_env_states = self._get_rollout_env_state(rng, rollout_constants, mjx_model, randomizers)
            rollout_shared_state = self._get_rollout_shared_state(mjx_model, model_arr)

            # Creates the markers.
            markers = self.get_markers(
                commands=rollout_constants.commands,
                observations=rollout_constants.observations,
                rewards=rollout_constants.rewards,
                randomizers=randomizers,
            )

            # Creates the viewer.
            viewer = get_viewer(
                mj_model=mj_model,
                config=self.config,
                mode="offscreen",
            )

            state = self.on_training_start(state)

            def on_exit() -> None:
                model = eqx.combine(rollout_shared_state.model_arr, rollout_constants.model_static)
                self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    opt_state=opt_state,
                    state=state,
                )

            # Handle user-defined interrupts during the training loop.
            self.add_signal_handler(on_exit, signal.SIGUSR1, signal.SIGTERM)

            is_first_step = True

            # Clean up variables which are not part of the control loop.
            del model_arr, model_static, mjx_model, randomizers

            try:
                while not self.is_training_over(state):
                    state = self.on_step_start(state)

                    # Use a different phase for logging full trajectories.
                    if self.valid_step_timer.is_valid_step(state):
                        state = state.replace(phase="valid")
                    else:
                        state = state.replace(phase="train")

                    # Runs the training loop.
                    rng, update_rng = jax.random.split(rng)
                    with xax.ContextTimer() as timer:
                        (
                            opt_state,
                            metrics,
                            rollout_env_states,
                            rollout_shared_state,
                            logged_traj,
                        ) = self._rl_train_loop_step(
                            optimizer=optimizer,
                            opt_state=opt_state,
                            rollout_constants=rollout_constants,
                            rollout_env_states=rollout_env_states,
                            rollout_shared_state=rollout_shared_state,
                            state=state,
                            rng=update_rng,
                        )

                        if self.config.profile_memory:
                            opt_state = jax.block_until_ready(opt_state)
                            rollout_env_states = jax.block_until_ready(rollout_env_states)
                            rollout_shared_state = jax.block_until_ready(rollout_shared_state)
                            logged_traj = jax.block_until_ready(logged_traj)
                            jax.profiler.save_device_memory_profile(self.exp_dir / "train_loop_step.prof")

                    # Updates the state.
                    num_steps = self.config.epochs_per_log_step
                    num_samples = self.rollout_num_samples * self.config.epochs_per_log_step
                    elapsed_time = timer.elapsed_time
                    if state.phase == "train":
                        state = state.replace(
                            num_steps=state.num_steps + num_steps,
                            num_samples=state.num_samples + num_samples,
                            elapsed_time_s=state.elapsed_time_s + elapsed_time,
                        )
                    else:
                        state = state.replace(
                            num_valid_steps=state.num_valid_steps + num_steps,
                            num_valid_samples=state.num_valid_samples + num_samples,
                            elapsed_time_s=state.elapsed_time_s + elapsed_time,
                        )

                    # Only log trajectory information on validation steps.
                    if state.phase == "valid":
                        self.log_logged_trajectory(logged_traj=logged_traj, markers=markers, viewer=viewer)

                    if is_first_step:
                        is_first_step = False
                        logger.log(
                            xax.LOG_STATUS,
                            "First step time: %s",
                            xax.format_timedelta(datetime.timedelta(seconds=timer.elapsed_time), short=True),
                        )

                    self.log_train_metrics(metrics)
                    self.log_state_timers(state)
                    self.write_logs(state)

                    if self.should_checkpoint(state):
                        model = eqx.combine(rollout_shared_state.model_arr, rollout_constants.model_static)
                        self.save_checkpoint(model=model, optimizer=optimizer, opt_state=opt_state, state=state)

                    state = self.on_step_end(state)

                # Save the checkpoint when done.
                model = eqx.combine(rollout_shared_state.model_arr, rollout_constants.model_static)
                self.save_checkpoint(model=model, optimizer=optimizer, opt_state=opt_state, state=state)

            except xax.TrainingFinishedError:
                if xax.is_master():
                    msg = f"Finished training after {state.num_steps}steps and {state.num_samples} samples"
                    xax.show_info(msg, important=True)

                model = eqx.combine(rollout_shared_state.model_arr, rollout_constants.model_static)
                self.save_checkpoint(model=model, optimizer=optimizer, opt_state=opt_state, state=state)

            except (KeyboardInterrupt, bdb.BdbQuit):
                if xax.is_master():
                    xax.show_info("Interrupted training", important=True)

            except BaseException:
                exception_tb = textwrap.indent(xax.highlight_exception_message(traceback.format_exc()), "  ")
                sys.stdout.write(f"Caught exception during training loop:\n\n{exception_tb}\n")
                sys.stdout.flush()

                model = eqx.combine(rollout_shared_state.model_arr, rollout_constants.model_static)
                self.save_checkpoint(model=model, optimizer=optimizer, opt_state=opt_state, state=state)

            finally:
                state = self.on_training_end(state)
