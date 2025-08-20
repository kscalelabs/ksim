"""Defines the base command class."""

__all__ = [
    "Command",
    "FloatVectorCommand",
    "IntVectorCommand",
    "LinearVelocityCommandValue",
    "LinearVelocityCommand",
    "AngularVelocityCommandValue",
    "AngularVelocityCommand",
    "StartPositionCommand",
    "StartQuaternionCommand",
    "CartesianCoordinateCommand",
    "SinusoidalGaitCommand",
    "SinusoidalGaitCommandValue",
    "BaseHeightCommand",
    "JointPositionCommand",
    "JointPositionCommandValue",
]

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Self

import attrs
import jax
import jax.numpy as jnp
import mujoco
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.types import PhysicsData, PhysicsModel, Trajectory
from ksim.utils.mujoco import get_joint_names_in_order
from ksim.vis import Marker

logger = logging.getLogger(__name__)


@attrs.define(frozen=True, kw_only=True)
class Command(ABC):
    """Base class for commands."""

    @abstractmethod
    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> PyTree:
        """Returns the initial command.

        Args:
            physics_data: The current physics data.
            curriculum_level: The current curriculum level, a value between
                zero and one that indicates the difficulty of the task.
            rng: The random number generator.

        Returns:
            The initial command, with shape (command_dim).
        """

    @abstractmethod
    def __call__(
        self,
        prev_command: PyTree,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> PyTree:
        """Updates the command.

        Args:
            prev_command: The previous command.
            physics_data: The current physics data.
            curriculum_level: The current curriculum level, a value between
                zero and one that indicates the difficulty of the task.
            rng: The random number generator.

        Returns:
            The command to perform, with shape (command_dim).
        """

    def get_markers(self, name: str) -> Collection[Marker]:
        """Get the visualizations for the command.

        Args:
            name: The name of the command.

        Returns:
            The visualizations to add to the scene.
        """
        return []


@attrs.define(frozen=True, kw_only=True)
class FloatVectorCommand(Command):
    """Samples a set of scalars uniformly within some bounding box.

    The commands update to some new commands with some probability. They can
    be used to represent any vector, such as target position, velocity, etc.

    The zero_prob can be used to set the command to zero with some probability.
    """

    ranges: tuple[tuple[float, float], ...] = attrs.field()
    switch_prob: float = attrs.field(default=0.0)
    zero_prob: float = attrs.field(default=0.0)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        ranges = jnp.array(self.ranges)  # (N, 2)
        zero_mask = jax.random.bernoulli(rng, self.zero_prob)
        return jnp.where(
            zero_mask, 0.0, jax.random.uniform(rng, (ranges.shape[0],), minval=ranges[:, 0], maxval=ranges[:, 1])
        )

    def __call__(
        self,
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)


@attrs.define(frozen=True, kw_only=True)
class IntVectorCommand(Command):
    """Samples an integer vector uniformly within some bounding box.

    The zero_prob can be used to set the command to zero with some probability.
    """

    ranges: tuple[tuple[int, int], ...] = attrs.field()
    switch_prob: float = attrs.field(default=0.0)
    zero_prob: float = attrs.field(default=0.0)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        ranges = jnp.array(self.ranges)  # (N, 2)
        zero_mask = jax.random.bernoulli(rng, self.zero_prob)
        return jnp.where(
            zero_mask, 0.0, jax.random.randint(rng, (ranges.shape[0],), minval=ranges[:, 0], maxval=ranges[:, 1] + 1)
        )

    def __call__(
        self,
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)


@attrs.define(frozen=True, kw_only=True)
class StartPositionCommand(Command):
    """Provides the initial position of the robot as a command."""

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        return jnp.array(physics_data.qpos[..., :3])

    def __call__(
        self,
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        return prev_command


@attrs.define(frozen=True, kw_only=True)
class StartQuaternionCommand(Command):
    """Provides the initial quaternion of the robot as a command."""

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        return jnp.array(physics_data.qpos[..., 3:7])

    def __call__(
        self,
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        return prev_command


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LinearVelocityCommandValue:
    vel: Array
    yaw: Array
    xvel: Array
    yvel: Array


@attrs.define(kw_only=True)
class LinearVelocityCommandMarker(Marker):
    """Visualises the planar (x,y) linear velocity command."""

    command_name: str = attrs.field()
    size: float = attrs.field(default=0.03)
    arrow_scale: float = attrs.field(default=0.3)
    height: float = attrs.field(default=0.5)
    base_length: float = attrs.field(default=0.15)
    zero_threshold: float = attrs.field(default=1e-4)

    def update(self, trajectory: Trajectory) -> None:
        cmd: LinearVelocityCommandValue = trajectory.command[self.command_name]
        vx = float(cmd.xvel)
        vy = float(cmd.yvel)
        speed = float(cmd.vel)

        self.pos = (0.0, 0.0, self.height)

        # Always show an arrow with base_length plus scaling by speed
        self.geom = mujoco.mjtGeom.mjGEOM_ARROW  # pyright: ignore[reportAttributeAccessIssue]
        arrow_length = self.base_length + self.arrow_scale * speed
        self.scale = (self.size, self.size, arrow_length)

        # If command is near-zero, show grey arrow pointing +X.
        if speed < self.zero_threshold:
            self.orientation = self.quat_from_direction((1.0, 0.0, 0.0))
            self.rgba = (0.8, 0.8, 0.8, 0.8)
        else:
            self.orientation = self.quat_from_direction((vx, vy, 0.0))
            self.rgba = (0.2, 0.8, 0.2, 0.8)

    @classmethod
    def get(
        cls,
        command_name: str,
        *,
        arrow_scale: float = 0.3,
        height: float = 0.5,
        base_length: float = 0.15,
    ) -> Self:
        return cls(
            command_name=command_name,
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_ARROW,  # pyright: ignore[reportAttributeAccessIssue]
            scale=(0.03, 0.03, base_length),
            arrow_scale=arrow_scale,
            height=height,
            base_length=base_length,
            track_rotation=True,
        )


@attrs.define(frozen=True)
class LinearVelocityCommand(Command):
    """Command to move the robot in a straight line.

    By convention, X is forward and Y is left. The switching probability is the
    probability of resampling the command at each step. The zero probability is
    the probability of the command being zero - this can be used to turn off
    any command.
    """

    min_vel: float = attrs.field()
    max_vel: float = attrs.field()
    max_yaw: float = attrs.field(default=0.0)
    zero_prob: float = attrs.field(default=0.0)
    backward_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)
    vis_height: float = attrs.field(default=0.5)
    vis_scale: float = attrs.field(default=0.05)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> LinearVelocityCommandValue:
        rng_vel, rng_yaw, rng_zero, rng_backward = jax.random.split(rng, 4)

        vel = jax.random.uniform(rng_vel, (), minval=self.min_vel, maxval=self.max_vel)
        yaw = jax.random.uniform(rng_yaw, (), minval=-self.max_yaw, maxval=self.max_yaw)

        zero_mask = jax.random.bernoulli(rng_zero, self.zero_prob)
        backward_mask = jax.random.bernoulli(rng_backward, self.backward_prob)
        vel = jnp.where(zero_mask, 0.0, jnp.where(backward_mask, -vel, vel))
        yaw = jnp.where(zero_mask, 0.0, yaw)

        xvel = vel * jnp.cos(yaw)
        yvel = vel * jnp.sin(yaw)

        return LinearVelocityCommandValue(
            vel=vel,
            yaw=yaw,
            xvel=xvel,
            yvel=yvel,
        )

    def __call__(
        self,
        prev_command: LinearVelocityCommandValue,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> LinearVelocityCommandValue:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jax.tree_util.tree_map(
            lambda x, y: jnp.where(switch_mask, y, x),
            prev_command,
            new_commands,
        )

    def get_markers(self, name: str) -> Collection[Marker]:
        return [LinearVelocityCommandMarker.get(command_name=name, height=self.vis_height)]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AngularVelocityCommandValue:
    vel: Array


@attrs.define(frozen=True)
class AngularVelocityCommand(Command):
    min_vel: float = attrs.field()
    max_vel: float = attrs.field()
    zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> AngularVelocityCommandValue:
        rng_vel, rng_zero = jax.random.split(rng)

        vel = jax.random.uniform(rng_vel, (), minval=self.min_vel, maxval=self.max_vel)

        zero_mask = jax.random.bernoulli(rng_zero, self.zero_prob)
        vel = jnp.where(zero_mask, 0.0, vel)

        return AngularVelocityCommandValue(vel=vel)

    def __call__(
        self,
        prev_command: AngularVelocityCommandValue,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> AngularVelocityCommandValue:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jax.tree_util.tree_map(
            lambda x, y: jnp.where(switch_mask, y, x),
            prev_command,
            new_commands,
        )


@attrs.define(kw_only=True)
class PositionCommandMarker(Marker):
    command_name: str = attrs.field()

    def update(self, trajectory: Trajectory) -> None:
        """Update the marker position and rotation."""
        self.pos = trajectory.command[self.command_name][:3]

    @classmethod
    def get(
        cls,
        command_name: str,
        base_name: str | None,
        radius: float,
        rgba: tuple[float, float, float, float],
    ) -> Self:
        return cls(
            command_name=command_name,
            target_name=base_name,
            geom=mujoco.mjtGeom.mjGEOM_SPHERE,  # pyright: ignore[reportAttributeAccessIssue]
            scale=(radius, radius, radius),
            rgba=rgba,
        )


@attrs.define(frozen=True)
class CartesianCoordinateCommand(Command):
    """Samples a target xyz position within a bounding box.

    The bounding box is defined by min and max coordinates.
    The target will smoothly transition between points within this box.
    """

    box_min: tuple[float, float, float] = attrs.field()
    box_max: tuple[float, float, float] = attrs.field()
    dt: float = attrs.field()
    base_name: str | None = attrs.field(default=None)
    vis_radius: float = attrs.field(default=0.05)
    vis_color: tuple[float, float, float, float] = attrs.field(default=(1.0, 0.0, 0.0, 0.8))
    min_speed: float = attrs.field(default=0.5)
    max_speed: float = attrs.field(default=3.0)
    switch_prob: float = attrs.field(default=0.0)
    jump_prob: float = attrs.field(default=0.0)
    unique_name: str | None = attrs.field(default=None)
    curriculum_scale: float = attrs.field(default=0.1)

    def _sample_box(
        self,
        rng: PRNGKeyArray,
        physics_data: PhysicsData,
        curriculum_level: Array,
    ) -> Array:
        min_coords = jnp.array(self.box_min)
        max_coords = jnp.array(self.box_max)

        # Calculate the center and half-size (extent) of the box
        center = (min_coords + max_coords) / 2.0
        half_size = (max_coords - min_coords) / 2.0

        # Scale the half-size based on curriculum level
        scale_factor = self.curriculum_scale * curriculum_level + 1.0
        scaled_half_size = half_size * scale_factor

        scaled_min_coords = center - scaled_half_size
        scaled_max_coords = center + scaled_half_size

        # Sample uniformly within the scaled box
        return jax.random.uniform(
            rng,
            shape=(3,),
            minval=scaled_min_coords,
            maxval=scaled_max_coords,
        )

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        # Sample initial target and speed
        rng_target, rng_speed = jax.random.split(rng)
        target = self._sample_box(rng_target, physics_data, curriculum_level)
        speed = jax.random.uniform(
            rng_speed,
            (),
            minval=self.min_speed,
            maxval=self.max_speed,
        )

        # Return [current_x, current_y, current_z, target_x, target_y, target_z, speed]
        return jnp.concatenate([target, target, jnp.array([speed])])

    def __call__(
        self,
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        # Unpack previous command
        current = prev_command[:3]
        target = prev_command[3:6]
        speed = prev_command[6]

        # Calculate distance to target
        distance = jnp.linalg.norm(target - current)

        # If we've reached the target, sample a new one
        rng_a, rng_b, rng_c = jax.random.split(rng, 3)
        reached_target = distance < self.dt * speed * 0.5

        # Sample new target and speed if reached
        new_target = self._sample_box(rng_a, physics_data, curriculum_level)
        new_speed = jax.random.uniform(
            rng_b,
            (),
            minval=self.min_speed,
            maxval=self.max_speed,
        )

        switch_mask = jax.random.bernoulli(rng_c, self.switch_prob)

        jump_mask = jax.random.bernoulli(rng_c, self.jump_prob)

        # Update target and speed if reached
        target = jnp.where(
            (reached_target & switch_mask) | jump_mask,
            new_target,
            target,
        )
        speed = jnp.where((reached_target & switch_mask) | jump_mask, new_speed, speed)

        # Calculate step size based on speed and timestep
        dt = self.dt
        step_size = speed * dt

        # Move current position towards target
        direction = target - current
        direction_norm = jnp.linalg.norm(direction)
        direction = jnp.where(direction_norm > 0, direction / direction_norm, direction)

        # Calculate new position
        new_current = current + direction * jnp.minimum(step_size, distance)

        # Return updated command
        return jnp.concatenate([new_current, target, jnp.array([speed])])

    def get_markers(self, name: str) -> Collection[Marker]:
        return [PositionCommandMarker.get(name, self.base_name, self.vis_radius, self.vis_color)]

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        box_min: tuple[float, float, float],
        box_max: tuple[float, float, float],
        vis_target_name: str | None = None,
        vis_radius: float = 0.05,
        vis_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.8),
        unique_name: str | None = None,
        min_speed: float = 0.5,
        max_speed: float = 3.0,
        switch_prob: float = 1.0,
        jump_prob: float = 0.0,
        curriculum_scale: float = 1.0,
    ) -> Self:
        return cls(
            base_name=vis_target_name,
            box_min=box_min,
            box_max=box_max,
            dt=float(model.opt.timestep),
            vis_radius=vis_radius,
            vis_color=vis_color,
            unique_name=unique_name,
            min_speed=min_speed,
            max_speed=max_speed,
            switch_prob=switch_prob,
            jump_prob=jump_prob,
            curriculum_scale=curriculum_scale,
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SinusoidalGaitCommandValue:
    moving_flag: Array
    phase: Array
    height: Array


@attrs.define(frozen=True, kw_only=True)
class SinusoidalGaitCommand(Command):
    gait_period: float = attrs.field()
    ctrl_dt: float = attrs.field()
    max_height: float = attrs.field()
    height_offset: float = attrs.field(default=0.0)
    num_feet: int = attrs.field(default=2)
    stance_ratio: float = attrs.field(default=0.6)
    moving_threshold: float = attrs.field(default=0.05)

    def _foot_height_profile(self, phase: Array) -> Array:
        """Computes foot height as a function of phase in [0, 1)."""
        # Define stance/swing phase cutoff
        swing_phase = phase >= self.stance_ratio

        # Normalize swing phase ∈ [0, 1]
        swing_progress = (phase - self.stance_ratio) / (1.0 - self.stance_ratio)
        swing_progress = jnp.clip(swing_progress, 0.0, 1.0)

        # Smooth foot lift trajectory: half-sine (or use poly for asymmetry)
        swing_height = jnp.sin(jnp.pi * swing_progress)  # ∈ [0, 1]
        target_height = jnp.where(swing_phase, self.max_height * swing_height, 0.0) + self.height_offset

        return target_height

    def _get_height(self, phase: Array) -> Array:
        foot_phase_offsets = jnp.linspace(0.0, 1.0, self.num_feet, endpoint=False)
        per_foot_phase = (phase + foot_phase_offsets) % 1.0
        return self._foot_height_profile(per_foot_phase)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> SinusoidalGaitCommandValue:
        moving_flag = jnp.ones((), dtype=jnp.bool_)
        phase = jnp.zeros((), dtype=jnp.float32)
        return SinusoidalGaitCommandValue(
            moving_flag=moving_flag,
            phase=phase,
            height=self._get_height(phase),
        )

    def __call__(
        self,
        prev_command: SinusoidalGaitCommandValue,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> SinusoidalGaitCommandValue:
        moving_flag = prev_command.moving_flag
        phase = prev_command.phase

        # Compute delta phase: dt / period
        dphase = self.ctrl_dt / self.gait_period
        new_phase = jnp.where(moving_flag, (phase + dphase) % 1.0, 0.0)

        # If not moving, reset the phase.
        is_moving = jnp.linalg.norm(physics_data.qvel[..., :2]) > self.moving_threshold
        moving_flag = jnp.where(is_moving, moving_flag, jnp.zeros_like(moving_flag))
        new_phase = jnp.where(is_moving, new_phase, 0.0)

        return SinusoidalGaitCommandValue(
            moving_flag=moving_flag,
            phase=new_phase,
            height=self._get_height(new_phase),
        )


@attrs.define(frozen=True, kw_only=True)
class BaseHeightCommand(Command):
    min_height: float = attrs.field()
    max_height: float = attrs.field()
    switch_prob: float = attrs.field(default=0.005)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        return jax.random.uniform(rng, (), minval=self.min_height, maxval=self.max_height)

    def __call__(
        self,
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        new_height = jax.random.uniform(rng, (), minval=self.min_height, maxval=self.max_height)
        return jnp.where(jax.random.bernoulli(rng, self.switch_prob), new_height, prev_command)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class JointPositionCommandValue:
    current_position: Array
    target_position: Array
    step_size: Array


@attrs.define(frozen=True, kw_only=True)
class JointPositionCommand(Command):
    """Command each joint to go to a specific position."""

    indices: tuple[int, ...] = attrs.field()
    ranges: tuple[tuple[float, float], ...] = attrs.field()
    ctrl_dt: float = attrs.field()
    min_time: float = attrs.field(validator=attrs.validators.gt(0.0))
    max_time: float = attrs.field(validator=attrs.validators.gt(0.0))

    def sample_target(self, rng: PRNGKeyArray) -> Array:
        ranges = jnp.array(self.ranges)  # (N, 2)
        return jax.random.uniform(rng, (ranges.shape[0],), minval=ranges[:, 0], maxval=ranges[:, 1])

    def sample_time(self, rng: PRNGKeyArray) -> Array:
        return jax.random.uniform(rng, (), minval=self.min_time, maxval=self.max_time)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> JointPositionCommandValue:
        rng_a, rng_b = jax.random.split(rng)
        target = self.sample_target(rng_a)
        start = physics_data.qpos[..., self.indices]
        time_steps = self.sample_time(rng_b) / self.ctrl_dt
        step_size = (target - start) / time_steps
        return JointPositionCommandValue(
            target_position=target,
            step_size=step_size,
            current_position=start,
        )

    def __call__(
        self,
        prev_command: JointPositionCommandValue,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> JointPositionCommandValue:
        target = prev_command.target_position
        current = prev_command.current_position
        step_size = prev_command.step_size
        at_target = (target - current) < step_size * 1.5

        next_command = JointPositionCommandValue(
            target_position=target,
            step_size=step_size,
            current_position=current + step_size,
        )

        # Choose a new target and speed once we're close to the old target.
        new_command = self.initial_command(physics_data, curriculum_level, rng)
        return jax.tree.map(lambda x, y: jnp.where(at_target, y, x), next_command, new_command)

    @classmethod
    def create(
        cls,
        physics_model: PhysicsModel,
        ctrl_dt: float,
        joint_names: Collection[str],
        min_time: float = 0.3,
        max_time: float = 2.0,
    ) -> Self:
        all_names = get_joint_names_in_order(physics_model)
        for joint_name in joint_names:
            if joint_name not in all_names:
                raise ValueError(f"Joint {joint_name} not found in the model! Options are: {all_names}")

        all_ranges = physics_model.jnt_range
        ranges_list = [(minv, maxv) for minv, maxv in all_ranges.tolist()]
        joint_name_to_indices = {name: idx for idx, name in enumerate(get_joint_names_in_order(physics_model))}
        ranges = tuple(ranges_list[joint_name_to_indices[name]] for name in joint_names)
        indices = tuple(joint_name_to_indices[name] + 7 for name in joint_names)

        return cls(
            indices=indices,
            ranges=ranges,
            ctrl_dt=ctrl_dt,
            min_time=min_time,
            max_time=max_time,
        )
