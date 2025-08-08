"""Defines the base command class."""

__all__ = [
    "Command",
    "FloatVectorCommand",
    "IntVectorCommand",
    "JoystickCommand",
    "JoystickCommandValue",
    "LinearVelocityCommand",
    "StartPositionCommand",
    "StartQuaternionCommand",
    "PositionCommand",
    "SinusoidalGaitCommand",
    "SinusoidalGaitCommandValue",
    "EasyJoystickCommand",
    "EasyJoystickCommandValue",
    "BaseHeightCommand",
]

import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Self

import attrs
import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.types import PhysicsData, PhysicsModel, Trajectory
from ksim.utils.validators import sample_probs_validator
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

    def get_markers(self) -> Collection[Marker]:
        """Get the visualizations for the command.

        Args:
            command: The command to get the visualizations for.

        Returns:
            The visualizations to add to the scene.
        """
        return []

    def get_name(self) -> str:
        """Get the name of the command."""
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def command_name(self) -> str:
        return self.get_name()


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
class JoystickCommandValue:
    command: Array
    vels: Array


@attrs.define(kw_only=True)
class JoystickCommandMarker(Marker):
    command_name: str = attrs.field()
    radius: float = attrs.field(default=0.1)
    size: float = attrs.field(default=0.03)
    arrow_len: float = attrs.field(default=1.0)
    height: float = attrs.field(default=0.5)

    def _update_arrow(self, cmd_x: float, cmd_y: float) -> None:
        self.geom = mujoco.mjtGeom.mjGEOM_ARROW  # pyright: ignore[reportAttributeAccessIssue]
        mag = (cmd_x * cmd_x + cmd_y * cmd_y) ** 0.5
        cmd_x, cmd_y = cmd_x / mag, cmd_y / mag
        self.orientation = self.quat_from_direction((cmd_x, cmd_y, 0.0))
        self.scale = (self.size, self.size, self.arrow_len * mag)

    def _update_circle(self) -> None:
        self.geom = mujoco.mjtGeom.mjGEOM_SPHERE  # pyright: ignore[reportAttributeAccessIssue]
        self.scale = (self.size, self.size, self.size)

    def _update_cylinder(self) -> None:
        self.geom = mujoco.mjtGeom.mjGEOM_CYLINDER  # pyright: ignore[reportAttributeAccessIssue]
        self.scale = (self.size, self.size, self.size)
        self.orientation = self.quat_from_direction((0.0, 0.0, 1.0))

    def update(self, trajectory: Trajectory) -> None:
        """Visualizes the joystick command target position and orientation."""
        cmd: JoystickCommandValue = trajectory.command[self.command_name]
        self._update_for(cmd, trajectory)

    def _update_for(self, cmd: JoystickCommandValue, trajectory: Trajectory) -> None:
        """Update the marker position and rotation."""
        cmd_idx, cmd_vel = cmd.command.argmax().item(), cmd.vels

        # Updates the marker color.
        r, g, b = [
            (1.0, 1.0, 1.0),  # Stand still (white)
            (0.0, 1.0, 0.0),  # Walk forward (green)
            (0.0, 0.0, 1.0),  # Run forward (blue)
            (1.0, 0.0, 0.0),  # Walk backward (red)
            (1.0, 0.0, 1.0),  # Turn left (purple)
            (0.0, 0.0, 0.0),  # Turn right (black)
            (0.0, 1.0, 1.0),  # Strafe left (cyan)
            (1.0, 1.0, 0.0),  # Strafe right (yellow)
        ][cmd_idx]
        self.rgba = (r, g, b, 1.0)

        cmd_x, cmd_y = cmd_vel[..., 0], cmd_vel[..., 1]

        # Gets the robot's current yaw.
        quat = trajectory.qpos[..., 3:7]
        cur_yaw = xax.quat_to_yaw(quat)

        # Rotates the command X and Y velocities to the robot's current yaw.
        cmd_x_rot = cmd_x * jnp.cos(cur_yaw) - cmd_y * jnp.sin(cur_yaw)
        cmd_y_rot = cmd_x * jnp.sin(cur_yaw) + cmd_y * jnp.cos(cur_yaw)

        self.pos = (0, 0, self.height)

        match cmd_idx:
            case 0:
                self._update_circle()
            case 1 | 2 | 3 | 6 | 7:
                self._update_arrow(cmd_x_rot.item(), cmd_y_rot.item())
            case 4 | 5:
                self._update_cylinder()
            case _:
                pass

    @classmethod
    def get(
        cls,
        command_name: str,
        radius: float = 0.05,
        size: float = 0.03,
        arrow_len: float = 0.25,
        rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.8),
        height: float = 0.5,
    ) -> Self:
        return cls(
            command_name=command_name,
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_SPHERE,  # pyright: ignore[reportAttributeAccessIssue]
            scale=(radius, radius, radius),
            size=size,
            arrow_len=arrow_len,
            radius=radius,
            rgba=rgba,
            height=height,
            track_x=True,
            track_y=True,
            track_z=True,
            track_rotation=False,
        )


@attrs.define(frozen=True, kw_only=True)
class JoystickCommand(Command):
    """Provides joystick-like controls for the robot.

    Commands are encoded as one-hot vectors. This command should be paired with
    the JoystickReward. The robot is expected to always start aligned with the
    forward X direction of the world frame - for example, we reward the robot
    for moving forward along the global X axis.

    Command mapping:

        0 = stand still
        1 = walk forward
        2 = run forward
        3 = walk backward
        4 = turn left
        5 = turn right
        6 = strafe left
        7 = strafe right

    The joystick command is composed of two parts:

    - A one-hot vector of length 8, which is the command to take.
    - A 3 dimensional vector representing the target X, Y and yaw at some time.
    """

    walk_speed: float = attrs.field()
    run_speed: float = attrs.field()
    strafe_speed: float = attrs.field()
    rotation_speed: float = attrs.field()
    sample_probs: tuple[float, float, float, float, float, float, float, float] = attrs.field(
        default=(0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
        validator=sample_probs_validator,
    )
    marker_z_offset: float = attrs.field(default=0.5)
    switch_prob: float = attrs.field(default=0.005)

    def _get_vel_tgts(self, physics_data: PhysicsData, command: Array) -> Array:
        # Gets the target X, Y, and Yaw targets.
        cmd_tgts = jnp.array(
            [
                [0.0, 0.0, 0.0],  # Stand still
                [self.walk_speed, 0.0, 0.0],  # Walk forward
                [self.run_speed, 0.0, 0.0],  # Run forward
                [-self.walk_speed, 0.0, 0.0],  # Walk backward
                [0.0, 0.0, self.rotation_speed],  # Turn left
                [0.0, 0.0, -self.rotation_speed],  # Turn right
                [0.0, self.strafe_speed, 0.0],  # Strafe left
                [0.0, -self.strafe_speed, 0.0],  # Strafe right
            ]
        )

        cmd_tgt = cmd_tgts[command]
        cmd_x, cmd_y, cmd_yaw = cmd_tgt[..., 0], cmd_tgt[..., 1], cmd_tgt[..., 2]

        return jnp.stack([cmd_x, cmd_y, cmd_yaw], axis=-1)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> JoystickCommandValue:
        command = jax.random.choice(rng, jnp.arange(len(self.sample_probs)), p=jnp.array(self.sample_probs))
        command_ohe = jax.nn.one_hot(command, num_classes=8)
        vel_tgts = self._get_vel_tgts(physics_data, command)
        return JoystickCommandValue(
            command=command_ohe,
            vels=vel_tgts,
        )

    def __call__(
        self,
        prev_command: JoystickCommandValue,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> JoystickCommandValue:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return JoystickCommandValue(
            command=jnp.where(switch_mask, new_commands.command, prev_command.command),
            vels=jnp.where(switch_mask, new_commands.vels, prev_command.vels),
        )

    def get_markers(self) -> Collection[Marker]:
        return [JoystickCommandMarker.get(self.command_name, height=self.marker_z_offset)]


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
        cmd = trajectory.command[self.command_name]
        vx, vy = float(cmd[0]), float(cmd[1])
        speed = (vx * vx + vy * vy) ** 0.5

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

    x_range: tuple[float, float] = attrs.field()
    y_range: tuple[float, float] = attrs.field()
    x_zero_prob: float = attrs.field(default=0.0)
    y_zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)
    vis_height: float = attrs.field(default=1.0)
    vis_scale: float = attrs.field(default=0.05)

    def initial_command(self, physics_data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        rng_x, rng_y, rng_zero_x, rng_zero_y = jax.random.split(rng, 4)
        (xmin, xmax), (ymin, ymax) = self.x_range, self.y_range
        x = jax.random.uniform(rng_x, (1,), minval=xmin, maxval=xmax)
        y = jax.random.uniform(rng_y, (1,), minval=ymin, maxval=ymax)
        x_zero_mask = jax.random.bernoulli(rng_zero_x, self.x_zero_prob)
        y_zero_mask = jax.random.bernoulli(rng_zero_y, self.y_zero_prob)
        return jnp.concatenate(
            [
                jnp.where(x_zero_mask, 0.0, x),
                jnp.where(y_zero_mask, 0.0, y),
            ]
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

    def get_markers(self) -> Collection[Marker]:
        return [
            LinearVelocityCommandMarker.get(
                command_name=self.command_name,
                height=0.5,
            )
        ]


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
class PositionCommand(Command):
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

    def get_markers(self) -> Collection[Marker]:
        return [
            PositionCommandMarker.get(
                self.command_name,
                self.base_name,
                self.vis_radius,
                self.vis_color,
            )
        ]

    def get_name(self) -> str:
        name = super().get_name()
        if self.unique_name is not None:
            name = f"{self.unique_name}_{name}"
        return name

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

    def set_moving(self, moving_flag: bool | Array, command: SinusoidalGaitCommandValue) -> SinusoidalGaitCommandValue:
        # Can use this method to toggle the robot to be moving or not moving
        # according to some requirements of the command. Turning off the moving
        # flag just disables the reward and resets the phase.
        moving_flag_arr = jnp.full_like(command.moving_flag, moving_flag)
        phase_arr = jnp.where(moving_flag_arr, command.phase, 0.0)
        return SinusoidalGaitCommandValue(
            moving_flag=moving_flag_arr,
            phase=phase_arr,
            height=self._get_height(phase_arr),
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

        return SinusoidalGaitCommandValue(
            moving_flag=moving_flag,
            phase=new_phase,
            height=self._get_height(new_phase),
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EasyJoystickCommandValue:
    gait: SinusoidalGaitCommandValue
    joystick: JoystickCommandValue


@attrs.define(kw_only=True)
class EasyJoystickCommandMarker(JoystickCommandMarker):
    command_name: str = attrs.field()

    def update(self, trajectory: Trajectory) -> None:
        cmd: EasyJoystickCommandValue = trajectory.command[self.command_name]
        self._update_for(cmd.joystick, trajectory)


@attrs.define(frozen=True, kw_only=True)
class EasyJoystickCommand(Command):
    gait: SinusoidalGaitCommand = attrs.field()
    joystick: JoystickCommand = attrs.field()

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> EasyJoystickCommandValue:
        return EasyJoystickCommandValue(
            gait=self.gait.initial_command(physics_data, curriculum_level, rng),
            joystick=self.joystick.initial_command(physics_data, curriculum_level, rng),
        )

    def __call__(
        self,
        prev_command: EasyJoystickCommandValue,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> EasyJoystickCommandValue:
        joystick_command = self.joystick(prev_command.joystick, physics_data, curriculum_level, rng)
        gait_command = self.gait(prev_command.gait, physics_data, curriculum_level, rng)
        gait_command = self.gait.set_moving(joystick_command.command.argmax(axis=-1) != 0, gait_command)
        return EasyJoystickCommandValue(
            gait=gait_command,
            joystick=joystick_command,
        )

    def get_markers(self) -> Collection[Marker]:
        return [EasyJoystickCommandMarker.get(self.command_name, height=self.joystick.marker_z_offset)]


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
