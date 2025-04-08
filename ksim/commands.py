"""Defines the base command class."""

__all__ = [
    "Command",
    "LinearVelocityCommand",
    "AngularVelocityCommand",
    "LinearVelocityStepCommand",
    "AngularVelocityStepCommand",
    "PositionCommand",
]

import functools
from abc import ABC, abstractmethod
from typing import Collection, Self

import attrs
import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.types import PhysicsData, PhysicsModel, Trajectory
from ksim.utils.mujoco import get_body_data_idx_from_name
from ksim.utils.types import CartesianIndex, dimension_index_validator
from ksim.vis import Marker


@attrs.define(frozen=True, kw_only=True)
class Command(ABC):
    """Base class for commands."""

    @abstractmethod
    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
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
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
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


@attrs.define(frozen=True)
class LinearVelocityCommand(Command):
    """Command to move the robot in a straight line.

    By convention, X is forward and Y is left. The switching probability is the
    probability of resampling the command at each step. The zero probability is
    the probability of the command being zero - this can be used to turn off
    any command.
    """

    range: tuple[float, float] = attrs.field()
    index: CartesianIndex | None = attrs.field(default=None, validator=dimension_index_validator)
    zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)
    vis_height: float = attrs.field(default=1.0)
    vis_scale: float = attrs.field(default=0.05)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng, rng_zero = jax.random.split(rng)
        minval, maxval = self.range
        value = jax.random.uniform(rng, (1,), minval=minval, maxval=maxval)
        zero_mask = jax.random.bernoulli(rng_zero, self.zero_prob)
        return jnp.where(zero_mask, 0.0, value)

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

    def get_name(self) -> str:
        return f"{super().get_name()}{'' if self.index is None else f'_{self.index}'}"


@attrs.define(frozen=True)
class AngularVelocityCommand(Command):
    """Command to turn the robot."""

    scale: float = attrs.field()
    index: CartesianIndex | None = attrs.field(default=None, validator=dimension_index_validator)
    zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        """Returns (1,) array with angular velocity."""
        rng_a, rng_b = jax.random.split(rng)
        zero_mask = jax.random.bernoulli(rng_a, self.zero_prob)
        cmd = jax.random.uniform(rng_b, (1,), minval=-self.scale, maxval=self.scale)
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

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

    def get_name(self) -> str:
        return f"{super().get_name()}{'' if self.index is None else f'_{self.index}'}"


@attrs.define(frozen=True)
class LinearVelocityStepCommand(Command):
    """This is the same as LinearVelocityCommand, but it is discrete."""

    x_range: tuple[float, float] = attrs.field()
    y_range: tuple[float, float] = attrs.field()
    x_fwd_prob: float = attrs.field()
    y_fwd_prob: float = attrs.field()
    x_zero_prob: float = attrs.field(default=0.0)
    y_zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)
    vis_height: float = attrs.field(default=1.0)
    vis_scale: float = attrs.field(default=0.05)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_x, rng_y, rng_zero_x, rng_zero_y = jax.random.split(rng, 4)
        (xmin, xmax), (ymin, ymax) = self.x_range, self.y_range
        x = jax.random.bernoulli(rng_x, self.x_fwd_prob, (1,)) * (xmax - xmin) + xmin
        y = jax.random.bernoulli(rng_y, self.y_fwd_prob, (1,)) * (ymax - ymin) + ymin
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


@attrs.define(frozen=True)
class AngularVelocityStepCommand(Command):
    """This is the same as AngularVelocityCommand, but it is discrete."""

    scale: float = attrs.field()
    prob: float = attrs.field(default=0.5)
    zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        cmd = (jax.random.bernoulli(rng_a, self.prob, (1,)) * 2 - 1) * self.scale
        zero_mask = jax.random.bernoulli(rng_b, self.zero_prob)
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

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


@attrs.define(kw_only=True)
class PositionCommandMarker(Marker):
    command_name: str = attrs.field()

    def update(self, trajectory: Trajectory) -> None:
        """Update the marker position and rotation."""
        self.pos = trajectory.command[self.command_name][..., :3]

    @classmethod
    def get(
        cls,
        command_name: str,
        radius: float,
        rgba: tuple[float, float, float, float],
    ) -> Self:
        return cls(
            command_name=command_name,
            target_name=None,
            geom=mujoco.mjtGeom.mjGEOM_SPHERE,
            scale=(radius, radius, radius),
            rgba=rgba,
        )


@attrs.define(frozen=True)
class PositionCommand(Command):
    """Samples a target xyz position within a bounding box relative to a base body.

    The bounding box is defined by min and max coordinates relative to the base body.
    The target will smoothly transition between points within this box.
    """

    box_min: tuple[float, float, float] = attrs.field()
    box_max: tuple[float, float, float] = attrs.field()
    base_id: int | None = attrs.field(default=None)
    vis_radius: float = attrs.field(default=0.05)
    vis_color: tuple[float, float, float, float] = attrs.field(default=(1.0, 0.0, 0.0, 0.8))
    min_speed: float = attrs.field(default=0.5)
    max_speed: float = attrs.field(default=3.0)
    unique_name: str | None = attrs.field(default=None)
    eps: float = attrs.field(default=1e-3)

    def _sample_box(self, rng: PRNGKeyArray, physics_data: PhysicsData) -> Array:
        # Sample uniformly within the box
        rng_x, rng_y, rng_z = jax.random.split(rng, 3)
        min_x, min_y, min_z = self.box_min
        max_x, max_y, max_z = self.box_max

        # Scale the box size based on curriculum level
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2

        # Sample and scale around center
        x = center_x + (jax.random.uniform(rng_x, ()) - 0.5) * (max_x - min_x)
        y = center_y + (jax.random.uniform(rng_y, ()) - 0.5) * (max_y - min_y)
        z = center_z + (jax.random.uniform(rng_z, ()) - 0.5) * (max_z - min_z)

        xyz = jnp.array([x, y, z])

        if self.base_id is not None:
            xyz = xyz + physics_data.xpos[self.base_id]

        return xyz

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        # Sample initial target and speed
        rng_target, rng_speed = jax.random.split(rng)
        target = self._sample_box(rng_target, physics_data)
        speed = jax.random.uniform(rng_speed, (), minval=self.min_speed, maxval=self.max_speed)

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
        rng_a, rng_b = jax.random.split(rng)
        reached_target = distance < self.eps

        # Sample new target and speed if reached
        new_target = self._sample_box(rng_a, physics_data)
        new_speed = jax.random.uniform(rng_b, (), minval=self.min_speed, maxval=self.max_speed)

        # Update target and speed if reached
        target = jnp.where(reached_target, new_target, target)
        speed = jnp.where(reached_target, new_speed, speed)

        # Calculate step size based on speed and timestep
        dt = physics_data.model.opt.timestep
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
        return [PositionCommandMarker.get(self.command_name, self.vis_radius, self.vis_color)]

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
        base_body_name: str | None = None,
        vis_radius: float = 0.05,
        vis_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.8),
        unique_name: str | None = None,
        min_speed: float = 0.5,
        max_speed: float = 3.0,
    ) -> Self:
        base_id = None if base_body_name is None else get_body_data_idx_from_name(model, base_body_name)
        return cls(
            base_id=base_id,
            box_min=box_min,
            box_max=box_max,
            vis_radius=vis_radius,
            vis_color=vis_color,
            unique_name=unique_name,
            min_speed=min_speed,
            max_speed=max_speed,
        )
