"""Defines the base command class."""

__all__ = [
    "Command",
    "LinearVelocityCommand",
    "AngularVelocityCommand",
    "LinearVelocityStepCommand",
    "AngularVelocityStepCommand",
]

import functools
from abc import ABC, abstractmethod
from typing import Collection, Literal, Self

import attrs
import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.types import Trajectory
from ksim.vis import Marker


@attrs.define(frozen=True)
class Command(ABC):
    """Base class for commands."""

    @abstractmethod
    def initial_command(self, rng: PRNGKeyArray) -> Array:
        """Returns the initial command.

        Args:
            rng: The random number generator.

        Returns:
            The initial command, with shape (command_dim).
        """

    @abstractmethod
    def __call__(self, prev_command: Array, time: Array, rng: PRNGKeyArray) -> Array:
        """Updates the command.

        Args:
            prev_command: The previous command.
            time: The current time.
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


VelocityAxis = Literal["x", "y"]


@attrs.define(kw_only=True)
class LinearVelocityArrow(Marker):
    command_name: str = attrs.field()
    axis: VelocityAxis = attrs.field()
    vis_height: float = attrs.field()
    vis_scale: float = attrs.field()

    @property
    def command_id(self) -> int:
        return {"x": 0, "y": 1}[self.axis]

    def update(self, trajectory: Trajectory) -> None:
        value = float(trajectory.command[self.command_name][self.command_id])
        self.scale = (self.vis_scale, self.vis_scale, value * 5.0 * self.vis_scale)
        match self.axis:
            case "x":
                self.pos = ((self.vis_scale if value > 0 else -self.vis_scale) * 2.0, 0.0, self.vis_height)
            case "y":
                self.pos = (0.0, (self.vis_scale if value > 0 else -self.vis_scale) * 2.0, self.vis_height)

    @classmethod
    def get(cls, command_name: str, axis: VelocityAxis, vis_height: float, vis_scale: float) -> Self:
        match axis:
            case "x":
                return cls(
                    command_name=command_name,
                    axis=axis,
                    geom=mujoco.mjtGeom.mjGEOM_ARROW,
                    orientation=cls.quat_from_direction((1.0, 0.0, 0.0)),
                    rgba=(1.0, 0.0, 0.0, 0.8),
                    target_type="root",
                    vis_height=vis_height,
                    vis_scale=vis_scale,
                )

            case "y":
                return cls(
                    command_name=command_name,
                    axis=axis,
                    geom=mujoco.mjtGeom.mjGEOM_ARROW,
                    orientation=cls.quat_from_direction((0.0, 1.0, 0.0)),
                    rgba=(0.0, 1.0, 0.0, 0.8),
                    target_type="root",
                    vis_height=vis_height,
                    vis_scale=vis_scale,
                )

            case _:
                raise ValueError(f"Invalid axis: {axis}")


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

    def initial_command(self, rng: PRNGKeyArray) -> Array:
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

    def __call__(self, prev_command: Array, time: Array, rng: PRNGKeyArray) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)

    def get_markers(self) -> Collection[Marker]:
        return [
            LinearVelocityArrow.get(self.command_name, "x", self.vis_height, self.vis_scale),
            LinearVelocityArrow.get(self.command_name, "y", self.vis_height, self.vis_scale),
        ]


@attrs.define(frozen=True)
class AngularVelocityCommand(Command):
    """Command to turn the robot."""

    scale: float = attrs.field()
    zero_prob: float = attrs.field(default=0.0)

    def initial_command(self, rng: PRNGKeyArray) -> Array:
        """Returns (1,) array with angular velocity."""
        rng_a, rng_b = jax.random.split(rng)
        zero_mask = jax.random.bernoulli(rng_a, self.zero_prob)
        cmd = jax.random.uniform(rng_b, (1,), minval=-self.scale, maxval=self.scale)
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

    def __call__(self, prev_command: Array, time: Array, rng: PRNGKeyArray) -> Array:
        return prev_command


@attrs.define(frozen=True)
class LinearVelocityStepCommand(Command):
    """This is the same as LinearVelocityCommand, but it is discrete."""

    x_range: tuple[float, float] = attrs.field()
    y_range: tuple[float, float] = attrs.field()
    x_fwd_prob: float = attrs.field()
    y_fwd_prob: float = attrs.field()
    x_zero_prob: float = attrs.field(default=0.0)
    y_zero_prob: float = attrs.field(default=0.0)
    vis_height: float = attrs.field(default=1.0)
    vis_scale: float = attrs.field(default=0.05)

    def initial_command(self, rng: PRNGKeyArray) -> Array:
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

    def __call__(self, prev_command: Array, time: Array, rng: PRNGKeyArray) -> Array:
        return prev_command

    def get_markers(self) -> Collection[Marker]:
        return [
            LinearVelocityArrow.get(self.command_name, "x", self.vis_height, self.vis_scale),
            LinearVelocityArrow.get(self.command_name, "y", self.vis_height, self.vis_scale),
        ]


@attrs.define(frozen=True)
class AngularVelocityStepCommand(Command):
    """This is the same as AngularVelocityCommand, but it is discrete."""

    scale: float = attrs.field()
    prob: float = attrs.field(default=0.5)
    zero_prob: float = attrs.field(default=0.0)

    def initial_command(self, rng: PRNGKeyArray) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        cmd = (jax.random.bernoulli(rng_a, self.prob, (1,)) * 2 - 1) * self.scale
        zero_mask = jax.random.bernoulli(rng_b, self.zero_prob)
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

    def __call__(self, prev_command: Array, time: Array, rng: PRNGKeyArray) -> Array:
        return prev_command
