"""Defines the base command class."""

__all__ = [
    "Command",
    "LinearVelocityCommand",
    "AngularVelocityCommand",
]

import functools
from abc import ABC, abstractmethod
from typing import Collection

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.utils.visualization import VelocityArrow, Visualization


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

    def update_scene(self, command: Array) -> Collection[Visualization]:
        """Updates the scene with elements from the command.

        Args:
            command: The command to update the scene with.

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

    x_range: tuple[float, float] = attrs.field(default=(-1.0, 1.0))
    y_range: tuple[float, float] = attrs.field(default=(-1.0, 1.0))
    switch_prob: float = attrs.field(default=0.0)
    zero_prob: float = attrs.field(default=0.0)
    vis_height: float = attrs.field(default=2.0)

    def initial_command(self, rng: PRNGKeyArray) -> Array:
        rng_x, rng_y, rng_zero = jax.random.split(rng, 3)
        (xmin, xmax), (ymin, ymax) = self.x_range, self.y_range
        x = jax.random.uniform(rng_x, (), minval=xmin, maxval=xmax)
        y = jax.random.uniform(rng_y, (), minval=ymin, maxval=ymax)
        zero_mask = jax.random.bernoulli(rng_zero, self.zero_prob)
        # TODO this is not consistent with other commands shape
        cmd = jnp.array([x, y])
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

    def __call__(self, prev_command: Array, time: Array, rng: PRNGKeyArray) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)

    def update_scene(self, command: Array) -> Collection[Visualization]:
        return [
            VelocityArrow(
                velocity=float(command[0]),
                base_pos=(0.0, 0.0, self.vis_height),
                scale=0.1,
                rgba=(1.0, 0.0, 0.0, 0.8),
                direction=(1.0, 0.0, 0.0),
                label=f"X: {command[0]:.2f}",
            ),
            VelocityArrow(
                velocity=float(command[1]),
                base_pos=(0.0, 0.0, self.vis_height + 0.2),
                scale=0.1,
                rgba=(0.0, 1.0, 0.0, 0.8),
                direction=(0.0, 1.0, 0.0),
                label=f"Y: {command[1]:.2f}",
            ),
        ]


@attrs.define(frozen=True)
class AngularVelocityCommand(Command):
    """Command to turn the robot."""

    scale: float = attrs.field(default=1.0)
    switch_prob: float = attrs.field(default=0.0)
    zero_prob: float = attrs.field(default=0.0)

    def initial_command(self, rng: PRNGKeyArray) -> Array:
        """Returns (1,) array with angular velocity."""
        rng_a, rng_b = jax.random.split(rng)
        zero_mask = jax.random.bernoulli(rng_a, self.zero_prob)
        cmd = jax.random.uniform(rng_b, (1,), minval=-self.scale, maxval=self.scale)
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

    def __call__(self, prev_command: Array, time: Array, rng: PRNGKeyArray) -> Array:
        """Get the command to perform: returns (command_dim,) array."""
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)
