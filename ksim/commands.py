"""Defines the base command class."""

__all__ = [
    "Command",
    "LinearVelocityCommand",
    "AngularVelocityCommand",
]

import functools
from abc import ABC, abstractmethod

import attrs
import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray


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

    def update_scene(self, scene: mujoco.MjvScene, command: Array) -> None:
        """Updates the scene with elements from the command.

        Args:
            scene: The scene to update.
            command: The command to update the scene with.
        """

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

    x_scale: float = attrs.field(default=1.0)
    y_scale: float = attrs.field(default=1.0)
    switch_prob: float = attrs.field(default=0.0)
    zero_prob: float = attrs.field(default=0.0)

    def initial_command(self, rng: PRNGKeyArray) -> Array:
        rng_x, rng_y, rng_zero = jax.random.split(rng, 3)
        x = jax.random.uniform(rng_x, (), minval=-self.x_scale, maxval=self.x_scale)
        y = jax.random.uniform(rng_y, (), minval=-self.y_scale, maxval=self.y_scale)
        zero_mask = jax.random.bernoulli(rng_zero, self.zero_prob)
        # TODO this is not consistent with other commands shape
        cmd = jnp.array([x, y])
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

    def __call__(self, prev_command: Array, time: Array, rng: PRNGKeyArray) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)

    def update_scene(self, scene: mujoco.MjvScene, command: Array) -> None:
        # TODO: Implement this so that we add an arrow to the scene pointing
        # in the direction of the command.
        pass


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
