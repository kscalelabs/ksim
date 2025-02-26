"""Defines the base command class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.utils.data import BuilderData


@attrs.define(frozen=True)
class Command(ABC):
    """Base class for commands."""

    @abstractmethod
    def __call__(self, rng: PRNGKeyArray, time: Array) -> Array:
        """Gets the command to perform: returns (command_dim,) array."""

    def update(self, prev_command: Array, rng: PRNGKeyArray, time: Array) -> Array:
        """Optionally updates the command to a new command."""
        return prev_command

    def get_name(self) -> str:
        """Get the name of the command."""
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def command_name(self) -> str:
        return self.get_name()


T = TypeVar("T", bound=Command)


class CommandBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, data: BuilderData) -> T:
        """Builds a command from a MuJoCo model."""


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

    def _get_command(self, rng: PRNGKeyArray) -> Array:
        """Returns (2,) array with x and y velocities."""
        rng_x, rng_y = jax.random.split(rng)
        x = jax.random.uniform(rng_x, (), minval=-self.x_scale, maxval=self.x_scale)
        y = jax.random.uniform(rng_y, (), minval=-self.y_scale, maxval=self.y_scale)
        return jnp.array([x, y])

    def __call__(self, rng: PRNGKeyArray, time: Array) -> Array:
        """Get the command to perform: returns (command_dim,) array."""
        rng_a, rng_b = jax.random.split(rng)
        zero_mask = jax.random.bernoulli(rng_a, self.zero_prob)
        commands = self._get_command(rng_b)
        return jnp.where(zero_mask, jnp.zeros_like(commands), commands)

    def update(self, prev_command: Array, rng: PRNGKeyArray, time: Array) -> Array:
        """Updates command: returns (command_dim,) array."""
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self(rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)


@attrs.define(frozen=True)
class AngularVelocityCommand(Command):
    """Command to turn the robot."""

    scale: float = attrs.field(default=1.0)
    switch_prob: float = attrs.field(default=0.0)
    zero_prob: float = attrs.field(default=0.0)

    def _get_command(self, rng: PRNGKeyArray) -> Array:
        """Returns (1,) array with angular velocity."""
        return jax.random.uniform(rng, (1,), minval=-self.scale, maxval=self.scale)

    def __call__(self, rng: PRNGKeyArray, time: Array) -> Array:
        """Get the command to perform: returns (command_dim,) array."""
        rng_a, rng_b = jax.random.split(rng)
        zero_mask = jax.random.bernoulli(rng_a, self.zero_prob)
        commands = self._get_command(rng_b)
        return jnp.where(zero_mask, jnp.zeros_like(commands), commands)

    def update(self, prev_command: Array, rng: PRNGKeyArray, time: Array) -> Array:
        """Updates command: returns (command_dim,) array."""
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self(rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)
