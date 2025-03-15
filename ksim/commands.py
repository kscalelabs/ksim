"""Defines the base command class."""

import functools
from abc import ABC, abstractmethod
from typing import get_args

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.types import CmdType


@attrs.define(frozen=True)
class Command(ABC):
    """Base class for commands."""

    cmd_type: CmdType = attrs.field(default="vector")

    def __attrs_post_init__(self) -> None:
        """Ensuring protected attributes are not present in the class name."""
        cmd_types = get_args(CmdType)
        name = self.__class__.__name__
        if "_" in name:
            raise ValueError("Class name cannot contain underscores")
        for cmd_type in cmd_types:
            if f"{cmd_type}" in name.lower():
                raise ValueError(f"Class name cannot contain protected string: {cmd_type}")

    @abstractmethod
    def initial_command(self, rng: PRNGKeyArray) -> Array: ...

    @abstractmethod
    def __call__(self, prev_command: Array, time: Array, rng: PRNGKeyArray) -> Array: ...

    def get_name(self) -> str:
        """Get the name of the command."""
        name = xax.camelcase_to_snakecase(self.__class__.__name__)
        name += f"_{self.cmd_type}"
        return name

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

    cmd_type: CmdType = "vector"

    x_scale: float = attrs.field(default=1.0)
    y_scale: float = attrs.field(default=1.0)
    switch_prob: float = attrs.field(default=0.0)
    zero_prob: float = attrs.field(default=0.0)

    def initial_command(self, rng: PRNGKeyArray) -> Array:
        """Returns (2,) array with x and y velocities."""
        rng_x, rng_y, rng_zero = jax.random.split(rng, 3)
        x = jax.random.uniform(rng_x, (), minval=-self.x_scale, maxval=self.x_scale)
        y = jax.random.uniform(rng_y, (), minval=-self.y_scale, maxval=self.y_scale)
        zero_mask = jax.random.bernoulli(rng_zero, self.zero_prob)
        cmd = jnp.array([x, y])
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

    def __call__(self, prev_command: Array | None, time: Array, rng: PRNGKeyArray) -> Array:
        """Get the command to perform: returns (command_dim,) array."""
        prev_command = jax.lax.cond(
            prev_command is None,
            lambda: jnp.zeros(2),
            lambda: prev_command,
        )
        assert isinstance(prev_command, Array)

        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)


@attrs.define(frozen=True)
class AngularVelocityCommand(Command):
    """Command to turn the robot."""

    cmd_type: CmdType = "vector"

    scale: float = attrs.field(default=1.0)
    switch_prob: float = attrs.field(default=0.0)
    zero_prob: float = attrs.field(default=0.0)

    def initial_command(self, rng: PRNGKeyArray) -> Array:
        """Returns (1,) array with angular velocity."""
        rng_a, rng_b = jax.random.split(rng)
        zero_mask = jax.random.bernoulli(rng_a, self.zero_prob)
        cmd = jax.random.uniform(rng_b, (1,), minval=-self.scale, maxval=self.scale)
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

    def __call__(self, prev_command: Array | None, time: Array, rng: PRNGKeyArray) -> Array:
        """Get the command to perform: returns (command_dim,) array."""
        prev_command = jax.lax.cond(
            prev_command is None,
            lambda: jnp.zeros(1),
            lambda: prev_command,
        )
        assert isinstance(prev_command, Array)

        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)
