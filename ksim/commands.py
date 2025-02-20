"""Defines the base command class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from jaxtyping import PRNGKeyArray

from ksim.utils.data import BuilderData


class Command(eqx.Module, ABC):
    @abstractmethod
    def __call__(self, rng: PRNGKeyArray) -> jnp.ndarray:
        """Gets the command to perform."""

    @eqx.filter_jit
    def update(self, prev_command: jnp.ndarray, rng: PRNGKeyArray, time: jnp.ndarray) -> jnp.ndarray:
        """Optionally updates the command to a new command."""
        return prev_command

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def command_name(self) -> str:
        return self.get_name()


T = TypeVar("T", bound=Command)


class CommandBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, data: BuilderData) -> T:
        """Builds a command from a MuJoCo model.

        Args:
            data: The data to build the command from.

        Returns:
            A command that can be applied to a state.
        """


class LinearVelocityCommand(Command):
    """Command to move the robot in a straight line.

    By convention, X is forward and Y is left. The switching probability is the
    probability of resampling the command at each step. The zero probability is
    the probability of the command being zero - this can be used to turn off
    any command.
    """

    x_scale: float = eqx.field(default=1.0, static=True)
    y_scale: float = eqx.field(default=1.0, static=True)
    switch_prob: float = eqx.field(default=0.0, static=True)
    zero_prob: float = eqx.field(default=0.0, static=True)

    def __init__(
        self,
        *,
        x_scale: float = 1.0,
        y_scale: float = 1.0,
        switch_prob: float = 0.0,
        zero_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.x_scale = x_scale
        self.y_scale = y_scale
        self.switch_prob = switch_prob
        self.zero_prob = zero_prob

    def _get_command(self, rng: PRNGKeyArray) -> jnp.ndarray:
        x = jax.random.uniform(rng, (1,), minval=-self.x_scale, maxval=self.x_scale)
        y = jax.random.uniform(rng, (1,), minval=-self.y_scale, maxval=self.y_scale)
        return jnp.concatenate([x, y], axis=-1)

    @eqx.filter_jit
    def __call__(self, rng: PRNGKeyArray) -> jnp.ndarray:
        rng_a, rng_b = jax.random.split(rng)
        return jax.lax.cond(
            jax.random.bernoulli(rng_a, self.zero_prob),
            lambda: jnp.zeros((2,)),
            lambda: self._get_command(rng_b),
        )

    @eqx.filter_jit
    def update(self, prev_command: jnp.ndarray, rng: PRNGKeyArray, time: jnp.ndarray) -> jnp.ndarray:
        rng_a, rng_b = jax.random.split(rng)
        return jax.lax.cond(
            jax.random.bernoulli(rng_a, self.switch_prob),
            lambda: self(rng_b),
            lambda: prev_command,
        )


class AngularVelocityCommand(Command):
    """Command to turn the robot."""

    scale: float
    switch_prob: float
    zero_prob: float

    def __init__(
        self,
        *,
        scale: float = 1.0,
        switch_prob: float = 0.0,
        zero_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.scale = scale
        self.switch_prob = switch_prob
        self.zero_prob = zero_prob

    def _get_command(self, rng: PRNGKeyArray) -> jnp.ndarray:
        return jax.random.uniform(rng, (1,), minval=-self.scale, maxval=self.scale)

    @eqx.filter_jit
    def __call__(self, rng: PRNGKeyArray) -> jnp.ndarray:
        rng_a, rng_b = jax.random.split(rng)
        return jax.lax.cond(
            jax.random.bernoulli(rng_a, self.zero_prob),
            lambda: jnp.zeros((1,)),
            lambda: self._get_command(rng_b),
        )

    @eqx.filter_jit
    def update(self, prev_command: jnp.ndarray, rng: PRNGKeyArray, time: jnp.ndarray) -> jnp.ndarray:
        rng_a, rng_b = jax.random.split(rng)
        return jax.lax.cond(
            jax.random.bernoulli(rng_a, self.switch_prob),
            lambda: self(rng_b),
            lambda: prev_command,
        )
