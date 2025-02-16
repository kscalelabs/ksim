"""Defines some useful command functions for MJCF environments."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from ksim.commands.base import Command


class LinearVelocityCommand(Command):
    """Command to move the robot in a straight line."""

    x_scale: float
    y_scale: float

    def __init__(self, x_scale: float = 1.0, y_scale: float = 1.0) -> None:
        super().__init__()

        self.x_scale = x_scale
        self.y_scale = y_scale

    @eqx.filter_jit
    def __call__(self, rng: PRNGKeyArray) -> jnp.ndarray:
        x = jax.random.uniform(rng, (1,), minval=-self.x_scale, maxval=self.x_scale)
        y = jax.random.uniform(rng, (1,), minval=-self.y_scale, maxval=self.y_scale)
        return jnp.concatenate([x, y], axis=-1)


class AngularVelocityCommand(Command):
    """Command to turn the robot."""

    scale: float

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()

        self.scale = scale

    @eqx.filter_jit
    def __call__(self, rng: PRNGKeyArray) -> jnp.ndarray:
        return jax.random.uniform(rng, (1,), minval=-self.scale, maxval=self.scale)
