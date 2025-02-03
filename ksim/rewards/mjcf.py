"""Defines reward functions to use with MJCF environments."""

import jax.numpy as jnp
from brax.mjx.base import State as MjxState

from ksim.rewards.base import Reward


class LinearVelocityZPenalty(Reward):
    """Penalty for how fast the robot is moving in the z-direction."""

    def __call__(self, state: MjxState) -> jnp.ndarray:
        lin_vel_z = state.qvel[..., 2]
        return jnp.square(lin_vel_z)
