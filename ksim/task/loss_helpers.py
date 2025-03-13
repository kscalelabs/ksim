"""Helpers useful for computing loss functions."""

import jax
import jax.numpy as jnp
from jax import Array


def compute_returns(rewards: Array, dones: Array, gamma: float) -> Array:
    """Calculate returns from rewards and dones.

    Dones are a mask of 0s and 1s, where 1s indicate the end of an episode.
    Gamma is the discount factor.
    """

    # calculating returns separately using gamma, decoupling w/ value targets
    def scan_fn(returns_t_plus_1: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        """Scanning this computes the returns in reverse order."""
        reward, mask = x
        returns = reward + gamma * mask * returns_t_plus_1
        return returns, returns

    _, returns = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(rewards[-1]),
        (rewards, dones),
        reverse=True,
    )
    return returns
