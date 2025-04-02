"""Tests custom distribution operations."""

import distrax
import jax.numpy as jnp

import ksim


def test_asymmetric_bijector() -> None:
    dist = distrax.Normal(0.0, 1.0)
    bijector = ksim.AsymmetricBijector(min=jnp.array([-1.0]), max=jnp.array([1.0]))
    dist = distrax.Transformed(dist, bijector)

    assert dist.support.min == -1.0
    assert dist.support.max == 1.0

    assert dist.prob(0.0) == distrax.Normal(0.0, 1.0).prob(0.0)
