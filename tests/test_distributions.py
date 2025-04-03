"""Tests custom distribution operations."""

import chex
import distrax
import jax
from jaxtyping import Array

import ksim

DISTRIBUTION_SIZE = 1000


def test_asymmetric_bijector() -> None:
    rng = jax.random.PRNGKey(1)
    mean = jax.random.uniform(rng, (DISTRIBUTION_SIZE,), minval=-1.0, maxval=1.0)
    std = jax.random.uniform(rng, (DISTRIBUTION_SIZE,), minval=0.01, maxval=1.0)

    def grad_fn(mean: Array, std: Array) -> Array:
        rng = jax.random.PRNGKey(0)
        dist = distrax.Normal(mean, std)

        min = jax.random.uniform(rng, (DISTRIBUTION_SIZE,), minval=-3.0, maxval=-1.0)
        max = jax.random.uniform(rng, (DISTRIBUTION_SIZE,), minval=1.0, maxval=3.0)
        dist = distrax.Transformed(dist, ksim.AsymmetricBijector(min=min, max=max))

        # dist = distrax.Transformed(dist, distrax.Tanh())

        sample = dist.sample(seed=rng)
        chex.assert_shape(sample, (DISTRIBUTION_SIZE,))
        log_prob = dist.log_prob(sample)
        chex.assert_shape(log_prob, (DISTRIBUTION_SIZE,))
        return log_prob.sum()

    grad_fn = jax.grad(grad_fn, argnums=(0, 1))
    gmean, gstd = grad_fn(mean, std)

    chex.assert_shape(gmean, (DISTRIBUTION_SIZE,))
    chex.assert_shape(gstd, (DISTRIBUTION_SIZE,))
