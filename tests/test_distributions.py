"""Tests custom distribution operations."""

import chex
import distrax
import jax
import jax.numpy as jnp
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

        sample = dist.sample(seed=rng)
        chex.assert_shape(sample, (DISTRIBUTION_SIZE,))
        log_prob = dist.log_prob(sample)
        chex.assert_shape(log_prob, (DISTRIBUTION_SIZE,))
        return log_prob.sum()

    grad_fn = jax.grad(grad_fn, argnums=(0, 1))
    gmean, gstd = grad_fn(mean, std)

    chex.assert_shape(gmean, (DISTRIBUTION_SIZE,))
    chex.assert_shape(gstd, (DISTRIBUTION_SIZE,))


def test_unit_interval_to_range_bijector() -> None:
    rng = jax.random.PRNGKey(1)
    mean = jax.random.uniform(rng, (DISTRIBUTION_SIZE,), minval=0.0, maxval=1.0)
    std = jax.random.uniform(rng, (DISTRIBUTION_SIZE,), minval=0.01, maxval=0.1)

    def grad_fn(mean: Array, std: Array) -> Array:
        rng = jax.random.PRNGKey(0)
        dist = distrax.Normal(mean, std)

        # Clip to ensure values are in [0,1]
        dist = distrax.Transformed(dist, distrax.Sigmoid())

        # Create target range
        min_val = jax.random.uniform(rng, (DISTRIBUTION_SIZE,), minval=-2.0, maxval=-1.0)
        max_val = jax.random.uniform(rng, (DISTRIBUTION_SIZE,), minval=1.0, maxval=2.0)
        dist = distrax.Transformed(dist, ksim.UnitIntervalToRangeBijector(min=min_val, max=max_val))

        # Sample and compute log probability
        sample = dist.sample(seed=rng)
        chex.assert_shape(sample, (DISTRIBUTION_SIZE,))

        log_prob = dist.log_prob(sample)
        chex.assert_shape(log_prob, (DISTRIBUTION_SIZE,))
        return log_prob.sum()

    # Test gradients
    grad_fn = jax.grad(grad_fn, argnums=(0, 1))
    gmean, gstd = grad_fn(mean, std)

    chex.assert_shape(gmean, (DISTRIBUTION_SIZE,))
    chex.assert_shape(gstd, (DISTRIBUTION_SIZE,))

    # Test forward and inverse transformations
    rng, rng1, rng2 = jax.random.split(rng, 3)
    x = jax.random.uniform(rng1, (DISTRIBUTION_SIZE,), minval=0.0, maxval=1.0)
    min_val = jax.random.uniform(rng2, (DISTRIBUTION_SIZE,), minval=-2.0, maxval=-1.0)
    max_val = jax.random.uniform(rng2, (DISTRIBUTION_SIZE,), minval=1.0, maxval=2.0)

    bijector = ksim.UnitIntervalToRangeBijector(min=min_val, max=max_val)

    # Test forward transformation
    y, log_det = bijector.forward_and_log_det(x)
    chex.assert_shape(y, (DISTRIBUTION_SIZE,))
    chex.assert_shape(log_det, (DISTRIBUTION_SIZE,))

    # Test inverse transformation
    x_recon, log_det_inv = bijector.inverse_and_log_det(y)
    chex.assert_trees_all_close(x, x_recon, atol=1e-6)
    chex.assert_trees_all_close(log_det, -log_det_inv, atol=1e-6)


def test_double_unit_interval_to_range_bijector() -> None:
    rng = jax.random.PRNGKey(1)
    mean = jax.random.uniform(rng, (DISTRIBUTION_SIZE,), minval=-1.0, maxval=1.0)
    std = jax.random.uniform(rng, (DISTRIBUTION_SIZE,), minval=0.01, maxval=0.1)

    def grad_fn(mean: Array, std: Array) -> Array:
        rng = jax.random.PRNGKey(0)
        dist = distrax.Normal(mean, std)

        # Create target range
        min_val = jax.random.uniform(rng, (DISTRIBUTION_SIZE,), minval=-2.0, maxval=-1.0)
        max_val = jax.random.uniform(rng, (DISTRIBUTION_SIZE,), minval=1.0, maxval=2.0)
        dist = distrax.Transformed(dist, ksim.DoubleUnitIntervalToRangeBijector(min=min_val, max=max_val))

        # Sample and compute log probability
        sample = dist.sample(seed=rng)
        chex.assert_shape(sample, (DISTRIBUTION_SIZE,))

        log_prob = dist.log_prob(sample)
        chex.assert_shape(log_prob, (DISTRIBUTION_SIZE,))
        return log_prob.sum()

    # Test gradients
    grad_fn = jax.grad(grad_fn, argnums=(0, 1))
    gmean, gstd = grad_fn(mean, std)

    chex.assert_shape(gmean, (DISTRIBUTION_SIZE,))
    chex.assert_shape(gstd, (DISTRIBUTION_SIZE,))

    # Test forward and inverse transformations
    rng, rng1, rng2 = jax.random.split(rng, 3)
    x = jax.random.uniform(rng1, (DISTRIBUTION_SIZE,), minval=-1.0, maxval=1.0)
    min_val = jax.random.uniform(rng2, (DISTRIBUTION_SIZE,), minval=-2.0, maxval=-1.0)
    max_val = jax.random.uniform(rng2, (DISTRIBUTION_SIZE,), minval=1.0, maxval=2.0)

    bijector = ksim.DoubleUnitIntervalToRangeBijector(min=min_val, max=max_val)

    # Test forward transformation
    y, log_det = bijector.forward_and_log_det(x)
    chex.assert_shape(y, (DISTRIBUTION_SIZE,))
    chex.assert_shape(log_det, (DISTRIBUTION_SIZE,))

    # Test inverse transformation
    x_recon, log_det_inv = bijector.inverse_and_log_det(y)
    chex.assert_trees_all_close(x, x_recon, atol=1e-6)
    chex.assert_trees_all_close(log_det, -log_det_inv, atol=1e-6)

    # Test bounds
    chex.assert_trees_all_close(jnp.minimum(y, max_val), y, atol=1e-6)
    chex.assert_trees_all_close(jnp.maximum(y, min_val), y, atol=1e-6)
