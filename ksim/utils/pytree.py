"""Utils for accessing, modifying, and otherwise manipulating pytrees."""

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree


def slice_array(x: Array, start: Array, slice_length: int) -> Array:
    """Get a slice of an array along the first dimension.

    For multi-dimensional arrays, this slices only along the first dimension
    and keeps all other dimensions intact.
    """
    chex.assert_shape(start, ())
    chex.assert_shape(slice_length, ())
    start_indices = (start,) + (0,) * (len(x.shape) - 1)
    slice_sizes = (slice_length,) + x.shape[1:]

    return jax.lax.dynamic_slice(x, start_indices, slice_sizes)


def slice_pytree(pytree: PyTree, start: Array, slice_length: int) -> PyTree:
    """Get a slice of a pytree."""
    return jax.tree_util.tree_map(lambda x: slice_array(x, start, slice_length), pytree)


def flatten_array(x: Array, flatten_size: int) -> Array:
    """Flatten an array into a (flatten_size, ...) array."""
    reshaped = jnp.reshape(x, (flatten_size, *x.shape[2:]))
    assert reshaped.shape[0] == flatten_size
    return reshaped


def flatten_pytree(pytree: PyTree, flatten_size: int) -> PyTree:
    """Flatten a pytree into a (flatten_size, ...) pytree."""
    return jax.tree_util.tree_map(lambda x: flatten_array(x, flatten_size), pytree)


def compute_nan_ratio(pytree: PyTree) -> Array:
    """Computes the ratio of NaNs vs non-NaNs in a given PyTree."""
    nan_counts = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.isnan(x)), pytree)
    total_counts = jax.tree_util.tree_map(lambda x: x.size, pytree)

    total_nans = jax.tree_util.tree_reduce(lambda a, b: a + b, nan_counts, 0)
    total_elements = jax.tree_util.tree_reduce(lambda a, b: a + b, total_counts, 0)
    overall_nan_ratio = jnp.array(total_nans / total_elements)

    return overall_nan_ratio
