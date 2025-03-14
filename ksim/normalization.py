"""Defines the base normalization class, along with some implementations."""

from abc import ABC, abstractmethod
from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree


class Normalizer(eqx.Module, ABC):
    """Normalizes a pytree of arrays.

    For more fine-grained normalization, feel free to subclass this and pass
    in to the task definition.
    """

    @abstractmethod
    def __call__(self, pytree: PyTree[Array]) -> PyTree[Array]:
        """Normalizes a pytree of arrays."""

    @abstractmethod
    def update(self, pytree: PyTree[Array]) -> Self:
        """Updates the normalization statistics."""


class PassThrough(Normalizer):
    """Passes through the pytree without normalization."""

    def __call__(self, pytree: PyTree[Array]) -> PyTree[Array]:
        """Passes through the pytree without normalization."""
        return pytree

    def update(self, pytree: PyTree[Array]) -> Self:
        """Returns self."""
        return self


class Standardize(Normalizer):
    """Standardizes a Pytree of arrays with online updates."""

    mean: PyTree[Array]
    std: PyTree[Array]
    alpha: float

    def __init__(self, pytree: PyTree[Array], *, alpha: float) -> None:
        """Initializes the normalization statistics."""
        leaf_shapes = jax.tree_map(lambda x: x.shape[-1], pytree)
        self.mean = jax.tree_map(jnp.zeros, leaf_shapes)
        self.std = jax.tree_map(jnp.ones, leaf_shapes)
        self.alpha = alpha

    def __call__(self, pytree: PyTree[Array]) -> PyTree[Array]:
        """Standardizes a pytree of arrays using the mean and std."""

        def normalize_leaf(x: Array, mean: Array, std: Array) -> Array:
            """Normalizes a leaf of the pytree."""
            std = jnp.where(std > 0, std, jnp.ones_like(std))
            return (x - mean) / std

        res = jax.tree_util.tree_map(normalize_leaf, pytree, self.mean, self.std)
        return res

    def update(self, pytree: PyTree[Array]) -> Self:
        """Updates the normalization statistics in a stateless manner."""

        def update_leaf_stats(x: Array, old_mean: Array, old_std: Array) -> tuple[Array, Array]:
            """Updates the normalization statistics for a leaf of the pytree."""
            batch_dims = x.shape[:-1]
            new_mean = jnp.mean(x, axis=tuple(range(len(batch_dims))))
            new_std = jnp.std(x, axis=tuple(range(len(batch_dims))))
            new_mean = (1 - self.alpha) * old_mean + self.alpha * new_mean
            new_std = (1 - self.alpha) * old_std + self.alpha * new_std
            return new_mean, new_std

        new_mean, new_std = jax.tree_util.tree_map(update_leaf_stats, pytree, self.mean, self.std)
        res = eqx.tree_at(lambda t: (t.mean, t.std), self, (new_mean, new_std))
        return res
