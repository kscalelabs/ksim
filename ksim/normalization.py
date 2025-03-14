"""Defines the base normalization class, along with some implementations."""

from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree


class Normalizer(eqx.Module, ABC):
    """Normalizes a pytree of arrays.

    For more fine-grained normalization, feel free to subclass this and pass
    in to the task definition.

    E.g.
    ```python
    class CustomNormalizer(Normalizer):
        def __call__(self, pytree: PyTree[Array]) -> PyTree[Array]:
            # maybe do batch-wise norm on vector observations
            # maybe do layer-wise norm on tensor observations
            return ...
    ```
    """

    @abstractmethod
    def __call__(self, pytree: PyTree[Array]) -> PyTree[Array]:
        """Normalizes a pytree of arrays."""

    @abstractmethod
    def update(self, pytree: PyTree[Array]) -> "Normalizer":
        """Updates the normalization statistics."""


class PassThrough(Normalizer):
    """Passes through the pytree without normalization."""

    def __call__(self, pytree: PyTree[Array]) -> PyTree[Array]:
        """Passes through the pytree without normalization."""
        return pytree


class Standardize(Normalizer):
    """Standardizes a pytree of arrays with mean and std along batch dims."""

    def __call__(self, pytree: PyTree[Array]) -> PyTree[Array]:
        """Standardizes a pytree of arrays with mean and std along batch dims."""

        def standardize_leaf(x: Array) -> Array:
            """Standardizes a leaf of the pytree."""
            batch_dims = x.shape[:-1]
            mean = jnp.mean(x, axis=tuple(range(len(batch_dims))))
            std = jnp.std(x, axis=tuple(range(len(batch_dims))))
            std = jax.lax.cond(
                std > 0,
                lambda: std,
                lambda: jnp.ones_like(std),
            )
            return (x - mean) / std

        return jax.tree_map(standardize_leaf, pytree)

    def update(self, pytree: PyTree[Array]) -> "Standardize":
        """Updates the normalization statistics."""
        return self


class OnlineStandardizer(Normalizer):
    """Standardizes a pytree of arrays with online updates."""

    mean: PyTree[Array]
    std: PyTree[Array]
    alpha: float

    def __init__(self, pytree: PyTree[Array]) -> None:
        """Initializes the normalization statistics."""
        leaf_shapes = jax.tree_map(lambda x: x.shape[-1:], pytree)
        self.mean = jax.tree_map(jnp.zeros, leaf_shapes)
        self.std = jax.tree_map(jnp.ones, leaf_shapes)

    def __call__(self, pytree: PyTree[Array]) -> PyTree[Array]:
        """Standardizes a pytree of arrays using the mean and std."""

        def normalize_leaf(x: Array, mean: Array, std: Array) -> Array:
            """Normalizes a leaf of the pytree."""
            std = jax.lax.cond(
                std > 0,
                lambda: std,
                lambda: jnp.ones_like(std),
            )
            return (x - mean) / std

        return jax.tree_util.tree_map(normalize_leaf, pytree, self.mean, self.std)

    def update(self, pytree: PyTree[Array]) -> "OnlineStandardizer":
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
        return eqx.tree_at(lambda t: (t.mean, t.std), self, (new_mean, new_std))
