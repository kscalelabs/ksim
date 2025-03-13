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


class MeanStdNormalizer(Normalizer):
    """Standardizes a pytree of arrays using the mean and std."""

    mean: PyTree[Array]
    std: PyTree[Array]

    def __init__(self, pytree: PyTree[Array]) -> None:
        """Initializes the normalization statistics."""
        self.mean = jax.tree_map(jnp.zeros_like, pytree)
        self.std = jax.tree_map(jnp.ones_like, pytree)

    def __call__(self, pytree: PyTree[Array]) -> PyTree[Array]:
        """Standardizes a pytree of arrays using the mean and std."""

        def normalize_leaf(x: Array, mean: Array, std: Array) -> Array:
            std = jax.lax.cond(
                std > 0,
                lambda: std,
                lambda: jnp.ones_like(std),
            )
            return (x - mean) / std

        return jax.tree.map(normalize_leaf, pytree, self.mean, self.std)

    def update(self, pytree: PyTree[Array]) -> "MeanStdNormalizer":
        """Updates the normalization statistics in a stateless manner."""
        new_mean = jax.tree_map(jnp.mean, pytree)
        new_std = jax.tree_map(jnp.std, pytree)
        return eqx.tree_at(lambda t: (t.mean, t.std), self, (new_mean, new_std))
