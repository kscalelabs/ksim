"""Defines the base normalization class, along with some implementations."""

from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import Array


class Normalization(eqx.Module, ABC):
    """Normalizes the observations and commands."""

    @abstractmethod
    def __call__(self, observation: Array) -> Array:
        """Normalizes the observations."""
