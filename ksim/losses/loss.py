"""Defines the base loss class, along with some implementations."""

from abc import ABC, abstractmethod
from typing import Any

import equinox as eqx
from jaxtyping import Array


class Loss(ABC):
    @abstractmethod
    def compute_loss(self, agent: eqx.Module, batch: Any, output: Any) -> Array:  # noqa: ANN401
        """Compute the loss for a given model and batch."""
