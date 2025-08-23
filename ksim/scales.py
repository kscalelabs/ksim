"""Defines scaling functions."""

__all__ = [
    "ConstantScale",
]

from abc import ABC, abstractmethod

import attrs
import jax.numpy as jnp
from jaxtyping import Array


@attrs.define(kw_only=True)
class Scale(ABC):
    """A scale function."""

    @abstractmethod
    def get_scale(self, curriculum_level: Array) -> Array: ...


@attrs.define(kw_only=True)
class ConstantScale(Scale):
    """A constant scale function."""

    scale: float = attrs.field(validator=attrs.validators.gt(0.0))

    def get_scale(self, curriculum_level: Array) -> Array:
        return self.scale * jnp.ones_like(curriculum_level)


@attrs.define(kw_only=True)
class LinearScale(Scale):
    """A linear scale function."""

    scale: float = attrs.field(validator=attrs.validators.gt(0.0))
    bias: float = attrs.field(validator=attrs.validators.gt(0.0), default=0.0)

    def get_scale(self, curriculum_level: Array) -> Array:
        return self.scale * curriculum_level + self.bias
