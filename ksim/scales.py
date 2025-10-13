"""Defines scaling functions."""

__all__ = [
    "Scale",
    "ConstantScale",
    "LinearScale",
    "QuadraticScale",
    "SquareRootScale",
    "NonZeroScale",
    "convert_to_scale",
]

from abc import ABC, abstractmethod
from typing import Self

import attrs
import jax.numpy as jnp
from jaxtyping import Array


@attrs.define(frozen=True, kw_only=True)
class Scale(ABC):
    """A scale function."""

    @abstractmethod
    def get_scale(self, curriculum_level: Array) -> Array: ...


@attrs.define(frozen=True, kw_only=True)
class ConstantScale(Scale):
    """A constant scale function."""

    scale: float = attrs.field(validator=attrs.validators.ge(0.0))

    def get_scale(self, curriculum_level: Array) -> Array:
        return self.scale * jnp.ones_like(curriculum_level)


@attrs.define(frozen=True, kw_only=True)
class LinearScale(Scale):
    """A linear scale function."""

    scale: float = attrs.field()
    bias: float = attrs.field(validator=attrs.validators.ge(0.0), default=0.0)

    @scale.validator
    def _validate_scale_bias_sum(self, attribute: str, value: float) -> None:
        if value + self.bias < 0.0:
            raise ValueError(f"scale + bias must be non-negative, got {value} + {self.bias} = {value + self.bias}")

    def get_scale(self, curriculum_level: Array) -> Array:
        return self.scale * curriculum_level + self.bias

    @classmethod
    def from_endpoints(cls, start: float, end: float) -> Self:
        return cls(scale=(end - start), bias=start)


@attrs.define(frozen=True, kw_only=True)
class QuadraticScale(Scale):
    """A exponential scale function."""

    scale: float = attrs.field()
    bias: float = attrs.field(validator=attrs.validators.ge(0.0), default=0.0)

    @scale.validator
    def _validate_scale_bias_sum(self, attribute: str, value: float) -> None:
        if value + self.bias < 0.0:
            raise ValueError(f"scale + bias must be non-negative, got {value} + {self.bias} = {value + self.bias}")

    def get_scale(self, curriculum_level: Array) -> Array:
        return self.scale * jnp.square(curriculum_level) + self.bias

    @classmethod
    def from_endpoints(cls, start: float, end: float) -> Self:
        return cls(scale=(end - start), bias=start)


@attrs.define(frozen=True, kw_only=True)
class SquareRootScale(Scale):
    """A square root scale function."""

    scale: float = attrs.field()
    bias: float = attrs.field(validator=attrs.validators.ge(0.0), default=0.0)

    @scale.validator
    def _validate_scale_bias_sum(self, attribute: str, value: float) -> None:
        if value + self.bias < 0.0:
            raise ValueError(f"scale + bias must be non-negative, got {value} + {self.bias} = {value + self.bias}")

    def get_scale(self, curriculum_level: Array) -> Array:
        return self.scale * jnp.sqrt(curriculum_level) + self.bias

    @classmethod
    def from_endpoints(cls, start: float, end: float) -> Self:
        return cls(scale=(end - start), bias=start)


@attrs.define(frozen=True, kw_only=True)
class NonZeroScale(Scale):
    """Apply the scale only when the curriculum level is non-zero."""

    scale: float = attrs.field()
    bias: float = attrs.field(validator=attrs.validators.ge(0.0), default=0.0)

    def get_scale(self, curriculum_level: Array) -> Array:
        return jnp.where(curriculum_level > 0.0, self.scale, self.bias)


def convert_to_scale(value: float | int | Scale) -> Scale:
    if isinstance(value, (float, int)):
        return ConstantScale(scale=value)
    if isinstance(value, Scale):
        return value
    raise ValueError(f"Invalid scale: {value}")
