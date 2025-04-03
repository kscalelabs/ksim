"""Defines some commonly-used types and helper functions."""

__all__ = [
    "CartesianIndex",
    "cartesian_index_to_dim",
    "dimension_index_validator",
]

from typing import Any, Literal, get_args

import attrs
import xax

CartesianIndex = Literal["x", "y", "z"]


def cartesian_index_to_dim(index: CartesianIndex) -> int:
    match index:
        case "x":
            return 0
        case "y":
            return 1
        case "z":
            return 2
        case _:
            raise ValueError(f"Invalid linear velocity index: {index}")


def dimension_index_validator(
    inst: Any,  # noqa: ANN401
    attr: attrs.Attribute,
    value: CartesianIndex | None,
) -> None:
    if value is not None:
        choices = get_args(CartesianIndex)
        if value not in choices:
            raise ValueError(f"Linear velocity index must be one of {choices}, got {value}")


def norm_validator(
    inst: Any,  # noqa: ANN401
    attr: attrs.Attribute,
    value: xax.NormType,
) -> None:
    choices = get_args(xax.NormType)
    if value not in choices:
        raise ValueError(f"Norm must be one of {choices}, got {value}")
