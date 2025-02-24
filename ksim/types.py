"""Includes all major typing."""

from jaxtyping import Array, PyTree

ModelObs = Array | PyTree[Array]
ModelOut = Array | PyTree[Array]
