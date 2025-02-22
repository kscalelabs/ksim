"""Includes all major typing."""

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

ModelObs = Array | PyTree[Array]
ModelOut = Array | PyTree[Array]
