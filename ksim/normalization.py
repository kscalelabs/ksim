"""Defines the base normalization class, along with some implementations."""

from abc import ABC, abstractmethod
from typing import Mapping, TypedDict

import attrs
import jax.numpy as jnp
from jaxtyping import Array
from mujoco import mjx

from ksim.utils.mujoco import MujocoMappings


@attrs.define(frozen=True)
class Normalization:
    """Normalizes the observations and commands."""

    mappings: MujocoMappings

    @abstractmethod
    def __call__(self, observation: Array) -> Array:
        """Normalizes the observations."""
