"""Defines state class for walking robots."""

from dataclasses import dataclass

import jax.numpy as jnp

from ksim.state.mjcf import MjcfState


@dataclass
class WalkingState(MjcfState):
    """Defines a state for a walking robot.

    This defines additional fields that are useful for walking robots.
    """

    commands_n3: jnp.ndarray
