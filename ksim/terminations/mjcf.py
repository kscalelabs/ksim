"""Defines some useful termination conditions for MJCF environments."""

import jax.numpy as jnp
from brax.base import State

from ksim.terminations.base import Termination


class PitchTooGreatTermination(Termination):
    """Terminates the episode if the pitch is too great."""

    max_pitch: float

    def __init__(self, max_pitch: float) -> None:
        super().__init__()

        self.max_pitch = max_pitch

    def __call__(self, state: State) -> jnp.ndarray:
        quat = state.q[3:7]
        pitch = jnp.arctan2(2 * quat[1] * quat[2] - 2 * quat[0] * quat[3], 1 - 2 * quat[1] ** 2 - 2 * quat[2] ** 2)
        return jnp.abs(pitch) > self.max_pitch


class RollTooGreatTermination(Termination):
    """Terminates the episode if the roll is too great."""

    max_roll: float

    def __init__(self, max_roll: float) -> None:
        super().__init__()

        self.max_roll = max_roll

    def __call__(self, state: State) -> jnp.ndarray:
        quat = state.q[3:7]
        roll = jnp.arctan2(2 * quat[0] * quat[3] - 2 * quat[1] * quat[2], 1 - 2 * quat[0] ** 2 - 2 * quat[3] ** 2)
        return jnp.abs(roll) > self.max_roll
