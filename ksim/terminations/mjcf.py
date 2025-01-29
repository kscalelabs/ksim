"""Defines some useful termination conditions for MJCF environments."""

import math
from typing import Generic, TypeVar

import jax.numpy as jnp

from ksim.env.mjcf import MjcfState
from ksim.resets.mjcf import XYPositionReset
from ksim.terminations.base import Termination

Tstate = TypeVar("Tstate", bound=MjcfState)


class EpisodeLengthTermination(Termination[Tstate], Generic[Tstate]):
    def __init__(self, max_episode_length_seconds: float, dt: float) -> None:
        super().__init__()

        self.max_episode_length = math.ceil(max_episode_length_seconds / dt)

    def __call__(self, state: Tstate) -> jnp.ndarray:
        return state.episode_length_n > self.max_episode_length


class PitchTooGreatTermination(Termination[Tstate], Generic[Tstate]):
    def __init__(self, max_pitch: float) -> None:
        super().__init__()

        self.max_pitch = max_pitch

    def __call__(self, state: Tstate) -> jnp.ndarray:
        return jnp.abs(state.base_euler_n3[:, 1]) > self.max_pitch


class RollToGreatTermination(Termination[Tstate], Generic[Tstate]):
    def __init__(self, max_roll: float) -> None:
        super().__init__()

        self.max_roll = max_roll

    def __call__(self, state: Tstate) -> jnp.ndarray:
        return jnp.abs(state.base_euler_n3[:, 0]) > self.max_roll
