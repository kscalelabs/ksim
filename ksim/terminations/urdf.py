"""Defines some useful termination conditions for URDF environments."""

import math

import jax.numpy as jnp

from ksim.env.mjcf import MjcfState
from ksim.terminations.base import Termination


class EpisodeLengthTermination(Termination[MjcfState]):
    def __init__(self, max_episode_length_seconds: float, dt: float) -> None:
        super().__init__()

        self.max_episode_length = math.ceil(max_episode_length_seconds / dt)

    def __call__(self, state: MjcfState) -> jnp.ndarray:
        return state.episode_length_n > self.max_episode_length


class PitchTooGreatTermination(Termination[MjcfState]):
    def __init__(self, max_pitch: float) -> None:
        super().__init__()

        self.max_pitch = max_pitch

    def __call__(self, state: MjcfState) -> jnp.ndarray:
        return jnp.abs(state.base_euler_n3[:, 1]) > self.max_pitch


class RollToGreatTermination(Termination[MjcfState]):
    def __init__(self, max_roll: float) -> None:
        super().__init__()

        self.max_roll = max_roll

    def __call__(self, state: MjcfState) -> jnp.ndarray:
        return jnp.abs(state.base_euler_n3[:, 0]) > self.max_roll
