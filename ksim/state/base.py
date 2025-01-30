"""Defines the base state class."""

from brax.envs.base import State as BraxState
from flax import struct


@struct.dataclass
class State(BraxState):
    pass
