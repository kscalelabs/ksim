"""Typing for the ksim library."""

from typing import Literal

from ksim.env.data import Transition
from ksim.task.types import RolloutTimeStats

NoiseType = Literal["gaussian", "uniform", "uniform_multiply"]
ObsType = Literal["proprio", "image"]
CmdType = Literal["vector", "text"]

__all__ = ["Transition", "NoiseType", "ObsType", "CmdType", "RolloutTimeStats"]
