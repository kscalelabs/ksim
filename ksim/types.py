"""Typing for the ksim library."""

from typing import Literal

from ksim.env.data import Transition
from ksim.model.types import ModelInput
from ksim.task.types import RolloutTimeStats

NoiseType = Literal["gaussian", "uniform"]
ObsType = Literal["proprio", "image"]
CmdType = Literal["vector", "text"]

__all__ = ["Transition", "ModelInput", "NoiseType", "ObsType", "CmdType", "RolloutTimeStats"]
