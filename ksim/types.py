"""Typing for the ksim library."""

from typing import Literal

from ksim.env.types import EnvState
from ksim.model.types import ModelInput
from ksim.task.types import RolloutTimeLossComponents

NoiseType = Literal["gaussian", "uniform"]
ObsType = Literal["proprio", "image"]
CmdType = Literal["vector", "text"]

__all__ = ["EnvState", "ModelInput", "NoiseType", "ObsType", "CmdType", "RolloutTimeLossComponents"]
