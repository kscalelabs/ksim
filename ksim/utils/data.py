"""Defines useful data containers."""

from dataclasses import dataclass

from mujoco import mjx

from ksim.utils.mujoco import MujocoMappings


@dataclass(frozen=True)
class BuilderData:
    """A trajectory of states."""

    model: mjx.Model
    dt: float
    ctrl_dt: float
    mujoco_mappings: MujocoMappings
