"""Defines useful data containers."""

from dataclasses import dataclass

import mujoco.mjx as mjx

from ksim.utils.mujoco import MujocoMappings


@dataclass(frozen=True)
class BuilderData:
    """A trajectory of states."""

    model: mjx.Model
    dt: float
    ctrl_dt: float
    mujoco_mappings: MujocoMappings
