"""Defines useful data containers."""

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class BuilderData:
    """A trajectory of states."""

    model: mujoco.MjModel
    dt: float
    ctrl_dt: float
    joint_name_to_idx: dict[str, int]
    body_name_to_idx: dict[str, int]
    actuator_name_to_idx: dict[str, int]
    geom_name_to_idx: dict[str, int]
    site_name_to_idx: dict[str, int]
    sensor_name_to_idx: dict[str, int]
