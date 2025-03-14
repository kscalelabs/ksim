"""Defines useful data containers."""

from dataclasses import dataclass

from kscale.web.gen.api import RobotURDFMetadataOutput
from mujoco import mjx

from ksim.utils.mujoco import MujocoMappings


@dataclass(frozen=True)
class BuilderData:
    """A trajectory of states."""

    robot_model: mjx.Model
    robot_metadata: RobotURDFMetadataOutput | None
    dt: float
    ctrl_dt: float
    mujoco_mappings: MujocoMappings
