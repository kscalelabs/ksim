"""MJX environment class."""

from abc import abstractmethod
from typing import Collection

import mujoco
from jaxtyping import Array
from kscale.web.gen.api import RobotURDFMetadataOutput
from mujoco import mjx

from ksim.actuators import Actuators, ActuatorsBuilder
from ksim.env.base_engine import BaseEngine
from ksim.env.data import PhysicsModel, PhysicsState
from ksim.resets import Reset, ResetBuilder


def create_mjx_model(
    scene_path: str,
    dt: float,
    iterations: int,
    ls_iterations: int,
    disableflags: int,
    solver: mujoco.mjtSolver = mujoco.mjtSolver.mjSOL_CG,
) -> mjx.Model:
    """Create MJX model from a scene path and options."""
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = dt
    model.opt.iterations = iterations
    model.opt.ls_iterations = ls_iterations
    model.opt.disableflags = disableflags
    model.opt.solver = solver
    return mjx.put_model(model)


class MjxEngine(BaseEngine):
    """MJX engine class.

    NOTE: resetting and actuator logic live here, not during unrolling. This is
    - Actuators necessarily must get applied during unrolling.
    - Resets are not necessarily applied during unrolling.
    """

    def __init__(
        self,
        default_physics_model: PhysicsModel,
        resetters: Collection[Reset],
        actuators: Actuators,
        *,
        dt: float,
        iterations: int,
        ls_iterations: int,
        disableflags: int,
    ) -> None:
        """Initialize the MJX engine with resetting and actuators."""
        assert isinstance(default_physics_model, mjx.Model)
        self.default_mjx_model = default_physics_model

        self.resetters = [r(self.default_mjx_model) if isinstance(r, ResetBuilder) else r for r in resetters]

        if isinstance(actuators, ActuatorsBuilder):
            assert robot_metadata is not None, "Robot metadata is required for actuators"
            joint_map = robot_metadata.joint_name_to_metadata
            assert joint_map is not None, "Joint name to metadata is required for actuators"

            self.actuators = actuators(joint_name_to_metadata=joint_map, mujoco_mappings=self.mujoco_mappings)

        else:
            self.actuators = actuators

    @abstractmethod
    def reset(self) -> PhysicsState:
        """Reset the engine and return the physics model and data."""

    @abstractmethod
    def step(
        self,
        action: Array,
        state: PhysicsState,
    ) -> PhysicsState:
        """Step the engine and return the physics model and data."""
