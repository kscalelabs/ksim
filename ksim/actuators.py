"""Defines the base actuators class, along with some implementations."""

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import jax.numpy as jnp
from jaxtyping import Array
from kscale.web.gen.api import JointMetadataOutput

from ksim.env.data import PhysicsData, PhysicsModel
from ksim.utils.mujoco import get_ctrl_data_idx_by_name

logger = logging.getLogger(__name__)


class Actuators(ABC):
    """Collection of actuators."""

    @abstractmethod
    def get_ctrl(self, action: Array, physics_data: PhysicsData) -> Array:
        """Get the control signal from the action vector."""


T = TypeVar("T", bound=Actuators)


class ActuatorsBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(
        self,
        physics_model: PhysicsModel,
        joint_name_to_metadata: dict[str, JointMetadataOutput],
    ) -> T:
        """Builds an observation from a MuJoCo model."""


class TorqueActuators(Actuators):
    """Direct torque control."""

    def get_ctrl(self, action: Array, physics_data: PhysicsData) -> Array:
        """Just use the action as the torque, the simplest actuator model."""
        action_max = 0.4
        action_min = -0.4
        ctrl = (action + 1) * (action_max - action_min) * 0.5 + action_min
 
        return ctrl


class MITPositionActuators(Actuators):
    """MIT Controller, as used by the Robstride actuators."""

    def __init__(
        self,
        physics_model: PhysicsModel,
        joint_name_to_metadata: dict[str, JointMetadataOutput],
    ) -> None:
        """Creates easily vector multipliable kps and kds."""
        ctrl_name_to_idx = get_ctrl_data_idx_by_name(physics_model)
        kps_list = [-1.0] * len(ctrl_name_to_idx)
        kds_list = [-1.0] * len(ctrl_name_to_idx)

        for joint_name, params in joint_name_to_metadata.items():
            actuator_name = self.get_actuator_name(joint_name)
            if actuator_name not in ctrl_name_to_idx:
                logger.warning("Joint %s has no actuator name. Skipping.", joint_name)
                continue
            actuator_idx = ctrl_name_to_idx[actuator_name]

            kp_str = params.kp
            kd_str = params.kd
            assert kp_str is not None and kd_str is not None, f"Missing kp or kd for joint {joint_name}"
            kp = float(kp_str)
            kd = float(kd_str)

            kps_list[actuator_idx] = kp
            kds_list[actuator_idx] = kd

        self.kps = jnp.array(kps_list)
        self.kds = jnp.array(kds_list)

        if any(self.kps < 0) or any(self.kds < 0):
            raise ValueError("Some KPs or KDs are negative. Check the provided metadata.")
        if any(self.kps == 0) or any(self.kds == 0):
            logger.warning("Some KPs or KDs are 0. Check the provided metadata.")

    def get_actuator_name(self, joint_name: str) -> str:
        # This can be overridden if necessary.
        return f"{joint_name}_ctrl"

    def get_ctrl(self, action: Array, physics_data: PhysicsData) -> Array:
        """Get the control signal from the (position) action vector."""
        current_pos = physics_data.qpos[7:]  # First 7 are always root pos.
        current_vel = physics_data.qvel[6:]  # First 6 are always root vel.
        target_velocities = jnp.zeros_like(action)

        pos_delta = action - current_pos
        vel_delta = target_velocities - current_vel

        ctrl = self.kps * pos_delta + self.kds * vel_delta
        return ctrl
