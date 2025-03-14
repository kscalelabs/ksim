"""Defines the base actuators class, along with some implementations."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import jax.numpy as jnp
from jaxtyping import Array
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx

from ksim.utils.data import BuilderData
from ksim.utils.mujoco import MujocoMappings


class Actuators(ABC):
    """Collection of actuators."""

    @abstractmethod
    def get_ctrl(self, mjx_data: mjx.Data, action: Array) -> Array:
        """Get the control signal from the action vector."""


T = TypeVar("T", bound=Actuators)


class ActuatorsBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, data: BuilderData) -> T:
        """Builds an observation from a MuJoCo model."""


class TorqueActuators(Actuators):
    def get_ctrl(self, mjx_data: mjx.Data, action: Array) -> Array:
        # Just use the action as the torque, the simplest actuator model.
        return action


class MITPositionActuators(Actuators):
    """MIT Controller, as used by the Robstride actuators."""

    def __init__(
        self,
        joint_to_kp_kds: dict[str, JointMetadataOutput],
        mujoco_mappings: MujocoMappings,
    ) -> None:
        """Creates easily vector multipliable kps and kds."""
        kps_list = [0.0] * len(mujoco_mappings.ctrl_name_to_idx)
        kds_list = [0.0] * len(mujoco_mappings.ctrl_name_to_idx)

        for joint_name, params in joint_to_kp_kds.items():
            actuator_name = self.get_actuator_name(joint_name)
            actuator_idx = mujoco_mappings.ctrl_name_to_idx[actuator_name]

            kp_str = params.kp
            kd_str = params.kd
            assert kp_str is not None and kd_str is not None, f"Missing kp or kd for joint {joint_name}"
            kp = float(kp_str)
            kd = float(kd_str)

            kps_list[actuator_idx] = kp
            kds_list[actuator_idx] = kd

        self.kps = jnp.array(kps_list)
        self.kds = jnp.array(kds_list)
        if any(self.kps == 0) or any(self.kds == 0):
            raise ValueError("Some kps or kds are 0. Check your actuators_metadata.")

    def get_actuator_name(self, joint_name: str) -> str:
        # This can be overridden if necessary.
        return f"{joint_name}_ctrl"

    def get_ctrl(self, mjx_data: mjx.Data, action: Array) -> Array:
        """Get the control signal from the (position) action vector."""
        current_pos = mjx_data.qpos[7:]  # First 7 are always root pos.
        current_vel = mjx_data.qvel[6:]  # First 6 are always root vel.
        target_velocities = jnp.zeros_like(action)

        pos_delta = action - current_pos
        vel_delta = target_velocities - current_vel

        ctrl = self.kps * pos_delta + self.kds * vel_delta
        return ctrl


class MITPositionActuatorsBuilder(ActuatorsBuilder[MITPositionActuators]):
    """Builder for MITPositionActuators."""

    def __call__(self, data: BuilderData) -> MITPositionActuators:
        """Builds an MITPositionActuators instance."""
        if data.robot_metadata.joint_name_to_metadata is None:
            raise ValueError("Missing joint_name_to_metadata within robot metadata")
        return MITPositionActuators(
            joint_to_kp_kds=data.robot_metadata.joint_name_to_metadata,
            mujoco_mappings=data.mujoco_mappings,
        )
