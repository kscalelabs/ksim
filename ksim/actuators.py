"""Defines the base actuators class, along with some implementations."""

from abc import ABC, abstractmethod
from typing import Mapping, TypedDict

import attrs
import jax.numpy as jnp
from jaxtyping import Array
from mujoco import mjx

from ksim.utils.mujoco import MujocoMappings


@attrs.define(frozen=True)
class Actuators(ABC):
    """Collection of actuators."""

    @abstractmethod
    def get_ctrl(self, mjx_data: mjx.Data, action: Array) -> Array:
        """Get the control signal from the action vector."""
        ...


class TorqueActuators(Actuators):
    def get_ctrl(self, mjx_data: mjx.Data, action: Array) -> Array:
        # Just use the action as the torque, the simplest actuator model.
        return action


class PositionParams(TypedDict):
    kp: float
    kd: float


class MITPositionActuators(Actuators):
    """MIT Controller, as used by the Robstride actuators."""

    def __init__(
        self,
        joint_to_kp_kds: Mapping[str, PositionParams],
        mujoco_mappings: MujocoMappings,
    ) -> None:
        """Creates easily vector multipliable kps and kds."""
        kps_list = [0.0] * len(mujoco_mappings.ctrl_name_to_idx)
        kds_list = [0.0] * len(mujoco_mappings.ctrl_name_to_idx)

        for joint_name, params in joint_to_kp_kds.items():
            actuator_name = self.get_actuator_name(joint_name)
            actuator_idx = mujoco_mappings.ctrl_name_to_idx[actuator_name]
            kps_list[actuator_idx] = params["kp"]
            kds_list[actuator_idx] = params["kd"]

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


class ScaledTorqueActuators(Actuators):
    def __init__(
        self,
        joint_to_gear_ratios: Mapping[str, float],
        joint_to_input_ranges: Mapping[str, tuple[float, float]],
        mujoco_mappings: MujocoMappings,
    ) -> None:
        """Creates an instance of ScaledTorqueActuators."""
        self.gear_ratios = jnp.array(
            [joint_to_gear_ratios[m] for m in mujoco_mappings.ctrl_name_to_idx]
        )
        self.input_ranges = jnp.array(
            [joint_to_input_ranges[m] for m in mujoco_mappings.ctrl_name_to_idx]
        )

    def get_ctrl(self, mjx_data: mjx.Data, action: Array) -> Array:
        """Get the control signal from the (position) action vector."""
        # Copy legacy and brax setup here
        action_min = self.input_ranges[:, 0]
        action_max = self.input_ranges[:, 1]
        # Scale action from [-1,1] to actuator limits
        ctrl = action * (action_max - action_min) * 0.5
        return ctrl
