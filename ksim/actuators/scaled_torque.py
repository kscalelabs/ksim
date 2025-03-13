"""Direct torque control, scaled to a specifiable input range and gear ratio."""

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
from jaxtyping import Array
from mujoco import mjx

from ksim.actuators.base import Actuators
from ksim.utils.mujoco import MujocoMappings


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
