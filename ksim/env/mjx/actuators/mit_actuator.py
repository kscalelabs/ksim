"""MIT Controller, as used by the Robstride actuators."""

from dataclasses import dataclass
from typing import Any, Dict

import jax.numpy as jnp
import mujoco.mjx as mjx
from jaxtyping import Array

from ksim.env.mjx.actuators.base_actuator import Actuators, BaseActuatorMetadata
from ksim.utils.mujoco import MujocoMappings


@dataclass
class MITPositionActuatorMetadata(BaseActuatorMetadata):
    kp: float
    kd: float


class MITPositionActuators(Actuators):
    """MIT Controller, as used by the Robstride actuators."""

    def __init__(
        self,
        actuators_metadata: Dict[str, BaseActuatorMetadata],
        mujoco_mappings: MujocoMappings,
        **kwargs: Any,
    ) -> None:
        """Creates easily vector multipliable kps and kds."""
        kps_list = [0.0] * len(mujoco_mappings.ctrl_name_to_idx)
        kds_list = [0.0] * len(mujoco_mappings.ctrl_name_to_idx)

        for joint_name, actuator_metadata in actuators_metadata.items():
            actuator_name = joint_name + "_ctrl"  # TODO: properly agree upon naming accross org...
            actuator_idx = mujoco_mappings.ctrl_name_to_idx[actuator_name]
            kp = getattr(actuator_metadata, "kp", None)
            kd = getattr(actuator_metadata, "kd", None)
            if kp is None or kd is None:
                raise ValueError(
                    f"actuator_metadata for {joint_name} missing required kp or kd attribute"
                )
            kps_list[actuator_idx] = kp
            kds_list[actuator_idx] = kd

        self.kps = jnp.array(kps_list)
        self.kds = jnp.array(kds_list)
        if any(self.kps == 0) or any(self.kds == 0):
            raise ValueError("Some kps or kds are 0. Check your actuators_metadata.")

    def get_ctrl(self, mjx_data: mjx.Data, action: Array) -> Array:
        """Get the control signal from the (position) action vector."""
        current_pos = mjx_data.qpos[7:]  # NOTE: we assume first 7 are always root pos.
        ctrl = self.kps * (action - current_pos)  # TODO: explore using velocity as damping...
        return ctrl
