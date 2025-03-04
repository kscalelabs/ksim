"""MIT Controller, as used by the Robstride actuators."""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array
from mujoco import mjx

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
        actuators_metadata: dict[str, BaseActuatorMetadata],
        mujoco_mappings: MujocoMappings,
        max_torque: float | None = None,
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

        self.max_torque = max_torque

    def get_ctrl(self, mjx_data: mjx.Data, action: Array) -> Array:
        """Get the control signal from the (position) action vector."""
        current_pos = mjx_data.qpos[7:]  # NOTE: we assume first 7 are always root pos.
        ctrl = self.kps * (action - current_pos)  # TODO: explore using velocity as damping...

        if self.max_torque is not None:
            ctrl = jnp.clip(ctrl, -self.max_torque, self.max_torque)
        return ctrl

    @property
    def actuator_input_size(self) -> int:
        """Get the size of the actuator input vector."""
        return self.kps.shape[0]
