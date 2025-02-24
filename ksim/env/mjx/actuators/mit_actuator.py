"""MIT Controller, as used by the Robstride actuators."""

from dataclasses import dataclass
from typing import Any, Dict

import jax.numpy as jnp
from jaxtyping import Array

from ksim.env.mjx.actuators.base_actuator import Actuators, BaseActuatorMetadata
from ksim.env.mjx.mjx_env import MjxEnvState
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
        self.kps = jnp.zeros(len(mujoco_mappings.name_to_ctrl))
        self.kds = jnp.zeros(len(mujoco_mappings.name_to_ctrl))

        for actuator_name, actuator_metadata in actuators_metadata.items():
            assert isinstance(actuator_metadata, MITPositionActuatorMetadata)
            actuator_idx = mujoco_mappings.name_to_ctrl[actuator_name]
            self.kps[actuator_idx] = actuator_metadata.kp
            self.kds[actuator_idx] = actuator_metadata.kd

        if any(self.kps == 0) or any(self.kds == 0):
            raise ValueError("Some kps or kds are 0. Check your actuators_metadata.")

    def get_ctrl(self, env_state: MjxEnvState, action: Array) -> Array:
        """Get the control signal from the (position) action vector."""
        current_pos = env_state.mjx_data.qpos[7:]  # NOTE: we assume first 7 are always root pos.
        ctrl = self.kps * (action - current_pos)  # TODO: explore using velocity as damping...
        return ctrl
