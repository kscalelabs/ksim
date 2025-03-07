"""Direct torque control, scaled to a specifiable input range and gear ratio."""

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array
from mujoco import mjx

from ksim.env.mjx.actuators.base_actuator import Actuators, BaseActuatorMetadata
from ksim.utils.mujoco import MujocoMappings


@dataclass
class ScaledTorqueActuatorMetadata(BaseActuatorMetadata):
    input_range: tuple[float, float]
    gear_ratio: float


class ScaledTorqueActuators(Actuators):
    def __init__(
        self,
        actuators_metadata: dict[str, BaseActuatorMetadata],
        mujoco_mappings: MujocoMappings,
    ) -> None:
        """Creates an instance of ScaledTorqueActuators."""
        self.gear_ratios = jnp.array(
            [getattr(metadata, "gear_ratio", 1.0) for metadata in actuators_metadata.values()]
        )
        self.input_ranges = jnp.array(
            [
                getattr(metadata, "input_range", (-1.0, 1.0))
                for metadata in actuators_metadata.values()
            ]
        )

    def get_ctrl(self, mjx_data: mjx.Data, action: Array) -> Array:
        """Get the control signal from the (position) action vector."""
        # Copy legacy and brax setup here
        action_min = self.input_ranges[:, 0]
        action_max = self.input_ranges[:, 1]
        # Scale action from [-1,1] to actuator limits
        ctrl = action * (action_max - action_min) * 0.5
        return ctrl

    @property
    def actuator_input_size(self) -> int:
        """Get the size of the actuator input vector."""
        return self.input_ranges.shape[0]
