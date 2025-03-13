"""Vanilla torque control actuator."""

from typing import Mapping, TypedDict

import jax.numpy as jnp
from jaxtyping import Array
from mujoco import mjx

from ksim.actuators.base import Actuators
from ksim.utils.mujoco import MujocoMappings


class TorqueActuators(Actuators):
    def get_ctrl(self, mjx_data: mjx.Data, action: Array) -> Array:
        # Just use the action as the torque.
        return action
