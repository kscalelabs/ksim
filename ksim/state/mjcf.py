"""Defines a state class for general-purpose URDF environments."""

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import PRNGKeyArray
from mujoco import mjx

from ksim.state.base import State
from ksim.utils.mujoco import get_sensor_data


@struct.dataclass
class MjcfState(State):
    """Defines a state for a walking robot.

    Tensor name conventions:
    - n: number of environments
    - j: number of degrees of freedom (joints)
    """

    rng: PRNGKeyArray
    model: mjx.Model
    data: mjx.Data
    done: jnp.ndarray

    def get_sensor_data(self, sensor_name: str) -> jnp.ndarray:
        return get_sensor_data(self.model, self.data, sensor_name)

    def get_gravity(self, imu_site_id: int) -> jnp.ndarray:
        return self.data.size_xmat[imu_site_id].T @ jnp.array([0, 0, -1.0])
