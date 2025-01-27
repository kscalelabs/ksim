"""Defines a state class for general-purpose URDF environments."""

from dataclasses import dataclass

import jax.numpy as jnp

from ksim.state.base import State


@dataclass
class MjcfState(State):
    """Defines a state for a walking robot.

    Tensor name conventions:
    - n: number of environments
    - j: number of degrees of freedom (joints)
    """

    # The base linear velocity.
    base_lin_vel_n3: jnp.ndarray
    # The base angular velocity.
    base_ang_vel_n3: jnp.ndarray
    # The projected gravity vector.
    projected_gravity_n3: jnp.ndarray
    # The global gravity vector.
    global_gravity_n3: jnp.ndarray
    # The actions to be executed.
    actions_nj: jnp.ndarray
    # The last actions executed.
    last_actions_nj: jnp.ndarray
    # The current joint positions.
    dof_pos_nj: jnp.ndarray
    # The current joint velocities.
    dof_vel_nj: jnp.ndarray
    # The previous joint positions.
    last_dof_pos_nj: jnp.ndarray
    # The previous joint velocities.
    last_dof_vel_nj: jnp.ndarray
    # The default joint positions.
    default_dof_pos_nj: jnp.ndarray
    # The default joint velocities.
    default_dof_vel_nj: jnp.ndarray
    # The initial base position.
    base_init_pos_3: jnp.ndarray
    # The initial base quaternion.
    base_init_quat_4: jnp.ndarray
    # The inverse of the initial base quaternion.
    inv_base_init_quat_4: jnp.ndarray
    # The base position.
    base_pos_n3: jnp.ndarray
    # The base quaternion, in the order (x, y, z, w).
    base_quat_n4: jnp.ndarray
    # The base euler angles, in the order (roll, pitch, yaw).
    base_euler_n3: jnp.ndarray
    # Whether the environment has been reset.
    reset_buf_n: jnp.ndarray
    # The current episode length.
    episode_length_n: jnp.ndarray
