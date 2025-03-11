"""Utility functions for transforming between different coordinate systems."""

import jax
import jax.numpy as jnp

from ksim.utils.constants import EPSILON


def quat_to_euler(quat: jax.Array) -> jax.Array:
    """Normalizes and converts a quaternion (w,x,y,z) to euler angles."""
    quat = quat / (jnp.linalg.norm(quat) + EPSILON)
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # Handle edge cases where |sinp| >= 1
    pitch = jnp.where(
        jnp.abs(sinp) >= 1.0,
        jnp.sign(sinp) * jnp.pi / 2.0,  # Use 90 degrees if out of range
        jnp.arcsin(sinp),
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return jnp.array([roll, pitch, yaw])


def get_projected_gravity_vector_from_quat(quat: jax.Array) -> jax.Array:
    """Calculates the gravity vector projected onto the local frame given a quaternion orientation.

    Args:
        quat: A quaternion (w,x,y,z) representing the orientation.

    Returns:
        A 3D vector representing the gravity in the local frame.
    """
    # Normalize quaternion
    quat = quat / (jnp.linalg.norm(quat) + EPSILON)
    w, x, y, z = quat

    # Gravity vector in world frame is [0, 0, -1] (pointing down)
    # Rotate gravity vector using quaternion rotation

    # Calculate quaternion rotation: q * [0,0,-1] * q^-1
    gx = 2 * (x * z - w * y)
    gy = 2 * (y * z + w * x)
    gz = w * w - x * x - y * y + z * z

    # Note: We're rotating [0,0,-1], so we negate gz to match the expected direction
    return jnp.array([gx, gy, -gz])


def get_projected_gravity_vector_from_euler(euler: jax.Array) -> jax.Array:
    """Calculates the gravity vector projected onto the local frame given Euler angles.

    Args:
        euler: An array of [roll, pitch, yaw] angles in radians.

    Returns:
        A 3D vector representing the gravity in the local frame.
    """
    roll, pitch, yaw = euler

    # Create rotation matrices for each axis
    # Roll (X-axis rotation)
    cr, sr = jnp.cos(roll), jnp.sin(roll)
    rx = jnp.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])

    # Pitch (Y-axis rotation)
    cp, sp = jnp.cos(pitch), jnp.sin(pitch)
    ry = jnp.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])

    # Yaw (Z-axis rotation)
    cy, sy = jnp.cos(yaw), jnp.sin(yaw)
    rz = jnp.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])

    # Combine rotations (order: yaw -> pitch -> roll)
    r = rx @ (ry @ rz)

    # Gravity vector in world frame is [0, 0, -1]
    # Apply rotation matrix
    gravity_world = jnp.array([0.0, 0.0, -1.0])
    gravity_local = r @ gravity_world

    return gravity_local
