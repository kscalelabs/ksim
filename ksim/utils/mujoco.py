"""Defines some Mujoco utility functions."""

__all__ = [
    "get_sensor_data_idxs_by_name",
    "get_qpos_data_idxs_by_name",
    "get_qvelacc_data_idxs_by_name",
    "get_ctrl_data_idx_by_name",
    "get_geom_data_idx_by_name",
    "get_body_data_idx_by_name",
    "get_floor_idx",
    "get_collision_info",
    "geoms_colliding",
    "get_joint_metadata",
    "update_model_field",
    "update_data_field",
    "slice_update",
    "quat_to_mat",
    "mat_to_quat",
    "get_body_pose",
    "get_geom_pose",
    "get_site_pose",
    "get_body_pose_by_name",
    "get_geom_pose_by_name",
    "get_site_pose_by_name",
]

import logging
from typing import Any, Hashable, TypeVar

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jaxtyping import Array, PyTree
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx

from ksim.types import PhysicsData, PhysicsModel

logger = logging.getLogger(__name__)

Tk = TypeVar("Tk", bound=Hashable)
Tv = TypeVar("Tv")


def get_sensor_data_idxs_by_name(physics_model: PhysicsModel) -> dict[str, tuple[int, int | None]]:
    """Get mappings from sensor names to their data indices."""
    sensor_mappings = {}
    for i in range(len(physics_model.sensor_adr)):
        start = physics_model.sensor_adr[i]
        end = physics_model.sensor_adr[i + 1] if i < len(physics_model.sensor_adr) - 1 else None

        name_start = physics_model.name_sensoradr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        sensor_mappings[name] = (start, end)
    return sensor_mappings


def get_qpos_data_idxs_by_name(physics_model: PhysicsModel) -> dict[str, tuple[int, int | None]]:
    """Get mappings from joint names to their position indices."""
    qpos_mappings = {}
    for i in range(len(physics_model.jnt_qposadr)):
        start = physics_model.jnt_qposadr[i]
        end = physics_model.jnt_qposadr[i + 1] if i < len(physics_model.jnt_qposadr) - 1 else None

        name_start = physics_model.name_jntadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        qpos_mappings[name] = (start, end)
    return qpos_mappings


def get_qvelacc_data_idxs_by_name(physics_model: PhysicsModel) -> dict[str, tuple[int, int | None]]:
    """Get mappings from joint names to their velocity/acceleration indices."""
    qvelacc_mappings = {}
    for i in range(len(physics_model.jnt_dofadr)):
        start = physics_model.jnt_dofadr[i]
        end = physics_model.jnt_dofadr[i + 1] if i < len(physics_model.jnt_dofadr) - 1 else None

        name_start = physics_model.name_jntadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        qvelacc_mappings[name] = (start, end)
    return qvelacc_mappings


def get_ctrl_data_idx_by_name(physics_model: PhysicsModel) -> dict[str, int]:
    """Get mappings from actuator names to their control indices."""
    ctrl_mappings = {}
    for i in range(len(physics_model.name_actuatoradr)):
        name_start = physics_model.name_actuatoradr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        ctrl_mappings[name] = i
    return ctrl_mappings


def get_geom_data_idx_by_name(physics_model: PhysicsModel) -> dict[str, int]:
    """Get mappings from geometry names to their indices."""
    geom_mappings = {}
    for i in range(physics_model.ngeom):
        name_start = physics_model.name_geomadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        geom_mappings[name] = i
    return geom_mappings


def get_geom_data_idx_from_name(physics_model: PhysicsModel, geom_name: str) -> int:
    for i in range(physics_model.ngeom):
        name_start = physics_model.name_geomadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        if name == geom_name:
            return i
    raise KeyError(f"Geometry '{geom_name}' not found in model")


def get_site_data_idx_from_name(physics_model: PhysicsModel, site_name: str) -> int:
    """Get mappings from site names to their indices."""
    for i in range(physics_model.nsite):
        name_start = physics_model.name_siteadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        if name == site_name:
            return i
    raise KeyError(f"Site '{site_name}' not found in model")


def get_body_data_idx_by_name(physics_model: PhysicsModel) -> dict[str, int]:
    """Get mappings from body names to their indices."""
    body_mappings = {}
    for i in range(physics_model.nbody):
        name_start = physics_model.name_bodyadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        body_mappings[name] = i
    return body_mappings


def get_body_data_idx_from_name(physics_model: PhysicsModel, body_name: str) -> int:
    for i in range(physics_model.nbody):
        name_start = physics_model.name_bodyadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        if name == body_name:
            return i
    raise KeyError(f"Body '{body_name}' not found in model")


def get_floor_idx(physics_model: PhysicsModel, floor_name: str = "floor") -> int | None:
    """Get the index of the floor geometry."""
    geom_mappings = get_geom_data_idx_by_name(physics_model)
    assert floor_name in geom_mappings, f"Floor name {floor_name} not found in model"
    return geom_mappings[floor_name]


def get_collision_info(contact: PyTree, geom1: int, geom2: int) -> tuple[jax.Array, jax.Array]:
    """Get the distance and normal of the collision between two geoms."""
    mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = jnp.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    # This reshape is nedded because contact.frame's shape depends on how many envs there are.
    normal = (dist < 0) * jnp.reshape(contact.frame[idx], (-1,))[:3]
    return dist, normal


def geoms_colliding(state: PhysicsData, geom1: int, geom2: int) -> jax.Array:
    """Return True if the two geoms are colliding."""
    return jax.lax.cond(
        jnp.equal(state.contact.geom.shape[0], 0),  # if no contacts, return False
        lambda _: jnp.array(False),
        lambda _: get_collision_info(state.contact, geom1, geom2)[0] < 0,
        operand=None,
    )


def get_joint_metadata(
    model: PhysicsModel,
    kp: float | None = None,
    kd: float | None = None,
    armature: float | None = None,
    friction: float | None = None,
) -> dict[str, JointMetadataOutput]:
    """Get the joint metadata from the model."""
    metadata = {}
    for i in range(model.njnt):
        name_start = model.name_jntadr[i]
        name = bytes(model.names[name_start:]).decode("utf-8").split("\x00")[0]
        metadata[name] = JointMetadataOutput(
            kp=None if kp is None else str(kp),
            kd=None if kd is None else str(kd),
            armature=None if armature is None else str(armature),
            friction=None if friction is None else str(friction),
            id=None,
            flipped=None,
            offset=None,
        )
    return metadata


def update_model_field(model: mujoco.MjModel | mjx.Model, name: str, new_value: Array) -> mujoco.MjModel | mjx.Model:
    """Handles the update of a model field in a mujoco.MjModel or mjx.Model object.

    This is necessary because Mujoco handles model elements as pointers, while
    MJX model elements are stateful.
    """
    if isinstance(model, mujoco.MjModel):
        getattr(model, name)[:] = new_value
    else:
        model = model.replace(**{name: new_value})
    return model


def update_data_field(data: mujoco.MjData | mjx.Data, name: str, new_value: Array) -> mujoco.MjData | mjx.Data:
    """Handles the update of a data field in a mujoco.MjData or mjx.Data object.

    This is necessary because Mujoco handles data elements as pointers, while
    MJX data elements are stateful.

    Args:
        data: The mujoco.MjData or mjx.Data object to update.
        name: The name of the field to update.
        new_value: The new value to set the field to.

    Returns:
        The updated mujoco.MjData or mjx.Data object.
    """
    if isinstance(data, mujoco.MjData):
        getattr(data, name)[:] = new_value
    else:
        data = data.replace(**{name: new_value})
    return data


def slice_update(
    model: mujoco.MjModel | mjx.Model | mujoco.MjData | mjx.Data,
    name: str,
    slice: Any,  # noqa: ANN401
    value: Array,
) -> Array:
    """Update a slice of a model field."""
    if isinstance(model, (mujoco.MjModel, mujoco.MjData)):
        val = getattr(model, name).copy()
        val[slice] = value
        return val
    if isinstance(model, (mjx.Model, mjx.Data)):
        return getattr(model, name).at[slice].set(value)
    raise ValueError(f"Model type {type(model)} not supported")


def load_model(model: mujoco.MjModel) -> mjx.Model:
    mjx_model = mjx.put_model(model)
    mjx_model = jax.tree.map(jnp.array, mjx_model)
    return mjx_model


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    rot_mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rot_mat, np.array(quat, dtype=np.float64))
    rot_mat = rot_mat.reshape(3, 3)
    return rot_mat


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.array(mat, np.float64).flatten())
    return quat


def get_body_pose(data: PhysicsData, body_idx: int) -> tuple[np.ndarray, np.ndarray]:
    position = data.xpos[body_idx].copy()
    quat = data.xquat[body_idx].copy()
    rot_mat = quat_to_mat(quat)
    return position, rot_mat


def get_geom_pose(data: PhysicsData, geom_idx: int) -> tuple[np.ndarray, np.ndarray]:
    position = data.geom_xpos[geom_idx].copy()
    rot_mat = data.geom_xmat[geom_idx].reshape(3, 3).copy()
    return position, rot_mat


def get_site_pose(data: PhysicsData, site_idx: int) -> tuple[np.ndarray, np.ndarray]:
    position = data.site_xpos[site_idx].copy()
    rot_mat = data.site_xmat[site_idx].reshape(3, 3).copy()
    return position, rot_mat


def get_body_pose_by_name(
    model: PhysicsModel,
    data: PhysicsData,
    body_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    body_idx = get_body_data_idx_from_name(model, body_name)
    return get_body_pose(data, body_idx)


def get_geom_pose_by_name(
    model: PhysicsModel,
    data: PhysicsData,
    geom_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    geom_idx = get_geom_data_idx_from_name(model, geom_name)
    return get_geom_pose(data, geom_idx)


def get_site_pose_by_name(
    model: PhysicsModel,
    data: PhysicsData,
    site_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    site_idx = get_site_data_idx_from_name(model, site_name)
    return get_site_pose(data, site_idx)
