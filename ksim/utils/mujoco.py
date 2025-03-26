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
]

import itertools
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


def get_body_data_idx_by_name(physics_model: PhysicsModel) -> dict[str, int]:
    """Get mappings from body names to their indices."""
    body_mappings = {}
    for i in range(physics_model.nbody):
        name_start = physics_model.name_bodyadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        body_mappings[name] = i
    return body_mappings


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


def slice_update(model: mujoco.MjModel | mjx.Model, name: str, slice: Any, value: Array) -> Array:  # noqa: ANN401
    """Update a slice of a model field."""
    if isinstance(model, mujoco.MjModel):
        val = getattr(model, name).copy()
        val[slice] = value
        return val
    if isinstance(model, mjx.Model):
        return getattr(model, name).at[slice].set(value)
    raise ValueError(f"Model type {type(model)} not supported")


def validate_model(mj: mujoco.MjModel) -> None:
    """Checks if a MuJoCo model is compatible with brax physics pipelines."""
    if mj.opt.integrator != 0:
        raise NotImplementedError("Only euler integration is supported.")
    if mj.opt.cone != 0:
        raise NotImplementedError("Only pyramidal cone friction is supported.")
    if (mj.geom_fluid != 0).any():
        raise NotImplementedError("Ellipsoid fluid model not implemented.")
    if mj.opt.wind.any():
        raise NotImplementedError("option.wind is not implemented.")
    if mj.opt.impratio != 1:
        raise NotImplementedError("Only impratio=1 is supported.")

    # actuators
    if any(i not in [0, 1] for i in mj.actuator_biastype):
        raise NotImplementedError("Only actuator_biastype in [0, 1] are supported.")
    if any(i != 0 for i in mj.actuator_gaintype):
        raise NotImplementedError("Only actuator_gaintype in [0] is supported.")
    if not (mj.actuator_trntype == 0).all():
        raise NotImplementedError("Only joint transmission types are supported for actuators.")

    # solver parameters
    if (mj.geom_solmix[0] != mj.geom_solmix).any():
        raise NotImplementedError("geom_solmix parameter not supported.")
    if (mj.geom_priority[0] != mj.geom_priority).any():
        raise NotImplementedError("geom_priority parameter not supported.")

    # check joints
    q_width = {0: 7, 1: 4, 2: 1, 3: 1}
    non_free = np.concatenate([[j != 0] * q_width[j] for j in mj.jnt_type])
    if mj.qpos0[non_free].any():
        raise NotImplementedError("The `ref` attribute on joint types is not supported.")

    for _, group in itertools.groupby(zip(mj.jnt_bodyid, mj.jnt_pos), key=lambda x: x[0]):
        position = np.array([p for _, p in group])
        if not (position == position[0]).all():
            raise RuntimeError("invalid joint stack: only one joint position allowed")

    # check dofs
    jnt_range = mj.jnt_range.copy()
    jnt_range[~(mj.jnt_limited == 1), :] = np.array([-np.inf, np.inf])
    for typ, limit, stiffness in zip(mj.jnt_type, jnt_range, mj.jnt_stiffness):
        if typ == 0:
            if stiffness > 0:
                raise RuntimeError("brax does not support stiffness for free joints")
        elif typ == 1:
            if np.any(~np.isinf(limit)):
                raise RuntimeError("brax does not support joint ranges for ball joints")
        elif typ in (2, 3):
            continue
        else:
            raise RuntimeError(f"invalid joint type: {typ}")

    for _, group in itertools.groupby(zip(mj.jnt_bodyid, mj.jnt_type), key=lambda x: x[0]):
        typs = [t for _, t in group]
        if len(typs) == 1 and typs[0] == 0:
            continue  # free
        elif 0 in typs:
            raise RuntimeError("invalid joint stack: cannot stack free joints")
        elif 1 in typs:
            raise NotImplementedError("ball joints not supported")

    # check collision geometries
    for i, typ in enumerate(mj.geom_type):
        mask = mj.geom_contype[i] | mj.geom_conaffinity[i] << 32
        if typ == 5:  # Cylinder
            _, halflength = mj.geom_size[i, 0:2]
            if halflength > 0.001 and mask > 0:
                raise NotImplementedError("Cylinders of half-length>0.001 are not supported for collision.")


def load_model(model: mujoco.MjModel) -> mjx.Model:
    validate_model(model)
    mjx_model = mjx.put_model(model)
    mjx_model = jax.tree.map(jnp.array, mjx_model)
    return mjx_model
