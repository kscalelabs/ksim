"""Defines some Mujoco utility functions.

Much of this is referenced from Mujoco Playground.
"""

import logging
from dataclasses import dataclass
from typing import Collection, Hashable, TypeVar

import jax
import jax.numpy as jnp
from mujoco import mjx

logger = logging.getLogger(__name__)


def init(
    model: mjx.Model,
    qpos_j: jnp.ndarray | None = None,
    qvel_j: jnp.ndarray | None = None,
    ctrl_j: jnp.ndarray | None = None,
    act_j: jnp.ndarray | None = None,
    mocap_pos_m3: jnp.ndarray | None = None,
    mocap_quat_m4: jnp.ndarray | None = None,
) -> mjx.Data:
    """Initialize MJX data.

    The tensor conventions here are:

    - n: number of environments
    - j: number of joints
    - m: number of mocap bodies

    Args:
        model: The MuJoCo model.
        qpos_j: The initial joint positions, of shape (n, j).
        qvel_j: The initial joint velocities, of shape (n, j).
        ctrl_j: The initial joint controls, of shape (n, j).
        act_j: The initial actions, of shape (n, j).
        mocap_pos_m3: The initial mocap positions, of shape (n, m, 3).
            Mocap is a special type of joint that allows for arbitrary
            3D positions.
        mocap_quat_m4: The initial mocap quaternions, of shape (n, m, 4).

    Returns:
        The initialized MJX data.
    """
    logger.debug("Initializing MJX data")
    data = mjx.make_data(model)
    if qpos_j is not None:
        data = data.replace(qpos=qpos_j)
    if qvel_j is not None:
        data = data.replace(qvel=qvel_j)
    if ctrl_j is not None:
        data = data.replace(ctrl=ctrl_j)
    if act_j is not None:
        data = data.replace(act=act_j)
    if mocap_pos_m3 is not None:
        data = data.replace(mocap_pos=mocap_pos_m3.reshape(model.nmocap, -1))
    if mocap_quat_m4 is not None:
        data = data.replace(mocap_quat=mocap_quat_m4.reshape(model.nmocap, -1))

    logger.debug("Forwarding MJX data")
    data = mjx.forward(model, data)
    logger.debug("MJX data initialized")
    return data


def step(
    model: mjx.Model,
    data: mjx.Data,
    action: jnp.ndarray,
    num_substeps: int = 1,
) -> mjx.Data:
    """Step the physics forward.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data.
        action: The action to take.
        num_substeps: The number of substeps to take.

    Returns:
        The updated MuJoCo data.
    """

    def single_step(data: mjx.Data, _: None) -> tuple[mjx.Data, None]:
        data = data.replace(ctrl=action)
        data = mjx.step(model, data)
        return data, None

    return jax.lax.scan(f=single_step, init=data, xs=None, length=num_substeps)[0]


Tk = TypeVar("Tk", bound=Hashable)
Tv = TypeVar("Tv")


def lookup_in_dict(names: Collection[Tk], mapping: dict[Tk, Tv], names_type: str) -> list[Tv]:
    """Lookup a list of names in a dictionary and return the corresponding values."""
    missing_names = [name for name in names if name not in mapping]
    if missing_names:
        available_names_str = "\n".join(name for name in sorted(str(lookup) for lookup in mapping.keys()))
        raise ValueError(f"{names_type} not found in model: {missing_names}\nAvailable:\n{available_names_str}")
    return [mapping[name] for name in names]


@dataclass
class MujocoMappings:
    """A minimal set of mappings helpful for constructing rewards, terminations, etc."""

    sensor_name_to_idx_range: dict[str, tuple[int, int | None]]
    qpos_name_to_idx_range: dict[str, tuple[int, int | None]]
    qvelacc_name_to_idx_range: dict[str, tuple[int, int | None]]
    ctrl_name_to_idx: dict[str, int]
    geom_name_to_idx: dict[str, int]
    body_name_to_idx: dict[str, int]
    floor_geom_idx: int | None


def make_mujoco_mappings(mjx_model: mjx.Model, floor_name: str = "floor") -> MujocoMappings:
    """Make a MujocoMappings object from a MuJoCo model."""
    # creating sensor mappings
    sensor_name_to_idx_range = {}
    for i in range(len(mjx_model.sensor_adr)):
        sensordata_start = mjx_model.sensor_adr[i]
        if i == len(mjx_model.sensor_adr) - 1:
            sensordata_end = None
        else:
            sensordata_end = mjx_model.sensor_adr[i + 1]

        name_start = mjx_model.name_sensoradr[i]
        name = bytes(mjx_model.names[name_start:]).decode("utf-8").split("\x00")[0]

        sensor_name_to_idx_range[name] = (sensordata_start, sensordata_end)

    # doing the same for qpos
    qpos_name_to_idx_range = {}
    for i in range(len(mjx_model.jnt_qposadr)):
        qpos_start = mjx_model.jnt_qposadr[i]
        if i == len(mjx_model.jnt_qposadr) - 1:
            qpos_end = None
        else:
            qpos_end = mjx_model.jnt_qposadr[i + 1]

        name_start = mjx_model.name_jntadr[i]
        name = bytes(mjx_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        qpos_name_to_idx_range[name] = (qpos_start, qpos_end)

    # doing the same for qvel/acc
    qvelacc_name_to_idx_range = {}
    for i in range(len(mjx_model.jnt_dofadr)):
        dof_start = mjx_model.jnt_dofadr[i]
        if i == len(mjx_model.jnt_dofadr) - 1:
            dof_end = None
        else:
            dof_end = mjx_model.jnt_dofadr[i + 1]

        name_start = mjx_model.name_jntadr[i]
        name = bytes(mjx_model.names[name_start:]).decode("utf-8").split("\x00")[0]

        qvelacc_name_to_idx_range[name] = (dof_start, dof_end)

    # doing the same for ctrl
    ctrl_name_to_idx = {}
    for i in range(len(mjx_model.name_actuatoradr)):
        name_start = mjx_model.name_actuatoradr[i]
        name = bytes(mjx_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        ctrl_name_to_idx[name] = i

    geom_name_to_idx = {}
    for i in range(mjx_model.ngeom):
        name_start = mjx_model.name_geomadr[i]
        name = bytes(mjx_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        geom_name_to_idx[name] = i

    body_name_to_idx = {}
    for i in range(mjx_model.nbody):
        name_start = mjx_model.name_bodyadr[i]
        name = bytes(mjx_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        body_name_to_idx[name] = i

    floor_idx = None

    try:
        floor_idx = geom_name_to_idx[floor_name]
    except KeyError:
        logger.warning("No floor geom %s found in model", floor_name)

    return MujocoMappings(
        sensor_name_to_idx_range=sensor_name_to_idx_range,
        qpos_name_to_idx_range=qpos_name_to_idx_range,
        qvelacc_name_to_idx_range=qvelacc_name_to_idx_range,
        ctrl_name_to_idx=ctrl_name_to_idx,
        geom_name_to_idx=geom_name_to_idx,
        body_name_to_idx=body_name_to_idx,
        floor_geom_idx=floor_idx,
    )


def get_qpos_from_name(name: str, mujoco_mappings: MujocoMappings, data: mjx.Data) -> jnp.ndarray:
    """Get the qpos from a name."""
    return data.qpos[mujoco_mappings.qpos_name_to_idx_range[name]]


def get_qvel_from_name(name: str, mujoco_mappings: MujocoMappings, data: mjx.Data) -> jnp.ndarray:
    """Get the qvel from a name."""
    return data.qvel[mujoco_mappings.qvelacc_name_to_idx_range[name]]


def get_ctrl_from_name(name: str, mujoco_mappings: MujocoMappings, data: mjx.Data) -> jnp.ndarray:
    """Get the ctrl from a name."""
    return data.ctrl[mujoco_mappings.ctrl_name_to_idx[name]]


def get_sensordata_from_name(name: str, mujoco_mappings: MujocoMappings, data: mjx.Data) -> jnp.ndarray:
    """Get the sensordata from a name."""
    return data.sensordata[mujoco_mappings.sensor_name_to_idx_range[name]]


def is_body_in_contact(
    body_name: str,
    mujoco_mappings: MujocoMappings,
    data: mjx.Data,
) -> bool:
    """Check if a body is in contact."""
    # TODO: implement this properly...
    return False


def get_collision_info(contact: mjx.Contact, geom1: int, geom2: int) -> tuple[jax.Array, jax.Array]:
    """Get the distance and normal of the collision between two geoms."""
    mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = jnp.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    # This reshape is nedded because contact.frame's shape depends on how many envs there are.
    normal = (dist < 0) * jnp.reshape(contact.frame[idx], (-1,))[:3]
    return dist, normal


def geoms_colliding(state: mjx.Data, geom1: int, geom2: int) -> jax.Array:
    """Return True if the two geoms are colliding."""
    return jax.lax.cond(
        jnp.equal(state.contact.geom.shape[0], 0),  # if no contacts, return False
        lambda _: jnp.array(False),
        lambda _: get_collision_info(state.contact, geom1, geom2)[0] < 0,
        operand=None,
    )
