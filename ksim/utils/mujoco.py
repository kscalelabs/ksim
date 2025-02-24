"""Defines some Mujoco utility functions.

Much of this is referenced from Mujoco Playground.
"""

from dataclasses import dataclass
import logging
from typing import Collection, Dict, Hashable, Sequence, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
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

    return jax.lax.scan(single_step, data, (), num_substeps)[0]


Tk = TypeVar("Tk", bound=Hashable)
Tv = TypeVar("Tv")


def lookup_in_dict(names: Collection[Tk], mapping: dict[Tk, Tv], names_type: str) -> list[Tv]:
    missing_names = [name for name in names if name not in mapping]
    if missing_names:
        available_names_str = "\n".join(str(name) for name in sorted(mapping.keys()))
        raise ValueError(f"{names_type} not found in model: {missing_names}\nAvailable:\n{available_names_str}")
    return [mapping[name] for name in names]


@dataclass
class MujocoMappings:
    """A minimal set of mappings helpful for constructing rewards, terminations, etc."""

    name_to_sensordata: Dict[str, Tuple[int, int | None]]
    name_to_qpos: Dict[str, Tuple[int, int | None]]
    name_to_qvelacc: Dict[str, Tuple[int, int | None]]
    name_to_ctrl: Dict[str, int]
    geom_id_to_body_name: Dict[int, str]


def make_mujoco_mappings(mjx_model: mjx.Model) -> MujocoMappings:
    """Make a MujocoMappings object from a MuJoCo model."""
    # creating sensor mappings
    name_to_sensordata = {}
    for i in range(len(mjx_model.sensor_adr)):
        sensordata_start = mjx_model.sensor_adr[i]
        if i == len(mjx_model.sensor_adr) - 1:
            sensordata_end = None
        else:
            sensordata_end = mjx_model.sensor_adr[i + 1]

        name_start = mjx_model.name_actuatoradr[i]
        name = bytes(mjx_model.names[name_start:]).decode("utf-8").split("\x00")[0]

        name_to_sensordata[name] = (sensordata_start, sensordata_end)

    # doing the same for qpos
    name_to_qpos = {}
    for i in range(len(mjx_model.jnt_qposadr)):
        qpos_start = mjx_model.jnt_qposadr[i]
        if i == len(mjx_model.jnt_qposadr) - 1:
            qpos_end = None
        else:
            qpos_end = mjx_model.jnt_qposadr[i + 1]

        name_start = mjx_model.name_jntadr[i]
        name = bytes(mjx_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        name_to_qpos[name] = (qpos_start, qpos_end)

    # doing the same for qvel
    name_to_qvelacc = {}
    for i in range(len(mjx_model.jnt_dofadr)):
        dof_start = mjx_model.jnt_dofadr[i]
        if i == len(mjx_model.jnt_dofadr) - 1:
            dof_end = None
        else:
            dof_end = mjx_model.jnt_dofadr[i + 1]

        name_start = mjx_model.name_jntadr[i]
        name = bytes(mjx_model.names[name_start:]).decode("utf-8").split("\x00")[0]

        name_to_qvelacc[name] = (dof_start, dof_end)

    # doing the same for ctrl
    name_to_ctrl = {}
    for i in range(len(mjx_model.name_actuatoradr)):
        name_start = mjx_model.name_actuatoradr[i]
        name = bytes(mjx_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        name_to_ctrl[name] = i

    # doing the same for geom_id_to_body_name
    geom_id_to_body_name = {}
    for i in range(len(mjx_model.geom_bodyid)):
        body_id = mjx_model.geom_bodyid[i]

        name_start = mjx_model.name_bodyadr[body_id]
        name = bytes(mjx_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        geom_id_to_body_name[i] = name

    return MujocoMappings(
        name_to_sensordata,
        name_to_qpos,
        name_to_qvelacc,
        name_to_ctrl,
        geom_id_to_body_name,
    )


def get_qpos_from_name(name: str, mujoco_mappings: MujocoMappings, data: mjx.Data) -> jnp.ndarray:
    """Get the qpos from a name."""
    return data.qpos[mujoco_mappings.name_to_qpos[name]]


def get_qvel_from_name(name: str, mujoco_mappings: MujocoMappings, data: mjx.Data) -> jnp.ndarray:
    """Get the qvel from a name."""
    return data.qvel[mujoco_mappings.name_to_qvelacc[name]]


def get_ctrl_from_name(name: str, mujoco_mappings: MujocoMappings, data: mjx.Data) -> jnp.ndarray:
    """Get the ctrl from a name."""
    return data.ctrl[mujoco_mappings.name_to_ctrl[name]]


def get_sensordata_from_name(name: str, mujoco_mappings: MujocoMappings, data: mjx.Data) -> jnp.ndarray:
    """Get the sensordata from a name."""
    return data.sensordata[mujoco_mappings.name_to_sensordata[name]]


def is_body_in_contact(
    body_name: str,
    mujoco_mappings: MujocoMappings,
    data: mjx.Data,
) -> bool:
    """Check if a body is in contact."""
    # TODO: implement this properly...
    return False
