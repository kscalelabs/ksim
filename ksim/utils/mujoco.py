"""Defines some Mujoco utility functions.

Much of this is referenced from Mujoco Playground.
"""

import logging
from typing import Collection, Hashable, Sequence, TypeVar, Union

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


def get_sensor_data(
    model: mujoco.MjModel,
    data: mjx.Data,
    sensor_name: str,
) -> jnp.ndarray:
    """Gets sensor data given sensor name.

    Args:
        model: The MuJoCo model.
        data: The MuJoCo data.
        sensor_name: The name of the sensor to get data from.

    Returns:
        The sensor data.
    """
    sensor_id = model.sensor(sensor_name).id
    sensor_adr = model.sensor_adr[sensor_id]
    sensor_dim = model.sensor_dim[sensor_id]
    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]


def dof_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
    """Get the dimensionality of the joint in qvel.

    Args:
        joint_type: The type of the joint.

    Returns:
        The dimensionality of the joint in qvel.
    """
    if isinstance(joint_type, mujoco.mjtJoint):
        joint_type = joint_type.value
    return {0: 6, 1: 3, 2: 1, 3: 1}[joint_type]


def qpos_width(joint_type: Union[int, mujoco.mjtJoint]) -> int:
    """Get the dimensionality of the joint in qpos.

    Args:
        joint_type: The type of the joint.

    Returns:
        The dimensionality of the joint in qpos.
    """
    if isinstance(joint_type, mujoco.mjtJoint):
        joint_type = joint_type.value
    return {0: 7, 1: 4, 2: 1, 3: 1}[joint_type]


def get_qpos_ids(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    """Get the indices of the qpos for a list of joints.

    Args:
        model: The MuJoCo model.
        joint_names: The names of the joints.

    Returns:
        The indices of the qpos for the joints.
    """
    index_list: list[int] = []
    for jnt_name in joint_names:
        jnt = model.joint(jnt_name).id
        jnt_type = model.jnt_type[jnt]
        qadr = model.jnt_dofadr[jnt]
        qdim = qpos_width(jnt_type)
        index_list.extend(range(qadr, qadr + qdim))
    return np.array(index_list)


def get_qvel_ids(model: mujoco.MjModel, joint_names: Sequence[str]) -> np.ndarray:
    """Get the indices of the qvel for a list of joints.

    Args:
        model: The MuJoCo model.
        joint_names: The names of the joints.

    Returns:
        The indices of the qvel for the joints.
    """
    index_list: list[int] = []
    for jnt_name in joint_names:
        jnt = model.joint(jnt_name).id
        jnt_type = model.jnt_type[jnt]
        vadr = model.jnt_dofadr[jnt]
        vdim = dof_width(jnt_type)
        index_list.extend(range(vadr, vadr + vdim))
    return np.array(index_list)
