"""Defines some Mujoco utility functions."""

__all__ = [
    "get_sensor_data_idxs_by_name",
    "get_qpos_data_idxs_by_name",
    "get_qvelacc_data_idxs_by_name",
    "get_ctrl_data_idx_by_name",
    "get_geom_data_idx_by_name",
    "get_geom_data_idx_from_name",
    "get_body_data_idx_by_name",
    "get_body_data_idx_from_name",
    "get_site_data_idx_from_name",
    "get_floor_idx",
    "geoms_colliding",
    "get_joint_names_in_order",
    "get_position_limits",
    "get_torque_limits",
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
    "remove_mujoco_joints_except",
    "remove_first_joint",
    "add_new_mujoco_body",
    "log_joint_config_table",
]

import logging
from collections import deque
from typing import Any, Hashable, TypeVar
from xml.etree import ElementTree as ET

import chex
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import xax
from jaxtyping import Array
from mujoco import mjx
from tabulate import tabulate

from ksim.types import Metadata, PhysicsData, PhysicsModel

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
    choices = ", ".join(sorted(get_geom_data_idx_by_name(physics_model).keys()))
    raise KeyError(f"Geometry '{geom_name}' not found in model. Choices: {choices}")


def get_site_data_idx_by_name(physics_model: PhysicsModel) -> dict[str, int]:
    """Get mappings from site names to their indices."""
    site_mappings = {}
    for i in range(physics_model.nsite):
        name_start = physics_model.name_siteadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        site_mappings[name] = i
    return site_mappings


def get_site_data_idx_from_name(physics_model: PhysicsModel, site_name: str) -> int:
    """Get mappings from site names to their indices."""
    for i in range(physics_model.nsite):
        name_start = physics_model.name_siteadr[i]
        name = bytes(physics_model.names[name_start:]).decode("utf-8").split("\x00")[0]
        if name == site_name:
            return i
    choices = ", ".join(sorted(get_site_data_idx_by_name(physics_model).keys()))
    raise KeyError(f"Site '{site_name}' not found in model. Choices: {choices}")


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
    choices = ", ".join(sorted(get_body_data_idx_by_name(physics_model).keys()))
    raise KeyError(f"Body '{body_name}' not found in model. Choices: {choices}")


def get_floor_idx(physics_model: PhysicsModel, floor_name: str = "floor") -> int | None:
    """Get the index of the floor geometry."""
    geom_mappings = get_geom_data_idx_by_name(physics_model)
    assert floor_name in geom_mappings, f"Floor name {floor_name} not found in model"
    return geom_mappings[floor_name]


def geoms_colliding(state: PhysicsData, geom1: Array, geom2: Array) -> Array:
    """Return True if the geoms are colliding."""
    chex.assert_shape(geom1, (None,))
    chex.assert_shape(geom2, (None,))

    def get_colliding_inner(geom: Array, dist: Array, geom1: Array, geom2: Array) -> Array:
        # Create all pairs of geom indices
        g1, g2 = jnp.meshgrid(geom1, geom2, indexing="ij")

        def when_contacts_exist() -> Array:
            # geom shape is (n_contacts, 2)
            contact_geoms = geom
            contact_dists = dist  # shape (n_contacts,)

            g1_expanded = g1[..., jnp.newaxis]  # Shape: (n_geom1, n_geom2, 1)
            g2_expanded = g2[..., jnp.newaxis]  # Shape: (n_geom1, n_geom2, 1)

            # Expand contact_geoms to shape (1, 1, n_contacts, 2)
            contacts_expanded = contact_geoms[jnp.newaxis, jnp.newaxis, :]  # Shape: (1, 1, n_contacts, 2)

            # Check for matches in both orders (forward and reversed)
            forward_match = (g1_expanded == contacts_expanded[..., 0]) & (g2_expanded == contacts_expanded[..., 1])
            reverse_match = (g1_expanded == contacts_expanded[..., 1]) & (g2_expanded == contacts_expanded[..., 0])

            any_match = forward_match | reverse_match  # Shape: (n_geom1, n_geom2, n_contacts)

            negative_dist = contact_dists < 0  # Shape: (n_contacts,)
            collision_mask = (
                any_match & negative_dist[jnp.newaxis, jnp.newaxis, :]
            )  # Shape: (n_geom1, n_geom2, n_contacts)

            is_colliding = collision_mask.any(axis=2)  # Shape: (n_geom1, n_geom2)

            return is_colliding

        return jax.lax.cond(
            geom.shape[0] == 0,
            lambda _: jnp.zeros(g1.shape, dtype=jnp.bool_),
            lambda _: when_contacts_exist(),
            operand=None,
        )

    return get_colliding_inner(state.contact.geom, state.contact.dist, geom1, geom2)


def get_joint_names_in_order(model: PhysicsModel) -> list[str]:
    """Get the joint names in order of their indices."""
    return [bytes(model.names[model.name_jntadr[i] :]).decode("utf-8").split("\x00")[0] for i in range(model.njnt)]


def get_position_limits(model: PhysicsModel) -> dict[str, tuple[float, float]]:
    """Get the ranges of the joints."""
    ranges = {}
    for i in range(model.njnt):
        name_start = model.name_jntadr[i]
        name = bytes(model.names[name_start:]).decode("utf-8").split("\x00")[0]
        ranges[name] = (float(model.jnt_range[i, 0]), float(model.jnt_range[i, 1]))
    return ranges


def get_torque_limits(model: PhysicsModel) -> dict[str, tuple[float, float]]:
    """Get the torque limits of the joints."""
    ranges = {}
    for i in range(model.njnt):
        name_start = model.name_jntadr[i]
        name = bytes(model.names[name_start:]).decode("utf-8").split("\x00")[0]
        ranges[name] = (float(model.jnt_actfrcrange[i, 0]), float(model.jnt_actfrcrange[i, 1]))
    return ranges


def log_joint_config_table(
    model: PhysicsModel,
    metadata: Metadata,
    xax_logger: xax.Logger,
) -> None:
    """Log configuration of joints and actuators in a table."""
    actuator_name_to_nn_id = get_ctrl_data_idx_by_name(model)
    joint_names = get_joint_names_in_order(model)
    joint_limits = get_position_limits(model)

    if metadata.joint_name_to_metadata is None:
        raise ValueError("Joint metadata is required for the joint config table")
    joint_metadata = metadata.joint_name_to_metadata

    # The \n is to make the table headers take up less horizontal space.
    headers = [
        "Joint Name",
        "Type",
        "Kp",
        "Kd",
        "Soft\nTorque\nLimit",
        "Dmp",
        "Arm",
        "Fric",
        "Joint\nLimit\nMin",
        "Joint\nLimit\nMax",
        "Ctrl\nMin",
        "Ctrl\nMax",
        "Force\nLimited",
        "Force\nMin",
        "Force\nMax",
        "Joint\nIdx",
        "DOF\nIdx",
    ]

    joint_data = []
    for joint_idx, joint_name in enumerate(joint_names):
        # Skip floating base and/or root joint as they are not actuated.
        if joint_name in {"floating_base", "root"}:
            continue

        dof_id = model.jnt_dofadr[joint_idx]
        actuator_name = f"{joint_name}_ctrl"

        # Checks for errors in the joint configuration setup
        if actuator_name not in actuator_name_to_nn_id:
            raise ValueError(f"Actuator {actuator_name} not found in model")
        actuator_nn_id = actuator_name_to_nn_id[actuator_name]

        if joint_name not in joint_metadata:
            raise ValueError(f"Joint {joint_name} not found in metadata")
        joint_meta = joint_metadata[joint_name]

        if joint_meta.soft_torque_limit is not None:
            stl_float = float(joint_meta.soft_torque_limit)
            if model.jnt_actfrclimited[joint_idx]:
                frcrange_float = float(model.jnt_actfrcrange[joint_idx][1])
                if stl_float > frcrange_float:
                    raise ValueError(f"Soft torque limit {stl_float} > max {frcrange_float} for {joint_name}")
        else:
            logger.debug("Joint %s has no soft torque limit", joint_name)

        joint_data.append(
            {
                "Joint Name": joint_name,
                "Type": joint_meta.actuator_type,
                "Kp": joint_meta.kp,
                "Kd": joint_meta.kd,
                "Soft\nTorque\nLimit": (
                    joint_meta.soft_torque_limit if joint_meta.soft_torque_limit is not None else "None"
                ),
                "Dmp": model.dof_damping[dof_id],
                "Arm": model.dof_armature[dof_id],
                "Fric": model.dof_frictionloss[dof_id],
                "Joint\nLimit\nMin": f"{joint_limits[joint_name][0]:.3f}",
                "Joint\nLimit\nMax": f"{joint_limits[joint_name][1]:.3f}",
                "Ctrl\nMin": model.actuator_ctrlrange[actuator_nn_id][0],
                "Ctrl\nMax": model.actuator_ctrlrange[actuator_nn_id][1],
                "Force\nLimited": model.jnt_actfrclimited[joint_idx],
                "Force\nMin": model.jnt_actfrcrange[joint_idx][0],
                "Force\nMax": model.jnt_actfrcrange[joint_idx][1],
                "Joint\nIdx": joint_idx,
                "DOF\nIdx": dof_id,
            }
        )

    joint_data.sort(key=lambda x: x["Joint Name"])
    table_data = [[joint[header] for header in headers] for joint in joint_data]
    table = tabulate(table_data, headers=headers, tablefmt="grid", numalign="right", stralign="left")
    logger.debug("Joint Configuration:\n%s", table)
    xax_logger.log_file("joint_config_table.txt", table)


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


def remove_first_joint(file_path: str) -> str:
    """Remove the first joint found in the model."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    queue = deque([root])

    while queue:
        current_element = queue.popleft()

        # Skip defaults
        if current_element.tag == "default":
            continue

        # Check all children of the current element
        for child in current_element:
            if child.tag in {"joint", "freejoint"}:
                logger.info("Removing joint %s", child.get("name"))
                current_element.remove(child)
                return ET.tostring(root, encoding="utf-8").decode("utf-8")

            queue.append(child)

    logger.warning("No joint found in model. Returning original model.")
    return ET.tostring(root, encoding="utf-8").decode("utf-8")


def remove_mujoco_joints_except(file_path: str, joint_names: list[str]) -> str:
    """Remove all joints and references unless listed."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    def dfs_remove_joints(element: ET.Element) -> None:
        for child in list(element):  # Use list to avoid modifying while iterating
            # Skip defaults as they are needed for armatures, etc.
            if child.tag == "default":
                continue

            if child.tag in {"joint", "freejoint"} and child.get("name") not in joint_names:
                element.remove(child)
            else:
                dfs_remove_joints(child)

    def dfs_remove_references(element: ET.Element) -> None:
        for child in list(element):
            # Check if the child references a joint not in joint_names
            joint_attr = child.get("joint")
            if joint_attr and joint_attr not in joint_names:
                element.remove(child)
            # Keyframes are difficult to reorder, so we remove them for now.
            elif child.tag == "keyframe":
                element.remove(child)
            else:
                dfs_remove_references(child)

    dfs_remove_joints(root)
    dfs_remove_references(root)

    # Re-write to file.
    return ET.tostring(root, encoding="utf-8").decode("utf-8")


def add_new_mujoco_body(
    file_path: str,
    parent_body_name: str,
    new_body_name: str,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    add_visual: bool = True,
    visual_geom_color: tuple[float, float, float, float] = (1, 0, 0, 1),
    visual_geom_size: tuple[float, float, float] = (0.05, 0.05, 0.05),
) -> str:
    """Add a new body to the model."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    parent_body = None

    def dfs_find_body(element: ET.Element) -> ET.Element | None:
        for child in element:
            if child.tag == "body" and child.get("name") == parent_body_name:
                return child

            found = dfs_find_body(child)
            if found is not None:
                return found
        return None

    parent_body = dfs_find_body(root)
    if parent_body is None:
        raise ValueError(f"Parent body '{parent_body_name}' not found in model")

    # add the new body to the model
    new_body = ET.Element(
        "body",
        {
            "name": new_body_name,
            "pos": f"{pos[0]} {pos[1]} {pos[2]}",
            "quat": f"{quat[0]} {quat[1]} {quat[2]} {quat[3]}",
        },
    )

    if add_visual:
        visual_geom = ET.Element(
            "geom",
            {
                "name": f"{new_body_name}_visual",
                "type": "sphere",
                "class": "visual",
                "size": f"{visual_geom_size[0]} {visual_geom_size[1]} {visual_geom_size[2]}",
                "rgba": f"{visual_geom_color[0]} {visual_geom_color[1]} {visual_geom_color[2]} {visual_geom_color[3]}",
                "pos": "0 0 0",
                "quat": "1 0 0 0",
            },
        )
        new_body.append(visual_geom)

    parent_body.append(new_body)

    return ET.tostring(root, encoding="utf-8").decode("utf-8")
