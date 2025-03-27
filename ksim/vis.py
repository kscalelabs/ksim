"""Defines some visualization utility functions."""

__all__ = [
    "rotation_matrix_from_direction",
    "Marker",
]

from typing import Literal, Self

import attrs
import mujoco
import numpy as np

from ksim.utils.mujoco import get_body_pose_by_name, get_geom_pose_by_name, mat_to_quat, quat_to_mat

TargetType = Literal["body", "geom"]


def get_target_pose(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    target_name: str,
    target_type: TargetType,
) -> tuple[np.ndarray, np.ndarray]:
    if target_type == "body":
        target_pos, target_rot = get_body_pose_by_name(mj_model, mj_data, target_name)
    elif target_type == "geom":
        target_pos, target_rot = get_geom_pose_by_name(mj_model, mj_data, target_name)
    else:
        raise ValueError(f"Unsupported target type '{target_type}'.")

    if target_pos.shape != (3,):
        raise ValueError(f"Target position has shape {target_pos.shape}, expected (3,)")
    if target_rot.shape != (3, 3):
        raise ValueError(f"Target rotation has shape {target_rot.shape}, expected (3,3)")
    return target_pos, target_rot


def rotation_matrix_from_direction(
    direction: tuple[float, float, float],
    reference: tuple[float, float, float] = (0, 0, 1),
) -> np.ndarray:
    """Compute a rotation matrix that aligns the reference vector with the direction vector.

    Args:
        direction: The direction vector to align.
        reference: The reference vector to align with.

    Returns:
        A rotation matrix that aligns the reference vector with the direction vector.
    """
    # Normalize direction vector
    dir_vec = np.array(direction, dtype=float)
    norm = np.linalg.norm(dir_vec)
    if norm < 1e-10:  # Avoid division by zero
        return np.eye(3)

    dir_vec = dir_vec / norm

    # Normalize reference vector
    ref_vec = np.array(reference, dtype=float)
    ref_vec = ref_vec / np.linalg.norm(ref_vec)

    # Simple case: vectors are nearly aligned
    if np.abs(np.dot(dir_vec, ref_vec) - 1.0) < 1e-10:
        return np.eye(3)

    # Simple case: vectors are nearly opposite
    if np.abs(np.dot(dir_vec, ref_vec) + 1.0) < 1e-10:
        # Flip around x-axis for [0,0,1] reference
        if np.allclose(ref_vec, [0, 0, 1]):
            return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # General case
        else:
            # Find an axis perpendicular to the reference
            perp = np.cross(ref_vec, [1, 0, 0])
            if np.linalg.norm(perp) < 1e-10:
                perp = np.cross(ref_vec, [0, 1, 0])
            perp = perp / np.linalg.norm(perp)

            # Rotate 180 degrees around this perpendicular axis
            c = -1  # cos(π)
            s = 0  # sin(π)
            t = 1 - c
            x, y, z = perp

            return np.array(
                [
                    [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
                    [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
                    [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
                ]
            )

    # General case: use cross product to find rotation axis
    axis = np.cross(ref_vec, dir_vec)
    axis = axis / np.linalg.norm(axis)

    # Angle between vectors
    angle = np.arccos(np.clip(np.dot(ref_vec, dir_vec), -1.0, 1.0))

    # Rodrigues rotation formula
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis

    return np.array(
        [
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c],
        ]
    )


@attrs.define(frozen=True)
class Marker:
    # Geometry parameters.
    geom: mujoco.MjsGeom = attrs.field()
    scale: tuple[float, float, float] = attrs.field()
    pos: tuple[float, float, float] = attrs.field()
    orientation: tuple[float, float, float, float] = attrs.field()
    rgba: tuple[float, float, float, float] = attrs.field()
    label: str | None = attrs.field(default=None)

    # Tracking parameters.
    target_name: str | None = attrs.field(default=None)
    target_type: TargetType = attrs.field(default="body")
    track_x: bool = attrs.field(default=True)
    track_y: bool = attrs.field(default=True)
    track_z: bool = attrs.field(default=True)
    track_rotation: bool = attrs.field(default=True)

    def get_pos_and_rot(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = np.array(self.pos), np.array(self.orientation)
        rot = quat_to_mat(quat)

        if self.target_name is None:
            return pos, rot

        target_pos, target_rot = get_target_pose(mj_model, mj_data, self.target_name, self.target_type)

        if self.track_x:
            pos[0] += target_pos[0]
        if self.track_y:
            pos[1] += target_pos[1]
        if self.track_z:
            pos[2] += target_pos[2]
        if self.track_rotation:
            rot = target_rot @ rot

        return pos, rot

    def __call__(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, scene: mujoco.MjvScene) -> bool:
        if scene.ngeom >= scene.maxgeom:
            return False

        pos, rot = self.get_pos_and_rot(mj_model, mj_data)

        # Set basic properties
        scene.geoms[scene.ngeom].type = self.geom
        scene.geoms[scene.ngeom].size[:] = self.scale
        scene.geoms[scene.ngeom].pos[:] = pos
        scene.geoms[scene.ngeom].mat[:] = rot
        scene.geoms[scene.ngeom].rgba[:] = self.rgba

        # Handle label conversion if needed
        if self.label is not None:
            scene.geoms[scene.ngeom].label = str(self.label).encode("utf-8")

        # Set other rendering properties
        scene.geoms[scene.ngeom].dataid = -1
        scene.geoms[scene.ngeom].objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        scene.geoms[scene.ngeom].objid = -1
        scene.geoms[scene.ngeom].category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.geoms[scene.ngeom].emission = 0
        scene.geoms[scene.ngeom].specular = 0.5
        scene.geoms[scene.ngeom].shininess = 0.5

        # Increment the geom count
        scene.ngeom += 1

        return True

    @classmethod
    def arrow(
        cls,
        magnitude: float,
        pos: tuple[float, float, float],
        direction: tuple[float, float, float],
        rgba: tuple[float, float, float, float],
        label: str | None = None,
        size: float = 0.025,
    ) -> Self:
        quat = mat_to_quat(rotation_matrix_from_direction(direction))

        return cls(
            geom=mujoco.mjtGeom.mjGEOM_ARROW,
            scale=(size, size, magnitude * size),
            pos=pos,
            orientation=tuple(quat),
            rgba=rgba,
            label=label,
        )
