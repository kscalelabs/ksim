"""Defines some visualization utility functions."""

__all__ = [
    "rotation_matrix_from_direction",
    "Marker",
]

from typing import Callable, Literal, Self

import attrs
import mujoco
import numpy as np

from ksim.types import Trajectory
from ksim.utils.mujoco import (
    get_body_pose,
    get_body_pose_by_name,
    get_geom_pose_by_name,
    mat_to_quat,
    quat_to_mat,
)

TargetType = Literal["body", "geom", "root"]
UpdateFn = Callable[["Marker", Trajectory], None]


def get_target_pose(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    target_name: str | None,
    target_type: TargetType,
) -> tuple[np.ndarray, np.ndarray]:
    match target_type:
        case "body":
            if target_name is None:
                raise ValueError("Target name must be provided for body target type.")
            target_pos, target_rot = get_body_pose_by_name(mj_model, mj_data, target_name)
        case "geom":
            if target_name is None:
                raise ValueError("Target name must be provided for geom target type.")
            target_pos, target_rot = get_geom_pose_by_name(mj_model, mj_data, target_name)
        case "root":
            target_pos, target_rot = get_body_pose(mj_data, 1)
        case _:
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


@attrs.define(kw_only=True)
class Marker:
    # Geometry parameters.
    geom: mujoco.MjsGeom = attrs.field()
    scale: tuple[float, float, float] = attrs.field(default=(0.0, 0.0, 0.0))
    pos: tuple[float, float, float] = attrs.field(default=(0.0, 0.0, 0.0))
    orientation: tuple[float, float, float, float] = attrs.field(default=(0.0, 0.0, 0.0, 1.0))
    rgba: tuple[float, float, float, float] = attrs.field(default=(1.0, 1.0, 1.0, 1.0))
    label: str | None = attrs.field(default=None)

    # Tracking parameters.
    target_name: str | None = attrs.field(default=None)
    target_type: TargetType = attrs.field(default="body")
    track_x: bool = attrs.field(default=True)
    track_y: bool = attrs.field(default=True)
    track_z: bool = attrs.field(default=True)
    track_rotation: bool = attrs.field(default=False)

    # Data parameters.
    geom_idx: int | None = attrs.field(default=None)
    update_fn: UpdateFn | None = attrs.field(default=None)

    def get_pos_and_rot(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = np.array(self.pos), np.array(self.orientation)
        rot = quat_to_mat(quat)

        if self.target_type != "root" and self.target_name is None:
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

    def _initialize_scene(self, scene: mujoco.MjvScene) -> int:
        if scene.ngeom >= scene.maxgeom:
            raise ValueError("Max number of geoms reached.")

        geom_idx = scene.ngeom
        g = scene.geoms[geom_idx]

        # Set other rendering properties
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5

        # Increment the geom count
        scene.ngeom += 1

        return geom_idx

    def _update_scene(self, mj_model: mujoco.MjModel, mj_data: mujoco.MjData, scene: mujoco.MjvScene) -> None:
        if self.geom_idx is None:
            self.geom_idx = self._initialize_scene(scene)

        pos, rot = self.get_pos_and_rot(mj_model, mj_data)
        g = scene.geoms[self.geom_idx]

        # Set basic properties
        g.type = self.geom
        g.size[:] = np.array(self.scale, dtype=np.float32)
        g.pos[:] = np.array(pos, dtype=np.float32)
        g.mat[:] = np.array(rot, dtype=np.float32)
        g.rgba[:] = np.array(self.rgba, dtype=np.float32)

        # Handle label conversion if needed
        g.label = ("" if self.label is None else self.label).encode("utf-8")

        scene.ngeom = max(scene.ngeom, self.geom_idx + 1)

    def update(self, transition: Trajectory) -> None:
        if self.update_fn is not None:
            self.update_fn(self, transition)

    def __call__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        scene: mujoco.MjvScene,
        transition: Trajectory,
    ) -> None:
        self.update(transition)
        self._update_scene(mj_model, mj_data, scene)

    @classmethod
    def quat_from_direction(cls, direction: tuple[float, float, float]) -> tuple[float, float, float, float]:
        return tuple(mat_to_quat(rotation_matrix_from_direction(direction)))

    @classmethod
    def arrow(
        cls,
        magnitude: float,
        pos: tuple[float, float, float],
        direction: tuple[float, float, float],
        rgba: tuple[float, float, float, float],
        label: str | None = None,
        size: float = 0.025,
        target_name: str | None = None,
        target_type: TargetType = "body",
        update_fn: UpdateFn | None = None,
    ) -> Self:
        return cls(
            geom=mujoco.mjtGeom.mjGEOM_ARROW,
            scale=(size, size, magnitude * size),
            pos=pos,
            orientation=cls.quat_from_direction(direction),
            rgba=rgba,
            label=label,
            target_name=target_name,
            target_type=target_type,
            update_fn=update_fn,
        )

    @classmethod
    def sphere(
        cls,
        pos: tuple[float, float, float],
        radius: float,
        rgba: tuple[float, float, float, float],
        label: str | None = None,
        target_name: str | None = None,
        target_type: TargetType = "body",
        update_fn: UpdateFn | None = None,
    ) -> Self:
        return cls(
            geom=mujoco.mjtGeom.mjGEOM_SPHERE,
            scale=(radius, radius, radius),
            pos=pos,
            orientation=cls.quat_from_direction((1.0, 0.0, 0.0)),
            rgba=rgba,
            label=label,
            target_name=target_name,
            target_type=target_type,
            update_fn=update_fn,
        )


def configure_scene(
    scene: mujoco.MjvScene,
    vopt: mujoco.MjvOption,
    shadow: bool = False,
    reflection: bool = False,
    contact_force: bool = False,
    contact_point: bool = False,
    inertia: bool = False,
) -> mujoco.MjvScene:
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow
    scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = reflection
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = contact_force
    vopt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = contact_point
    vopt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = inertia
    return scene
