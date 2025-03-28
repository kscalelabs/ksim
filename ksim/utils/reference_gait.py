"""Reference gait utilities."""

import argparse
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import bvhio
import glm
import jax
import mujoco
import numpy as np
from bvhio.lib.hierarchy import Joint as BvhioJoint
from scipy.spatial.transform import Rotation as R
import xax


@dataclass
class ReferenceMapping:
    reference_joint_name: str
    mj_body_name: str

@dataclass
class ReferenceGait:
    body_id: int
    """The order in which the body appears in a MuJoCo-like xpos struct."""
    local_pos: np.ndarray
    """The position of the joint relative to the torso or base."""


def _add_reference_marker_to_scene(
    scene: mujoco.MjvScene,
    *,
    pos: np.ndarray,
    color: np.ndarray,
    scale: np.ndarray,
    label: str | None = None,
) -> None:
    """Adds a sphere to the scene.

    Args:
        scene: The MjvScene to add the sphere to
        pos: The position of the sphere
        color: The color of the sphere
        scale: The scale of the sphere
        label: The label of the sphere
    """
    if scene.ngeom >= scene.maxgeom:
        raise ValueError("Scene is full")

    g = scene.geoms[scene.ngeom]
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[:] = scale
    g.pos[:] = pos
    g.mat[:] = np.eye(3)
    g.rgba[:] = color
    g.label = label or ""
    g.dataid = -1
    g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid = -1
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    g.emission = 0
    g.specular = 0.5
    g.shininess = 0.5

    scene.ngeom += 1


def _get_local_xpos(xpos: np.ndarray | jax.Array, body_id: int, base_id: int) -> np.ndarray | jax.Array:
    """Gets the cartesian pos of a point w.r.t. the base (e.g. pelvis)."""
    return xpos[body_id] - xpos[base_id]


def _get_local_reference_pos(
    root: BvhioJoint, reference_id: int, reference_base_id: int, scaling_factor: float = 1.0
) -> np.ndarray | jax.Array:
    """Gets the cartesian pos of a reference joint w.r.t. the base (e.g. pelvis)."""
    layout = root.layout()
    joint = layout[reference_id][0]
    reference_joint = layout[reference_base_id][0]
    ref_pos = np.array(joint.PositionWorld) - np.array(reference_joint.PositionWorld)
    return ref_pos * scaling_factor


def _local_to_absolute(
    xpos: np.ndarray | jax.Array, local_pos: np.ndarray | jax.Array, base_id: int
) -> np.ndarray | jax.Array:
    """Gets the absolute xpos from a local position (for visualization)."""
    return local_pos + xpos[base_id]

def _get_reference_joint_id(root: BvhioJoint, reference_joint_name: str) -> int:
    for joint, index, depth in root.layout():
        if joint.Name == reference_joint_name:
            return index
    raise ValueError(f"Joint {reference_joint_name} not found")

def _get_body_id(model: mujoco.MjModel, mj_body_name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mj_body_name)

def get_reference_joint_ids(root: BvhioJoint, mappings: tuple[ReferenceMapping, ...]) -> tuple[int, ...]:
    return tuple([_get_reference_joint_id(root, mapping.reference_joint_name) for mapping in mappings])

def get_reference_body_ids(model: mujoco.MjModel, mappings: tuple[ReferenceMapping, ...]) -> tuple[int, ...]:
    return tuple([_get_body_id(model, mapping.mj_body_name) for mapping in mappings])

def visualize_reference_gait(
    model: mujoco.MjModel,
    *,
    base_id: int,
    reference_gait: xax.FrozenDict[int, np.ndarray],
) -> None:
    """Animates the model and adds real geoms to the scene for each joint position.

    Args:
        model: The Mujoco model
        base_id: The ID of the Mujoco base
        reference_gait: The reference gait (if root and reference_base_id are None)
    """

    total_frames = list(reference_gait.values())[0].shape[0]
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            frame = frame % total_frames

            scene = viewer.user_scn
            scene.ngeom = 0  # Clear previous geoms

            # Showing default humanoid geoms for reference
            for body_id, reference_poses in reference_gait.items():
                agent_local_pos = _get_local_xpos(data.xpos, body_id, base_id)
                agent_xpos = _local_to_absolute(data.xpos, agent_local_pos, base_id)
                assert isinstance(agent_xpos, np.ndarray)
                _add_reference_marker_to_scene(
                    scene,
                    pos=agent_xpos,
                    color=np.array([1, 0, 0, 1]),
                    scale=np.array([0.08, 0.08, 0.08]),
                )

                reference_local_pos = reference_poses[frame]
                reference_xpos = _local_to_absolute(data.xpos, reference_local_pos, base_id)
                assert isinstance(reference_xpos, np.ndarray)
                _add_reference_marker_to_scene(
                    scene,
                    pos=reference_xpos,
                    color=np.array([0, 1, 0, 1]),
                    scale=np.array([0.08, 0.08, 0.08]),
                )

            viewer.sync()
            time.sleep(0.01)
            frame += 1


def generate_reference_gait(
    mappings: tuple[ReferenceMapping, ...],
    root: BvhioJoint,
    reference_base_id: int,
    root_callback: Callable[[BvhioJoint], None] | None,
    scaling_factor: float = 1.0,
) -> xax.FrozenDict[int, np.ndarray]:
    """Generates the reference gait for the given model and data.
    
    Args:
        mappings: The mappings of BVH joints to Mujoco bodies
        root: The root of the BVH tree
        reference_base_id: The ID of the reference base (of the BVH file)
        root_callback: A callback to modify the root of the BVH tree
        scaling_factor: The scaling factor for the reference gait

    Returns:
        A tuple of tuples, where each tuple contains a Mujoco body ID and the target positions.
        The result will be of shape [T, 3].
    """
    reference_joint_ids = get_reference_joint_ids(root, mappings)
    reference_body_ids = get_reference_body_ids(model, mappings)

    total_frames = len(root.layout()[0][0].Keyframes)
    reference_gait = {body_id: np.zeros((total_frames, 3)) for body_id in reference_body_ids}
    reference_gait = xax.FrozenDict(reference_gait)

    for frame in range(total_frames):
        root.loadPose(frame)

        if root_callback:
            root_callback(root)

        for body_id in reference_body_ids:
            ref_pos = _get_local_reference_pos(root, reference_joint_ids[body_id], reference_base_id, scaling_factor)
            reference_gait[body_id][frame] = ref_pos

    return reference_gait

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", action="store_true", default=False)
    parser.add_argument("--generate", action="store_true", default=False)
    parser.add_argument("--reference_gait_path", type=str, default=None)
    parser.add_argument("--bvh_path", type=str, default=None)
    parser.add_argument("--scene_path", type=str, default=None)
    parser.add_argument("--scaling_factor", type=float, default=1 / 100)
    args = parser.parse_args()

    local_path = Path(__file__).parent / "data"
    scene_path = str(local_path / "scene.mjcf") if args.scene_path is None else args.scene_path
    bvh_path = str(local_path / "walk-relaxed_actorcore.bvh") if args.bvh_path is None else args.bvh_path

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    root = bvhio.readAsHierarchy(bvh_path)

    # Mapping the most relevant joints to the Mujoco model.
    mappings = [builder.build(model, root) for builder in HUMANOID_MAPPING_SPEC]
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    reference_base_id = get_reference_joint_id(root, "CC_Base_Pelvis")

    # Rotates the BVH tree 90 degrees to match the Mujoco model.
    def rotation_callback(root: BvhioJoint) -> None:
        euler_rotation = np.array([0, np.pi / 2, 0])
        quat = R.from_euler("xyz", euler_rotation).as_quat(scalar_first=True)
        root.applyRotation(glm.quat(*quat), bake=True)

    if args.overlay:
        if args.reference_gait_path is not None:
            with open(args.reference_gait_path, "rb") as f:
                reference_gait = pickle.load(f)
        else:
            reference_gait = generate_reference_gait(
                mappings=mappings,
                root=root,
                reference_base_id=reference_base_id,
                root_callback=rotation_callback,
                scaling_factor=args.scaling_factor,
            )

        overlay(
            model,
            data,
            mappings=mappings,
            base_id=base_id,
            reference_gait=reference_gait,
        )
    elif args.generate:
        if args.reference_gait_path is not None:
            save_path = args.reference_gait_path
        else:
            save_path = local_path / "reference_gait.pkl"
        print(f"Generating reference gait and saving to {save_path}")
        reference_gait = generate_reference_gait(
            mappings=mappings,
            root=root,
            reference_base_id=reference_base_id,
            root_callback=rotation_callback,
            scaling_factor=args.scaling_factor,
        )
        with open(save_path, "wb") as f:
            pickle.dump(reference_gait, f)
    else:
        raise ValueError("No action specified.")
