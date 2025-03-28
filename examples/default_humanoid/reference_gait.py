"""Generates and visualizes the reference gait."""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import bvhio
import glm
import mujoco
import numpy as np
from bvhio.lib.hierarchy import Joint as BvhioJoint
from scipy.spatial.transform import Rotation as R


@dataclass
class ReferenceMarker:
    """Maps a BVH joint to a Mujoco body in constant time."""

    reference_joint_id: int
    mj_body_id: int
    mj_body_offset: np.ndarray


def get_reference_joint_id(root: BvhioJoint, reference_joint_name: str) -> int:
    for joint, index, depth in root.layout():
        if joint.Name == reference_joint_name:
            return index
    raise ValueError(f"Joint {reference_joint_name} not found")


class ReferenceMarkerBuilder:
    """Builder for ReferenceMarker objects."""

    def __init__(self, reference_joint_name: str, mujoco_body_name: str, mujoco_body_offset: np.ndarray):
        self.reference_joint_name = reference_joint_name
        self.mujoco_body_name = mujoco_body_name
        self.mujoco_body_offset = mujoco_body_offset

    def build(self, model: mujoco.MjModel, root: BvhioJoint) -> ReferenceMarker:
        reference_joint_id = get_reference_joint_id(root, self.reference_joint_name)
        mujoco_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.mujoco_body_name)
        return ReferenceMarker(
            reference_joint_id=reference_joint_id,
            mj_body_id=mujoco_body_id,
            mj_body_offset=self.mujoco_body_offset,
        )


def add_sphere_to_scene(
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


def get_local_point_pos(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int, offset_from_body: np.ndarray, base_id: int
) -> np.ndarray:
    """Gets the relative position of a point w.r.t. the torso or base."""
    body_pos = data.xpos[body_id]
    body_rot = data.xmat[body_id].reshape(3, 3)
    return body_pos + body_rot @ offset_from_body - data.xpos[base_id]


def get_local_reference_pos(
    root: BvhioJoint, reference_id: int, reference_base_id: int, scaling_factor: float = 1.0
) -> np.ndarray:
    """Gets the relative position of a reference joint w.r.t. the torso or base."""
    layout = root.layout()
    joint = layout[reference_id][0]
    reference_joint = layout[reference_base_id][0]
    ref_pos = np.array(joint.PositionWorld) - np.array(reference_joint.PositionWorld)
    return ref_pos * scaling_factor


def local_to_xpos(model: mujoco.MjModel, data: mujoco.MjData, local_pos: np.ndarray, base_id: int) -> np.ndarray:
    """Converts a position relative to the torso or base to the absolute position."""
    return local_pos + data.xpos[base_id]


def overlay(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    mappings: list[ReferenceMarker],
    root: BvhioJoint,
    base_id: int,
    reference_base_id: int,
    root_callback: Callable[[BvhioJoint], None] | None,
) -> None:
    """Animates the model and adds real geoms to the scene for each joint position.

    Args:
        model: The Mujoco model
        data: The Mujoco data
        mappings: The mappings of BVH joints to Mujoco bodies
        root: The root of the BVH tree
        base_id: The ID of the Mujoco base
        reference_base_id: The ID of the reference base (of the BVH file)
    """
    total_frames = len(root.layout()[0][0].Keyframes)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            frame = frame % total_frames
            root.loadPose(frame)
            if root_callback:
                root_callback(root)

            scene = viewer.user_scn
            scene.ngeom = 0  # Clear previous geoms

            # Showing default humanoid geoms for reference
            for mapping in mappings:
                agent_local_pos = get_local_point_pos(model, data, mapping.mj_body_id, mapping.mj_body_offset, base_id)
                agent_xpos = local_to_xpos(model, data, agent_local_pos, base_id)
                add_sphere_to_scene(
                    scene,
                    pos=agent_xpos,
                    color=np.array([1, 0, 0, 1]),
                    scale=np.array([0.08, 0.08, 0.08]),
                )

                reference_local_pos = get_local_reference_pos(
                    root, mapping.reference_joint_id, reference_base_id, 1 / 100
                )
                reference_xpos = local_to_xpos(model, data, reference_local_pos, base_id)
                add_sphere_to_scene(
                    scene,
                    pos=reference_xpos,
                    color=np.array([0, 1, 0, 1]),
                    scale=np.array([0.08, 0.08, 0.08]),
                )

            viewer.sync()
            time.sleep(0.01)
            frame += 1


def generate_reference_gait(
    mappings: list[ReferenceMarker],
    root: BvhioJoint,
    reference_base_id: int,
    root_callback: Callable[[BvhioJoint], None] | None,
) -> np.ndarray:
    """Generates the reference gait for the given model and data.

    Args:
        mappings: The mappings of BVH joints to Mujoco bodies
        root: The root of the BVH tree
        reference_base_id: The ID of the reference base (of the BVH file)
        root_callback: A callback to modify the root of the BVH tree
    """
    total_frames = len(root.layout()[0][0].Keyframes)
    reference_gait = np.zeros((total_frames, len(mappings)))

    for frame in range(total_frames):
        root.loadPose(frame)

        if root_callback:
            root_callback(root)

        for mapping in mappings:
            breakpoint()
            reference_gait[frame, mapping.reference_joint_id] = get_local_reference_pos(
                root, mapping.reference_joint_id, reference_base_id
            )

    return reference_gait


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", action="store_true", default=False)
    parser.add_argument("--generate", action="store_true", default=False)
    args = parser.parse_args()

    local_path = Path(__file__).parent / "data"
    model = mujoco.MjModel.from_xml_path(str(local_path / "scene.mjcf"))
    data = mujoco.MjData(model)
    root = bvhio.readAsHierarchy(str(local_path / "walk-relaxed_actorcore.bvh"))

    # Mapping the most relevant joints to the Mujoco model.
    mapping_builders = [
        ReferenceMarkerBuilder("CC_Base_L_ThighTwist01", "thigh_left", np.array([0, 0, 0])),  # hip
        ReferenceMarkerBuilder("CC_Base_L_CalfTwist01", "shin_left", np.array([0, 0, 0])),  # knee
        ReferenceMarkerBuilder("CC_Base_L_Foot", "foot_left", np.array([0, 0, 0])),  # foot
        ReferenceMarkerBuilder("CC_Base_L_UpperarmTwist01", "upper_arm_left", np.array([0, 0, 0])),  # shoulder
        ReferenceMarkerBuilder("CC_Base_L_ForearmTwist01", "lower_arm_left", np.array([0, 0, 0])),  # elbow
        ReferenceMarkerBuilder("CC_Base_L_Hand", "hand_left", np.array([0, 0, 0])),  # hand
        ReferenceMarkerBuilder("CC_Base_R_ThighTwist01", "thigh_right", np.array([0, 0, 0])),  # hip
        ReferenceMarkerBuilder("CC_Base_R_CalfTwist01", "shin_right", np.array([0, 0, 0])),  # knee
        ReferenceMarkerBuilder("CC_Base_R_Foot", "foot_right", np.array([0, 0, 0])),  # foot
        ReferenceMarkerBuilder("CC_Base_R_UpperarmTwist01", "upper_arm_right", np.array([0, 0, 0])),  # shoulder
        ReferenceMarkerBuilder("CC_Base_R_ForearmTwist01", "lower_arm_right", np.array([0, 0, 0])),  # elbow
        ReferenceMarkerBuilder("CC_Base_R_Hand", "hand_right", np.array([0, 0, 0])),  # hand
    ]
    mappings = [builder.build(model, root) for builder in mapping_builders]

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    reference_base_id = get_reference_joint_id(root, "CC_Base_Pelvis")

    # Rotates the BVH tree 90 degrees to match the Mujoco model.
    def rotation_callback(root: BvhioJoint) -> None:
        euler_rotation = np.array([0, np.pi / 2, 0])
        quat = R.from_euler("xyz", euler_rotation).as_quat(scalar_first=True)
        root.applyRotation(glm.quat(*quat), bake=True)

    if args.overlay:
        overlay(
            model,
            data,
            mappings=mappings,
            root=root,
            base_id=base_id,
            reference_base_id=reference_base_id,
            root_callback=rotation_callback,
        )
    elif args.generate:
        save_path = local_path / "reference_gait.npz"
        print(f"Generating reference gait and saving to {save_path}")
        reference_gait = generate_reference_gait(
            mappings=mappings,
            root=root,
            reference_base_id=reference_base_id,
            root_callback=rotation_callback,
        )
        breakpoint()
        np.savez(save_path, reference_gait=reference_gait)
    else:
        raise ValueError("No action specified.")
