"""Reference motion utilities."""

__all__ = [
    "MotionReferenceMapping",
    "MotionReferenceData",
    "get_local_xpos",
    "get_local_reference_pos",
    "local_to_absolute",
    "get_reference_joint_id",
    "get_body_id",
    "get_reference_joint_ids",
    "get_body_ids",
    "visualize_reference_points",
    "visualize_reference_motion",
    "get_reference_cartesian_poses",
    "generate_reference_motion",
]

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Self, Sequence, cast

import bvhio
import colorlogging
import glm
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import xax
from bvhio.lib.hierarchy import Joint as BvhioJoint
from jaxtyping import Array
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from ksim.viewer import GlfwMujocoViewer

logger = logging.getLogger(__name__)


@dataclass
class MotionReferenceMapping:
    reference_joint_name: str
    mj_body_name: str


@dataclass(frozen=True)
class MotionReferenceData:
    """Stores reference motion data (qpos and Cartesian poses)."""

    qpos: xax.HashableArray  # Shape: [T, nq]
    qvel: xax.HashableArray  # Shape: [T, nq]
    cartesian_poses: xax.FrozenDict[int, xax.HashableArray]  # Dict: body_id -> [T, 3]
    ctrl_dt: float

    @property
    def num_frames(self) -> int:
        """Returns the total number of frames in the reference motion."""
        return self.qpos.array.shape[0]

    def get_qpos_at_step(self, step: int | Array) -> Array:
        """Gets the reference qpos at a specific step (or steps)."""
        frame_index = step % self.num_frames
        return jnp.take(self.qpos.array, frame_index, axis=0)

    def get_qvel_at_step(self, step: int | Array) -> Array:
        """Gets the reference qvel at a specific step (or steps)."""
        frame_index = step % self.num_frames
        return jnp.take(self.qvel.array, frame_index, axis=0)

    def get_cartesian_pose_at_step(self, step: int | Array) -> xax.FrozenDict[int, Array]:
        """Gets the reference Cartesian pose at a specific step (or steps)."""
        frame_index = step % self.num_frames
        return jax.tree.map(
            lambda hashable_arr: jnp.take(hashable_arr.array, frame_index, axis=0), self.cartesian_poses
        )

    def get_qpos_at_time(self, time: float | Array) -> Array:
        """Gets the reference qpos closest to a specific time."""
        step = jnp.int32(jnp.round(time / self.ctrl_dt))
        return self.get_qpos_at_step(step)

    def get_cartesian_pose_at_time(self, time: float | Array) -> xax.FrozenDict[int, Array]:
        """Gets the reference Cartesian pose closest to a specific time."""
        step = jnp.int32(jnp.round(time / self.ctrl_dt))
        return self.get_cartesian_pose_at_step(step)

    def save(self, path: str | Path) -> None:
        """Saves the MotionReferenceData to a .npz file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        body_ids = np.array(list(self.cartesian_poses.keys()))
        pose_arrays = [pose.array for pose in self.cartesian_poses.values()]

        pose_arrays_np = [np.asarray(arr) for arr in pose_arrays]
        cartesian_poses_stacked = np.stack(pose_arrays_np, axis=0)  # Shape [num_bodies, T, 3]

        np.savez(
            save_path,
            qpos=np.asarray(self.qpos.array),
            qvel=np.asarray(self.qvel.array),
            cartesian_pose_body_ids=body_ids,
            cartesian_poses_stacked=cartesian_poses_stacked,
            ctrl_dt=np.array(self.ctrl_dt),
        )
        logger.info("Saved reference motion data to %s", save_path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Loads MotionReferenceData from a .npz file."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Reference motion file not found: {load_path}")

        logger.info("Loading reference motion data from %s", load_path)
        data = np.load(load_path)

        qpos = xax.HashableArray(jnp.array(data["qpos"]))
        qvel = xax.HashableArray(jnp.array(data["qvel"]))
        ctrl_dt = float(data["ctrl_dt"].item())

        # Reconstruct cartesian poses dictionary
        body_ids = data["cartesian_pose_body_ids"]
        cartesian_poses_stacked = data["cartesian_poses_stacked"]
        cartesian_poses: xax.FrozenDict[int, xax.HashableArray] = xax.FrozenDict(
            {
                int(body_id): xax.HashableArray(jnp.array(cartesian_poses_stacked[i]))
                for i, body_id in enumerate(body_ids)
            }
        )

        return cls(
            qpos=qpos,
            qvel=qvel,
            cartesian_poses=cartesian_poses,
            ctrl_dt=ctrl_dt,
        )


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


def get_local_xpos(xpos: np.ndarray | jax.Array, body_id: int, base_id: int) -> np.ndarray | jax.Array:
    """Gets the cartesian pos of a point w.r.t. the base (e.g. pelvis)."""
    return xpos[body_id] - xpos[base_id]


def get_local_reference_pos(
    root: BvhioJoint,
    reference_id: int,
    reference_base_id: int,
    scaling_factor: float = 1.0,
) -> np.ndarray | jax.Array:
    """Gets the cartesian pos of a reference joint w.r.t. the base (e.g. pelvis)."""
    layout = root.layout()
    joint = layout[reference_id][0]
    reference_joint = layout[reference_base_id][0]
    ref_pos = np.array(joint.PositionWorld) - np.array(reference_joint.PositionWorld)
    return ref_pos * scaling_factor


def local_to_absolute(
    xpos: np.ndarray | jax.Array, local_pos: np.ndarray | jax.Array, base_id: int
) -> np.ndarray | jax.Array:
    """Gets the absolute xpos from a local position (for visualization)."""
    return local_pos + xpos[base_id]


def get_reference_joint_id(root: BvhioJoint, reference_joint_name: str) -> int:
    for joint, index, _ in root.layout():
        if joint.Name == reference_joint_name:
            return index
    raise ValueError(f"Joint {reference_joint_name} not found")


def get_body_id(model: mujoco.MjModel, mj_body_name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mj_body_name)


def get_reference_joint_ids(root: BvhioJoint, mappings: tuple[MotionReferenceMapping, ...]) -> tuple[int, ...]:
    return tuple([get_reference_joint_id(root, mapping.reference_joint_name) for mapping in mappings])


def get_body_ids(model: mujoco.MjModel, mappings: tuple[MotionReferenceMapping, ...]) -> tuple[int, ...]:
    return tuple([get_body_id(model, mapping.mj_body_name) for mapping in mappings])


def visualize_reference_points(
    model: mujoco.MjModel,
    base_id: int,
    reference_motion: xax.FrozenDict[int, np.ndarray],
) -> None:
    """Animates the model and adds real geoms to the scene for each joint position.

    Args:
        model: The Mujoco model
        base_id: The ID of the Mujoco base
        reference_motion: The reference motion (if root and reference_base_id are None)
        display_names: Whether to display the names of the bodies / reference joints
    """
    total_frames = list(reference_motion.values())[0].shape[0]
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            frame = frame % total_frames

            scene = viewer.user_scn
            scene.ngeom = 0  # Clear previous geoms

            # Showing default humanoid geoms for reference
            for body_id, reference_poses in reference_motion.items():
                agent_local_pos = get_local_xpos(data.xpos, body_id, base_id)
                agent_xpos = local_to_absolute(data.xpos, agent_local_pos, base_id)
                assert isinstance(agent_xpos, np.ndarray)
                _add_reference_marker_to_scene(
                    scene,
                    pos=agent_xpos,
                    color=np.array([1, 0, 0, 1]),
                    scale=np.array([0.08, 0.08, 0.08]),
                )

                reference_local_pos = reference_poses[frame]
                reference_xpos = local_to_absolute(data.xpos, reference_local_pos, base_id)
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


def visualize_reference_motion(
    model: mujoco.MjModel,
    reference_qpos: np.ndarray,
    cartesian_motion: xax.FrozenDict[int, np.ndarray],
    mj_base_id: int,
) -> None:
    data = mujoco.MjData(model)
    viewer = GlfwMujocoViewer(model, data, mode="window", width=1024, height=768)

    # Set some nice camera parameters
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 45.0
    viewer.cam.elevation = -20.0

    # Keep the window open until you close it
    while viewer.is_alive:
        # Calculate which frame we should be on
        frame = int(data.time / model.opt.timestep) % len(reference_qpos)

        # Update the pose from reference qpos
        data.qpos = reference_qpos[frame]
        mujoco.mj_forward(model, data)

        # Clear previous geoms
        scene = viewer.scn
        scene.ngeom = 0

        def marker_callback(
            model: mujoco.MjModel,
            data: mujoco.MjData,
            scn: mujoco.MjvScene,
            frame_id: int = frame,
        ) -> None:
            # Add markers for cartesian targets
            for body_id, target_poses in cartesian_motion.items():
                target_local_pos = target_poses[frame_id]
                target_xpos = local_to_absolute(data.xpos, target_local_pos, mj_base_id)
                assert isinstance(target_xpos, np.ndarray)
                _add_reference_marker_to_scene(
                    scn,
                    pos=target_xpos,
                    color=np.array([0, 1, 0, 0.8]),
                    scale=np.array([0.05, 0.05, 0.05]),
                    label=f"target_{model.body(body_id).name}",
                )

        # Advance simulation time
        data.time += model.opt.timestep

        viewer.render(callback=marker_callback)


def get_reference_cartesian_poses(
    mappings: tuple[MotionReferenceMapping, ...],
    model: mujoco.MjModel,
    root: BvhioJoint,
    reference_base_id: int,
    root_callback: Callable[[BvhioJoint], None] | None,
    scaling_factor: float = 1.0,
    offset: np.ndarray | None = None,
) -> xax.FrozenDict[int, np.ndarray]:
    """Generates the reference motion for the given model and data.

    Args:
        mappings: The mappings of BVH joints to Mujoco bodies
        model: The Mujoco model
        root: The root of the BVH tree
        reference_base_id: The ID of the reference base (of the BVH file)
        root_callback: A callback to modify the root of the BVH tree
        scaling_factor: The scaling factor for the reference motion
        offset: The offset of the reference motion

    Returns:
        A tuple of tuples, where each tuple contains a Mujoco body ID and the target positions.
        The result will be of shape [T, 3].
    """
    if offset is None:
        offset = np.array([0.0, 0.0, 0.0])

    reference_joint_ids = get_reference_joint_ids(root, mappings)
    body_ids = get_body_ids(model, mappings)

    total_frames = len(root.layout()[0][0].Keyframes)
    reference_motion: xax.FrozenDict[int, np.ndarray] = xax.FrozenDict(
        {body_id: np.zeros((total_frames, 3)) for body_id in body_ids}
    )

    for frame in range(total_frames):
        root.loadPose(frame)

        if root_callback:
            root_callback(root)

        for body_id, reference_joint_id in zip(body_ids, reference_joint_ids, strict=False):
            ref_pos = get_local_reference_pos(root, reference_joint_id, reference_base_id, scaling_factor)
            reference_motion[body_id][frame] = ref_pos + offset

    return reference_motion


def multi_body_ik_error(
    x: np.ndarray,
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mj_base_id: int,
    constrained_jnt_mask: np.ndarray,
    cartesian_pose: xax.FrozenDict[int, np.ndarray],
    neutral_qpos: np.ndarray,
    prev_qpos: np.ndarray,
    neutral_similarity_weight: float = 0.1,
    temporal_consistency_weight: float = 0.1,
) -> np.ndarray:
    """Computes the error of the multi-body IK.

    Args:
        x: Candidate qpos (without constrained joints)
        model: The Mujoco model
        data: The Mujoco data
        mj_base_id: The ID of the Mujoco base (e.g. pelvis)
        constrained_jnt_mask: In order of the qpos array, 1 = fix to neutral pos
        cartesian_pose: The Cartesian position with respect to the base
        neutral_qpos: The neutral qpos
        prev_qpos: The previous qpos
        neutral_similarity_weight: The weight of the neutral similarity term
        temporal_consistency_weight: The weight of the temporal consistency term

    Returns:
        An array of individual error terms.
    """
    new_qpos = np.copy(neutral_qpos)
    new_qpos[~constrained_jnt_mask] = x
    data.qpos = new_qpos
    mujoco.mj_forward(model, data)  # Needed to get xpos

    errors = []
    for body_id, target_pos in cartesian_pose.items():
        # The target pos is w.r.t. the base, so we need to get the local pos
        current_pos = get_local_xpos(data.xpos, body_id, mj_base_id)
        error = target_pos - current_pos
        errors.append(error)

    if neutral_qpos is not None:
        neutral_similarity_error = (data.qpos - neutral_qpos) * neutral_similarity_weight
        errors.append(neutral_similarity_error)

    if prev_qpos is not None:
        temporal_consistency_error = (data.qpos - prev_qpos) * temporal_consistency_weight
        errors.append(temporal_consistency_error)

    return np.concatenate(errors)


def solve_multi_body_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mj_base_id: int,
    constrained_jnt_mask: np.ndarray,
    cartesian_pose: dict[int, np.ndarray],
    neutral_qpos: np.ndarray,
    prev_qpos: np.ndarray,
    neutral_similarity_weight: float = 0.1,
    temporal_consistency_weight: float = 0.1,
    n_restarts: int = 3,
    error_acceptance_threshold: float = 1e-4,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    max_nfev: int = 2000,
    verbose: bool = False,
) -> np.ndarray:
    """Solves the IK for the multi-body system.

    Args:
        model: The Mujoco model
        data: The Mujoco data
        mj_base_id: The ID of the Mujoco base
        constrained_jnt_mask: In order of the qpos array, 1 = fix to neutral pos
        cartesian_pose: The Cartesian position with respect to the base
        neutral_qpos: The neutral qpos
        prev_qpos: The previous qpos
        neutral_similarity_weight: The weight of the neutral similarity term
        temporal_consistency_weight: The weight of the temporal consistency term
        n_restarts: The number of random restarts to try
        error_acceptance_threshold: The threshold for the error
        ftol: The tolerance for the function value
        xtol: The tolerance for the solution
        max_nfev: The maximum number of function evaluations
        verbose: Whether to print verbose output

    Returns:
        The complete qpos array after solving inverse kinematics.
    """
    best_result = None
    best_cost = float("inf")

    for _ in range(n_restarts):
        qpos = data.qpos[~constrained_jnt_mask]
        initial_guess = qpos + np.random.normal(0, 0.1, size=qpos.shape)

        result = least_squares(
            multi_body_ik_error,
            initial_guess,
            kwargs={
                "model": model,
                "data": data,
                "mj_base_id": mj_base_id,
                "constrained_jnt_mask": constrained_jnt_mask,
                "cartesian_pose": cartesian_pose,
                "neutral_qpos": neutral_qpos,
                "prev_qpos": prev_qpos,
                "neutral_similarity_weight": neutral_similarity_weight,
                "temporal_consistency_weight": temporal_consistency_weight,
            },
            verbose=2 if verbose else 0,
            ftol=ftol,
            xtol=xtol,
            max_nfev=max_nfev,
        )

        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result

        if best_cost < error_acceptance_threshold:
            break

    if best_result is None:
        raise RuntimeError("All IK attempts failed")

    qpos = np.copy(neutral_qpos)
    qpos[~constrained_jnt_mask] = best_result.x
    return qpos


def generate_reference_motion(
    model: mujoco.MjModel,
    mj_base_id: int,
    bvh_root: BvhioJoint,
    bvh_to_mujoco_names: tuple[MotionReferenceMapping, ...],
    bvh_base_id: int,
    ctrl_dt: float,
    bvh_offset: np.ndarray | None = None,
    bvh_root_callback: Callable[[BvhioJoint], None] | None = None,
    bvh_scaling_factor: float = 1.0,
    constrained_joint_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
    neutral_qpos: np.ndarray | None = None,
    neutral_similarity_weight: float = 0.1,
    temporal_consistency_weight: float = 0.1,
    n_restarts: int = 3,
    error_acceptance_threshold: float = 1e-4,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    max_nfev: int = 2000,
    verbose: bool = False,
) -> MotionReferenceData:
    """Generates reference qpos and cartesian poses from BVH data.

    Args:
        model: The Mujoco model
        mj_base_id: The ID of the Mujoco base
        bvh_root: The root of the BVH tree
        bvh_to_mujoco_names: The mappings of BVH joints to Mujoco bodies
        bvh_base_id: The ID of the reference base (of the BVH file)
        ctrl_dt: The control timestep, used for time-based lookups.
        bvh_offset: Helps line up root with mj base
        bvh_root_callback: Modifies the root of the BVH tree (e.g. rotation)
        bvh_scaling_factor: The scaling factor for the reference motion
        constrained_joint_ids: The indices of the joints to constrain
        neutral_qpos: Helps with optimization, by default the starting qpos
        neutral_similarity_weight: Weight of neutral similarity term
        temporal_consistency_weight: Weight of temporal consistency term
        n_restarts: Number of random restarts to try
        error_acceptance_threshold: The threshold for the error
        ftol: The tolerance for the function value
        xtol: The tolerance for the solution
        max_nfev: The maximum number of function evaluations
        verbose: Whether to print verbose output

    Returns:
        A ReferenceMotionData object containing qpos and Cartesian poses.
    """
    if bvh_offset is None:
        bvh_offset = np.array([0.0, 0.0, 0.0])

    # 1. Generate Cartesian Poses
    np_cartesian_motion = get_reference_cartesian_poses(
        mappings=bvh_to_mujoco_names,
        model=model,
        root=bvh_root,
        reference_base_id=bvh_base_id,
        root_callback=bvh_root_callback,
        scaling_factor=bvh_scaling_factor,
        offset=bvh_offset,
    )
    total_frames = list(np_cartesian_motion.values())[0].shape[0]
    body_ids = list(np_cartesian_motion.keys())

    # 2. Solve IK for Qpos
    data = mujoco.MjData(model)
    constrained_jnt_mask = np.zeros(data.qpos.shape, dtype=bool)
    constrained_jnt_mask[np.array(constrained_joint_ids)] = True

    if neutral_qpos is None:
        neutral_qpos = np.copy(data.qpos)
    previous_qpos = np.copy(neutral_qpos)

    qpos_reference_motion = []
    for frame in range(total_frames):
        # Reload pose for consistency if callback modifies it in place
        bvh_root.loadPose(frame)
        if bvh_root_callback:
            bvh_root_callback(bvh_root)

        cartesian_pose_at_frame = {body_id: np_cartesian_motion[body_id][frame] for body_id in body_ids}
        qpos = solve_multi_body_ik(
            data=data,
            model=model,
            mj_base_id=mj_base_id,
            constrained_jnt_mask=constrained_jnt_mask,
            cartesian_pose=cartesian_pose_at_frame,
            neutral_qpos=neutral_qpos,
            prev_qpos=previous_qpos,
            neutral_similarity_weight=neutral_similarity_weight,
            temporal_consistency_weight=temporal_consistency_weight,
            n_restarts=n_restarts,
            error_acceptance_threshold=error_acceptance_threshold,
            ftol=ftol,
            xtol=xtol,
            max_nfev=max_nfev,
            verbose=verbose,
        )
        qpos_reference_motion.append(qpos)
        previous_qpos = qpos  # Update previous qpos for the next frame

    # 3. Compute qvel
    qvel_reference_motion = []
    for frame in range(total_frames - 1):
        qvel = (qpos_reference_motion[frame + 1] - qpos_reference_motion[frame]) / ctrl_dt
        qvel_reference_motion.append(qvel)

    jnp_reference_qpos = jnp.array(qpos_reference_motion)
    jnp_cartesian_motion = jax.tree.map(lambda arr: xax.HashableArray(arr), np_cartesian_motion)
    jnp_qvel = jnp.array(qvel_reference_motion)

    # 3. Create and return the data object
    return MotionReferenceData(
        qpos=xax.HashableArray(jnp_reference_qpos),
        qvel=xax.HashableArray(jnp_qvel),
        cartesian_poses=jnp_cartesian_motion,
        ctrl_dt=ctrl_dt,
    )


def vis_entry_point() -> None:
    parser = argparse.ArgumentParser(description="Visualize reference motion from a config file")
    parser.add_argument("motion", type=str, help="Path to the configuration file")
    parser.add_argument("--model", type=str, required=True, help="Path to the Mujoco model")
    parser.add_argument("--base_name", type=str, default="pelvis", help="Name of the Mujoco base")
    args = parser.parse_args()

    reference_data = MotionReferenceData.load(args.motion)
    model = mujoco.MjModel.from_xml_path(args.model)
    mj_base_id = get_body_id(model, args.base_name)

    np_cartesian_motion = jax.tree.map(lambda x: np.asarray(x.array), reference_data.cartesian_poses)

    visualize_reference_motion(
        model=model,
        reference_qpos=np.asarray(reference_data.qpos.array),
        cartesian_motion=np_cartesian_motion,
        mj_base_id=mj_base_id,
    )


@dataclass
class IKParams:
    neutral_similarity_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight of neutral similarity term",
        },
    )
    temporal_consistency_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight of temporal consistency term",
        },
    )
    n_restarts: int = field(
        default=5,
        metadata={
            "help": "Number of random restarts to try",
        },
    )
    error_acceptance_threshold: float = field(
        default=1e-4,
        metadata={
            "help": "The threshold for the error",
        },
    )
    ftol: float = field(
        default=1e-8,
        metadata={
            "help": "The tolerance for the function value",
        },
    )
    xtol: float = field(
        default=1e-8,
        metadata={
            "help": "The tolerance for the solution",
        },
    )
    max_nfev: int = field(
        default=2000,
        metadata={
            "help": "The maximum number of function evaluations",
        },
    )


@dataclass
class ReferenceMotionGeneratorConfig:
    model_path: str = field(
        default=MISSING,
        metadata={
            "help": "Path to the Mujoco model",
        },
    )
    bvh_path: str = field(
        default=MISSING,
        metadata={
            "help": "Path to the BVH file",
        },
    )
    output_path: str = field(
        default=MISSING,
        metadata={
            "help": "Path to the output file",
        },
    )
    mj_base_name: str = field(
        default=MISSING,
        metadata={
            "help": "Name of the Mujoco base",
        },
    )
    bvh_base_name: str = field(
        default=MISSING,
        metadata={
            "help": "Name of the BVH base",
        },
    )
    mappings: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of mappings in 'BvhJointName:MjBodyName' format",
        },
    )
    ctrl_dt: float = field(
        default=0.02,
        metadata={
            "help": "The control timestep",
        },
    )
    bvh_offset: list[float] = field(
        default_factory=lambda: [0, 0, 0],
        metadata={
            "help": "Offset of the BVH root",
        },
    )
    rotate_bvh_euler: list[float] = field(
        default_factory=lambda: [0, 0, 0],
        metadata={
            "help": "Euler angles to rotate the BVH root",
        },
    )
    bvh_scaling_factor: float = field(
        default=1.0,
        metadata={
            "help": "Scaling factor for the reference motion",
        },
    )
    constrained_joint_ids: list[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6],
        metadata={
            "help": "Indices of the constrained joints",
        },
    )
    ik_params: IKParams = field(
        default_factory=IKParams,
        metadata={
            "help": "IK parameters",
        },
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Whether to print verbose output"},
    )

    @classmethod
    def from_cli_args(cls, args: Sequence[str] | None = None) -> Self:
        """Parses configuration from command line arguments and optional config file."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("-f", "--config", type=Path, default=None, help="Path to the configuration file.")
        parsed_args, remaining_args = parser.parse_known_args(args)
        config_path: Path | None = parsed_args.config

        cfg = cast(Self, OmegaConf.structured(cls))

        if config_path is not None:
            if config_path.is_file():
                logger.info("Loading configuration from: %s", config_path)
                try:
                    file_conf = OmegaConf.load(config_path)
                    cfg = cast(Self, OmegaConf.merge(cfg, file_conf))
                except Exception as e:
                    logger.error("Error loading config file '%s': %s", config_path, e)
                    sys.exit(1)
            else:
                logger.warning("Config file specified but not found: %s", config_path)

        try:
            cli_conf = OmegaConf.from_cli(remaining_args)
            cfg = cast(Self, OmegaConf.merge(cfg, cli_conf))
        except ConfigKeyError as e:
            logger.error(
                "Invalid command line config override: %s. Check if the key exists in the configuration structure.", e
            )
            sys.exit(1)
        except Exception as e:
            logger.error("Error parsing command line arguments: %s", e)
            sys.exit(1)

        try:
            OmegaConf.resolve(cast(DictConfig, cfg))
            assert isinstance(cfg.mappings, ListConfig), "Config error: 'mappings' must be a list of strings."
            mapping_msg = "Config error: All 'mappings' must be strings in 'BvhJointName:MjBodyName' format."
            assert all(isinstance(m, str) and ":" in m for m in cfg.mappings), mapping_msg
            bvh_offset_msg = "Config error: 'bvh_offset' must be a list of 3 floats."
            assert isinstance(cfg.bvh_offset, ListConfig) and len(cfg.bvh_offset) == 3, bvh_offset_msg
            rotate_bvh_euler_msg = "Config error: 'rotate_bvh_euler' must be a list of 3 floats."
            assert isinstance(cfg.rotate_bvh_euler, ListConfig) and len(cfg.rotate_bvh_euler) == 3, rotate_bvh_euler_msg
            constrained_joint_ids_msg = "Config error: 'constrained_joint_ids' must be a list of integers."
            assert isinstance(cfg.constrained_joint_ids, ListConfig), constrained_joint_ids_msg

        except MissingMandatoryValue as e:
            logger.error("Missing required configuration value: %s", e)
            logger.error("Please provide it via command line (e.g., model_path=...) or in a config file.")
            sys.exit(1)
        except AssertionError as e:
            logger.error(e)
            sys.exit(1)
        except Exception as e:
            logger.error("Configuration Error: %s", e)
            sys.exit(1)

        return cfg


def main() -> None:
    colorlogging.configure(level=logging.INFO)

    cfg = ReferenceMotionGeneratorConfig.from_cli_args(sys.argv[1:])

    logger.info("Final configuration:\n%s", OmegaConf.to_yaml(cfg))

    try:
        parsed_mappings_list = []
        for mapping_str in cfg.mappings:
            parts = mapping_str.split(":")
            if len(parts) != 2:
                raise ValueError("Invalid mapping format: '%s'. Use 'BvhJointName:MjBodyName'.", mapping_str)
            parsed_mappings_list.append(MotionReferenceMapping(reference_joint_name=parts[0], mj_body_name=parts[1]))
        parsed_mappings: tuple[MotionReferenceMapping, ...] = tuple(parsed_mappings_list)
    except ValueError as e:
        logger.error("Configuration Error: %s", e)
        sys.exit(1)

    logger.info("Loading MuJoCo model from %s", cfg.model_path)
    model = mujoco.MjModel.from_xml_path(cfg.model_path)

    logger.info("Loading BVH file from %s", cfg.bvh_path)
    bvh_root = bvhio.readAsHierarchy(cfg.bvh_path)

    constrained_joint_ids = tuple(cfg.constrained_joint_ids)

    mj_base_id = get_body_id(model, cfg.mj_base_name)
    bvh_base_id = get_reference_joint_id(bvh_root, cfg.bvh_base_name)

    logger.info("Generating reference motion...")
    output_dir = Path(cfg.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    def rotation_callback(root: BvhioJoint) -> None:
        euler_rotation = np.array(cfg.rotate_bvh_euler)
        quat = R.from_euler("xyz", euler_rotation).as_quat(scalar_first=True)
        root.applyRotation(glm.quat(*quat), bake=True)

    reference_data = generate_reference_motion(
        model=model,
        mj_base_id=mj_base_id,
        bvh_root=bvh_root,
        bvh_to_mujoco_names=parsed_mappings,
        bvh_base_id=bvh_base_id,
        ctrl_dt=cfg.ctrl_dt,
        bvh_offset=np.array(cfg.bvh_offset),
        bvh_root_callback=rotation_callback,
        bvh_scaling_factor=cfg.bvh_scaling_factor,
        constrained_joint_ids=constrained_joint_ids,
        neutral_qpos=None,
        neutral_similarity_weight=cfg.ik_params.neutral_similarity_weight,
        temporal_consistency_weight=cfg.ik_params.temporal_consistency_weight,
        n_restarts=cfg.ik_params.n_restarts,
        error_acceptance_threshold=cfg.ik_params.error_acceptance_threshold,
        ftol=cfg.ik_params.ftol,
        xtol=cfg.ik_params.xtol,
        max_nfev=cfg.ik_params.max_nfev,
        verbose=cfg.verbose,
    )

    logger.info("Saving reference motion data to %s", cfg.output_path)
    reference_data.save(cfg.output_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
