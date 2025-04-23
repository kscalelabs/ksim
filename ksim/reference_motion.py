"""Motion priors utilities."""

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import xax
from bvhio.lib.hierarchy import Joint as BvhioJoint
from jaxtyping import Array

from ksim.utils.motion_prior_utils import (
    get_body_id,
    get_local_reference_pos,
    get_reference_joint_id,
    solve_multi_body_ik,
)


@dataclass
class ReferenceMapping:
    reference_joint_name: str
    mj_body_name: str


@dataclass(frozen=True)
class ReferenceMotionData:
    """Stores reference motion data (qpos and Cartesian poses)."""

    qpos: xax.HashableArray  # Shape: [T, nq]
    qvel: xax.HashableArray  # Shape: [T, nq]
    cartesian_poses: xax.FrozenDict[int, xax.HashableArray]  # Dict: body_id -> [T, 3]
    ctrl_dt: float

    qroot: Optional[xax.HashableArray] = None  # Shape: [T, 3]
    qroot_vel: Optional[xax.HashableArray] = None  # Shape: [T, 3]

    @property
    def num_frames(self) -> int:
        """Returns the total number of frames in the reference motion."""
        return self.qpos.array.shape[0]

    def get_qroot_at_step(self, step: int | Array) -> Array:
        """Gets the reference qroot at a specific step (or steps)."""
        frame_index = step % self.num_frames
        return jnp.take(self.qroot.array, frame_index, axis=0)

    def get_qroot_vel_at_step(self, step: int | Array) -> Array:
        """Gets the reference qroot_vel at a specific step (or steps)."""
        frame_index = step % self.num_frames
        return jnp.take(self.qroot_vel.array, frame_index, axis=0)

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


def get_reference_joint_ids(root: BvhioJoint, mappings: tuple[ReferenceMapping, ...]) -> tuple[int, ...]:
    return tuple([get_reference_joint_id(root, mapping.reference_joint_name) for mapping in mappings])


def get_body_ids(model: mujoco.MjModel, mappings: tuple[ReferenceMapping, ...]) -> tuple[int, ...]:
    return tuple([get_body_id(model, mapping.mj_body_name) for mapping in mappings])


def compute_qvel(qpos: Array, ctrl_dt: float, freejoint: bool = False) -> Array:
    """Compute finite-difference qvel from qpos."""
    qvel_reference_motion = []
    total_frames = qpos.shape[0]

    for frame in range(total_frames - 1):
        if freejoint:
            free_joint_pos = qpos[:, :3]
            free_joint_quat = qpos[:, 3:7]

            free_joint_quat_scalar_last = free_joint_quat[..., [1, 2, 3, 0]]
            free_joint_rotvec = xax.rotvec_from_quat(free_joint_quat_scalar_last)

            free_joint_vel = (free_joint_pos[1:] - free_joint_pos[:-1]) / ctrl_dt
            free_joint_vel_rot = (free_joint_rotvec[1:] - free_joint_rotvec[:-1]) / ctrl_dt

            free_joint_pos = free_joint_pos[1:-1]
            free_joint_quat = free_joint_quat[1:-1]

            joint_pos = qpos[:, 7:]
            joints_vel = (joint_pos[1:] - joint_pos[:-1]) / ctrl_dt
            joint_pos = joint_pos[1:-1]

            free_joint_qvel = np.concatenate([free_joint_vel, free_joint_vel_rot], axis=1)
            qvel_reference_motion.append(np.concatenate([free_joint_qvel, joints_vel], axis=1))
        else:
            qvel = (qpos[frame + 1] - qpos[frame]) / ctrl_dt
            qvel_reference_motion.append(qvel)
    return qvel_reference_motion


def generate_reference_motion(
    model: mujoco.MjModel,
    mj_base_id: int,
    bvh_root: BvhioJoint,
    bvh_to_mujoco_names: tuple[ReferenceMapping, ...],
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
) -> ReferenceMotionData:
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
    qvel_reference_motion = compute_qvel(qpos_reference_motion, ctrl_dt, freejoint=False)

    jnp_reference_qpos = jnp.array(qpos_reference_motion)
    jnp_cartesian_motion = jax.tree.map(lambda arr: xax.HashableArray(arr), np_cartesian_motion)
    jnp_qvel = jnp.array(qvel_reference_motion)

    # 3. Create and return the data object
    return ReferenceMotionData(
        qpos=xax.HashableArray(jnp_reference_qpos),
        qvel=xax.HashableArray(jnp_qvel),
        cartesian_poses=jnp_cartesian_motion,
        ctrl_dt=ctrl_dt,
    )


def get_reference_cartesian_poses(
    mappings: tuple[ReferenceMapping, ...],
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
