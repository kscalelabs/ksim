"""Reference motion utilities."""

import time
from dataclasses import dataclass
from typing import Callable

import jax
import mujoco
import numpy as np
import xax
from bvhio.lib.hierarchy import Joint as BvhioJoint
from scipy.optimize import least_squares

from ksim.viewer import GlfwMujocoViewer


@dataclass
class ReferenceMapping:
    reference_joint_name: str
    mj_body_name: str


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
    root: BvhioJoint, reference_id: int, reference_base_id: int, scaling_factor: float = 1.0
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
    for joint, index, depth in root.layout():
        if joint.Name == reference_joint_name:
            return index
    raise ValueError(f"Joint {reference_joint_name} not found")


def get_body_id(model: mujoco.MjModel, mj_body_name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mj_body_name)


def get_reference_joint_ids(root: BvhioJoint, mappings: tuple[ReferenceMapping, ...]) -> tuple[int, ...]:
    return tuple([get_reference_joint_id(root, mapping.reference_joint_name) for mapping in mappings])


def get_body_ids(model: mujoco.MjModel, mappings: tuple[ReferenceMapping, ...]) -> tuple[int, ...]:
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

        def marker_callback(model: mujoco.MjModel, data: mujoco.MjData, scn: mujoco.MjvScene) -> None:
            # Add markers for cartesian targets
            for body_id, target_poses in cartesian_motion.items():
                target_local_pos = target_poses[frame]
                target_xpos = local_to_absolute(data.xpos, target_local_pos, mj_base_id)
                assert isinstance(target_xpos, np.ndarray)
                _add_reference_marker_to_scene(
                    scene,
                    pos=target_xpos,
                    color=np.array([0, 1, 0, 0.8]),
                    scale=np.array([0.05, 0.05, 0.05]),
                    label=f"target_{model.body(body_id).name}",
                )

        # Advance simulation time
        data.time += model.opt.timestep

        viewer.render(callback=marker_callback)


def get_reference_cartesian_poses(
    mappings: tuple[ReferenceMapping, ...],
    model: mujoco.MjModel,
    root: BvhioJoint,
    reference_base_id: int,
    root_callback: Callable[[BvhioJoint], None] | None,
    scaling_factor: float = 1.0,
    offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
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

        for body_id, reference_joint_id in zip(body_ids, reference_joint_ids):
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


def get_reference_qpos(
    model: mujoco.MjModel,
    mj_base_id: int,
    bvh_root: BvhioJoint,
    bvh_to_mujoco_names: tuple[ReferenceMapping, ...],
    bvh_base_id: int,
    bvh_offset: np.ndarray = np.array([0.0, 0.0, 0.0]),
    bvh_root_callback: Callable[[BvhioJoint], None] | None = None,
    bvh_scaling_factor: float = 1.0,
    constrained_joint_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),  # maybe need 6???
    neutral_qpos: np.ndarray | None = None,
    neutral_similarity_weight: float = 0.1,
    temporal_consistency_weight: float = 0.1,
    n_restarts: int = 3,
    error_acceptance_threshold: float = 1e-4,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    max_nfev: int = 2000,
    verbose: bool = False,
) -> np.ndarray:
    """Generates the reference motion for the given model and data.

    Args:
        model: The Mujoco model
        mj_base_id: The ID of the Mujoco base
        bvh_root: The root of the BVH tree
        bvh_to_mujoco_names: The mappings of BVH joints to Mujoco bodies
        bvh_base_id: The ID of the reference base (of the BVH file)
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
        Numpy array of shape (time, qpos)
    """
    cartesian_motion = get_reference_cartesian_poses(
        mappings=bvh_to_mujoco_names,
        model=model,
        root=bvh_root,
        reference_base_id=bvh_base_id,
        root_callback=bvh_root_callback,
        scaling_factor=bvh_scaling_factor,
        offset=bvh_offset,
    )
    total_frames = list(cartesian_motion.values())[0].shape[0]
    body_ids = list(cartesian_motion.keys())
    data = mujoco.MjData(model)
    constrained_jnt_mask = np.zeros(data.qpos.shape, dtype=bool)
    constrained_jnt_mask[np.array(constrained_joint_ids)] = True

    if neutral_qpos is None:
        neutral_qpos = np.copy(data.qpos)
    previous_qpos = np.copy(neutral_qpos)

    qpos_reference_motion = []
    for frame in range(total_frames):
        bvh_root.loadPose(frame)
        if bvh_root_callback:
            bvh_root_callback(bvh_root)

        cartesian_pose = {body_id: cartesian_motion[body_id][frame] for body_id in body_ids}
        qpos = solve_multi_body_ik(
            data=data,
            model=model,
            mj_base_id=mj_base_id,
            constrained_jnt_mask=constrained_jnt_mask,
            cartesian_pose=cartesian_pose,
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
        previous_qpos = qpos

    return np.array(qpos_reference_motion)
