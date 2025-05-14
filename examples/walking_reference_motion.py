"""Walking default humanoid task with reference gait tracking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

import attrs
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import xax

try:
    import bvhio
    import glm
    from bvhio.lib.hierarchy import Joint as BvhioJoint
except ImportError as e:
    raise ImportError(
        "In order to use reference motion utilities, please install Bvhio, using 'pip install bvhio'."
    ) from e


from jaxtyping import Array
from scipy.spatial.transform import Rotation as R

import ksim
from ksim.types import PhysicsModel
from ksim.utils.priors import (
    MotionReferenceMapping,
    generate_reference_motion,
    get_reference_joint_id,
    local_to_absolute,
    visualize_reference_motion,
    visualize_reference_points,
)

from .walking import HumanoidWalkingTask, HumanoidWalkingTaskConfig


@dataclass
class HumanoidWalkingReferenceMotionTaskConfig(HumanoidWalkingTaskConfig):
    bvh_path: str = xax.field(
        value=str(Path(__file__).parent / "data" / "walk-relaxed_actorcore.bvh"),
        help="The path to the BVH file.",
    )
    rotate_bvh_euler: tuple[float, float, float] = xax.field(
        value=(0, 0, 0),
        help="Optional rotation to ensure the BVH tree matches the Mujoco model.",
    )
    bvh_scaling_factor: float = xax.field(
        value=1.0,
        help="Scaling factor to ensure the BVH tree matches the Mujoco model.",
    )
    bvh_offset: tuple[float, float, float] = xax.field(
        value=(0.0, 0.0, 0.0),
        help="Offset to ensure the BVH tree matches the Mujoco model.",
    )
    mj_base_name: str = xax.field(
        value="pelvis",
        help="The Mujoco body name of the base of the humanoid",
    )
    reference_base_name: str = xax.field(
        value="CC_Base_Pelvis",
        help="The BVH joint name of the base of the humanoid",
    )
    visualize_reference_points: bool = xax.field(
        value=False,
        help="Whether to visualize the reference points.",
    )
    visualize_reference_motion: bool = xax.field(
        value=False,
        help="Whether to visualize the reference motion after running IK.",
    )


HUMANOID_REFERENCE_MAPPINGS = (
    MotionReferenceMapping("CC_Base_L_ThighTwist01", "thigh_left"),  # hip
    MotionReferenceMapping("CC_Base_L_CalfTwist01", "shin_left"),  # knee
    MotionReferenceMapping("CC_Base_L_Foot", "foot_left"),  # foot
    MotionReferenceMapping("CC_Base_L_UpperarmTwist01", "upper_arm_left"),  # shoulder
    MotionReferenceMapping("CC_Base_L_ForearmTwist01", "lower_arm_left"),  # elbow
    MotionReferenceMapping("CC_Base_L_Hand", "hand_left"),  # hand
    MotionReferenceMapping("CC_Base_R_ThighTwist01", "thigh_right"),  # hip
    MotionReferenceMapping("CC_Base_R_CalfTwist01", "shin_right"),  # knee
    MotionReferenceMapping("CC_Base_R_Foot", "foot_right"),  # foot
    MotionReferenceMapping("CC_Base_R_UpperarmTwist01", "upper_arm_right"),  # shoulder
    MotionReferenceMapping("CC_Base_R_ForearmTwist01", "lower_arm_right"),  # elbow
    MotionReferenceMapping("CC_Base_R_Hand", "hand_right"),  # hand
)


Config = TypeVar("Config", bound=HumanoidWalkingReferenceMotionTaskConfig)


def create_tracked_marker_update_fn(
    body_id: int, mj_base_id: int, tracked_pos_fn: Callable[[ksim.Trajectory], xax.FrozenDict[int, Array]]
) -> Callable[[ksim.Marker, ksim.Trajectory], None]:
    """Factory function to create a marker update for the tracked positions."""

    def _actual_update_fn(marker: ksim.Marker, transition: ksim.Trajectory) -> None:
        tracked_pos = tracked_pos_fn(transition)
        abs_pos = local_to_absolute(transition.xpos, tracked_pos[body_id], mj_base_id)
        marker.pos = tuple(abs_pos)

    return _actual_update_fn


def create_target_marker_update_fn(
    body_id: int, mj_base_id: int, target_pos_fn: Callable[[ksim.Trajectory], xax.FrozenDict[int, Array]]
) -> Callable[[ksim.Marker, ksim.Trajectory], None]:
    """Factory function to create a marker update for the target positions."""

    def _target_update_fn(marker: ksim.Marker, transition: ksim.Trajectory) -> None:
        target_pos = target_pos_fn(transition)
        abs_pos = local_to_absolute(transition.xpos, target_pos[body_id], mj_base_id)
        marker.pos = tuple(abs_pos)

    return _target_update_fn


@attrs.define(frozen=True, kw_only=True)
class QposReferenceMotionReward(ksim.Reward):
    reference_qpos: xax.HashableArray
    ctrl_dt: float
    norm: xax.NormType = attrs.field(default="l1")
    sensitivity: float = attrs.field(default=5.0)

    @property
    def num_frames(self) -> int:
        return self.reference_qpos.array.shape[0]

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        qpos = trajectory.qpos
        step_number = jnp.int32(jnp.round(trajectory.timestep / self.ctrl_dt)) % self.num_frames
        reference_qpos = jnp.take(self.reference_qpos.array, step_number, axis=0)
        error = xax.get_norm(reference_qpos - qpos, self.norm)
        mean_error = error.mean(axis=-1)
        reward = jnp.exp(-mean_error * self.sensitivity)
        return reward


class HumanoidWalkingReferenceMotionTask(HumanoidWalkingTask[Config], Generic[Config]):
    reference_qpos: Array

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards = [
            ksim.StayAliveReward(scale=1.0),
            QposReferenceMotionReward(
                reference_qpos=self.reference_motion.qpos,
                ctrl_dt=self.config.ctrl_dt,
                scale=0.5,
            ),
            # ksim.NaiveForwardReward(scale=1.0),
        ]

        return rewards

    def run(self) -> None:
        mj_model: PhysicsModel = self.get_mujoco_model()
        root: BvhioJoint = bvhio.readAsHierarchy(self.config.bvh_path)
        reference_base_id = get_reference_joint_id(root, self.config.reference_base_name)
        mj_base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.config.mj_base_name)

        def rotation_callback(root: BvhioJoint) -> None:
            euler_rotation = np.array(self.config.rotate_bvh_euler)
            quat = R.from_euler("xyz", euler_rotation).as_quat(scalar_first=True)
            root.applyRotation(glm.quat(*quat), bake=True)

        reference_motion = generate_reference_motion(
            model=mj_model,
            mj_base_id=mj_base_id,
            bvh_root=root,
            bvh_to_mujoco_names=HUMANOID_REFERENCE_MAPPINGS,
            bvh_base_id=reference_base_id,
            bvh_offset=np.array(self.config.bvh_offset),
            bvh_root_callback=rotation_callback,
            bvh_scaling_factor=self.config.bvh_scaling_factor,
            ctrl_dt=self.config.ctrl_dt,
            neutral_qpos=None,
            neutral_similarity_weight=0.1,
            temporal_consistency_weight=0.1,
            n_restarts=3,
            error_acceptance_threshold=1e-4,
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=2000,
            verbose=False,
        )
        self.reference_motion = reference_motion
        np_cartesian_motion = jax.tree.map(np.asarray, self.reference_motion.cartesian_poses)

        if self.config.visualize_reference_points:
            visualize_reference_points(
                model=mj_model,
                base_id=mj_base_id,
                reference_motion=np_cartesian_motion,
            )
        elif self.config.visualize_reference_motion:
            visualize_reference_motion(
                model=mj_model,
                reference_qpos=np.asarray(self.reference_motion.qpos),
                cartesian_motion=np_cartesian_motion,
                mj_base_id=mj_base_id,
            )
        else:
            super().run()


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.walking_reference_motion
    # To visualize the environment, use the following command:
    #   python -m examples.walking_reference_motion run_mode=view
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m examples.walking_reference_motion num_envs=2 batch_size=1
    HumanoidWalkingReferenceMotionTask.launch(
        HumanoidWalkingReferenceMotionTaskConfig(
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            iterations=8,
            ls_iterations=8,
            epochs_per_log_step=1,
            valid_every_n_steps=10,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            rollout_length_seconds=5.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            global_grad_clip=2.0,
            # Gait matching parameters.
            bvh_path=str(Path(__file__).parent / "data" / "walk_normal_dh.bvh"),
            rotate_bvh_euler=(0, np.pi / 2, 0),
            bvh_scaling_factor=1 / 100,
            mj_base_name="pelvis",
            reference_base_name="CC_Base_Pelvis",
        ),
    )
