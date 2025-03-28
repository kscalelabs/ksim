"""Walking default humanoid task with reference gait tracking."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import attrs
import bvhio
import glm
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import xax
from bvhio.lib.hierarchy import Joint as BvhioJoint
from jaxtyping import Array
from scipy.spatial.transform import Rotation as R

import ksim
from ksim.types import PhysicsModel
from ksim.utils.reference_gait import (
    ReferenceMapping,
    generate_reference_gait,
    get_local_xpos,
    get_reference_joint_id,
    visualize_reference_gait,
)

from .walking import HumanoidWalkingTask, HumanoidWalkingTaskConfig, NaiveVelocityReward


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxTransitionOutputs:
    tracked_pos: xax.FrozenDict[int, Array]


@dataclass
class HumanoidWalkingGaitMatchingTaskConfig(HumanoidWalkingTaskConfig):
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
    mj_base_name: str = xax.field(
        value="pelvis",
        help="The Mujoco body name of the base of the humanoid",
    )
    reference_base_name: str = xax.field(
        value="CC_Base_Pelvis",
        help="The BVH joint name of the base of the humanoid",
    )
    visualize_reference_gait: bool = xax.field(
        value=False,
        help="Whether to visualize the reference gait.",
    )
    # REMOVE?
    device: str = xax.field(
        value="cpu",
        help="The device to run the task on.",
    )


HUMANOID_REFERENCE_MAPPINGS = (
    ReferenceMapping("CC_Base_L_ThighTwist01", "thigh_left"),  # hip
    ReferenceMapping("CC_Base_L_CalfTwist01", "shin_left"),  # knee
    ReferenceMapping("CC_Base_L_Foot", "foot_left"),  # foot
    ReferenceMapping("CC_Base_L_UpperarmTwist01", "upper_arm_left"),  # shoulder
    ReferenceMapping("CC_Base_L_ForearmTwist01", "lower_arm_left"),  # elbow
    ReferenceMapping("CC_Base_L_Hand", "hand_left"),  # hand
    ReferenceMapping("CC_Base_R_ThighTwist01", "thigh_right"),  # hip
    ReferenceMapping("CC_Base_R_CalfTwist01", "shin_right"),  # knee
    ReferenceMapping("CC_Base_R_Foot", "foot_right"),  # foot
    ReferenceMapping("CC_Base_R_UpperarmTwist01", "upper_arm_right"),  # shoulder
    ReferenceMapping("CC_Base_R_ForearmTwist01", "lower_arm_right"),  # elbow
    ReferenceMapping("CC_Base_R_Hand", "hand_right"),  # hand
)


Config = TypeVar("Config", bound=HumanoidWalkingGaitMatchingTaskConfig)


@attrs.define(frozen=True, kw_only=True)
class GaitMatchingPenalty(ksim.Reward):
    reference_gait: xax.FrozenDict[int, xax.HashableArray]

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        assert isinstance(trajectory.aux_transition_outputs, AuxTransitionOutputs)
        num_frames = list(self.reference_gait.values())[0].array.shape[0]
        reference_gait: xax.FrozenDict[int, Array] = jax.tree.map(lambda x: x.array, self.reference_gait)

        # Computes MSE error between the tracked and target positions per transition.
        def compute_error(num_steps: Array, transition: ksim.Trajectory) -> tuple[Array, Array]:
            assert isinstance(transition.aux_transition_outputs, AuxTransitionOutputs)
            frame_idx = num_steps % num_frames
            target_pos = jax.tree.map(lambda x: x[frame_idx], reference_gait)  # 3
            tracked_pos = transition.aux_transition_outputs.tracked_pos  # 3
            error = jax.tree.map(lambda target, tracked: jnp.mean((target - tracked) ** 2), target_pos, tracked_pos)
            mean_error = jnp.mean(jnp.array(list(error.values())))
            next_num_steps = jax.lax.select(transition.done, 0, num_steps + 1)

            return next_num_steps, mean_error

        _, errors = jax.lax.scan(compute_error, jnp.array(0), trajectory)
        jax.debug.breakpoint()
        return errors


class HumanoidWalkingGaitMatchingTask(HumanoidWalkingTask[Config], Generic[Config]):

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards = [
            ksim.BaseHeightRangeReward(z_lower=0.8, z_upper=1.5, scale=0.5),
            ksim.LinearVelocityZPenalty(scale=-0.01),
            ksim.AngularVelocityXYPenalty(scale=-0.01),
            NaiveVelocityReward(scale=0.1),
            GaitMatchingPenalty(reference_gait=self.reference_gait, scale=-0.5),
        ]

        return rewards

    def get_transition_aux_outputs(
        self,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        next_physics_state: ksim.PhysicsState,
        action: Array,
        terminated: Array,
    ) -> AuxTransitionOutputs:
        # Getting the local cartesian positions for all tracked bodies.
        tracked_positions: dict[int, Array] = {}
        for body_id in self.tracked_body_ids:
            body_pos = get_local_xpos(physics_state.data.xpos, body_id, self.mj_base_id)
            assert isinstance(body_pos, Array)
            tracked_positions[body_id] = body_pos

        return AuxTransitionOutputs(tracked_pos=xax.FrozenDict(tracked_positions))

    def run(self) -> None:
        mj_model: PhysicsModel = self.get_mujoco_model()
        root: BvhioJoint = bvhio.readAsHierarchy(self.config.bvh_path)
        reference_base_id = get_reference_joint_id(root, self.config.reference_base_name)
        self.mj_base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.config.mj_base_name)

        def rotation_callback(root: BvhioJoint) -> None:
            euler_rotation = np.array(self.config.rotate_bvh_euler)
            quat = R.from_euler("xyz", euler_rotation).as_quat(scalar_first=True)
            root.applyRotation(glm.quat(*quat), bake=True)

        np_reference_gait = generate_reference_gait(
            mappings=HUMANOID_REFERENCE_MAPPINGS,
            model=mj_model,
            root=root,
            reference_base_id=reference_base_id,
            root_callback=rotation_callback,
            scaling_factor=self.config.bvh_scaling_factor,
        )
        self.reference_gait: xax.FrozenDict[int, xax.HashableArray] = jax.tree.map(
            lambda x: xax.hashable_array(jnp.array(x)), np_reference_gait
        )
        self.tracked_body_ids = tuple(self.reference_gait.keys())

        if self.config.visualize_reference_gait:
            visualize_reference_gait(
                mj_model,
                base_id=self.mj_base_id,
                reference_gait=np_reference_gait,
            )
        else:
            super().run()


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.default_humanoid.walking
    # To visualize the environment, use the following command:
    #   python -m examples.default_humanoid.walking run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m examples.default_humanoid.walking num_envs=8 num_batches=2
    HumanoidWalkingGaitMatchingTask.launch(
        HumanoidWalkingGaitMatchingTaskConfig(
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=21.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            use_mit_actuators=True,
            valid_every_n_steps=50,
            # Gait matching parameters.
            bvh_path=str(Path(__file__).parent / "data" / "walk-relaxed_actorcore.bvh"),
            rotate_bvh_euler=(0, np.pi / 2, 0),
            bvh_scaling_factor=1 / 100,
            mj_base_name="pelvis",
            reference_base_name="CC_Base_Pelvis",
        ),
    )
