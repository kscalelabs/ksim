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
from jaxtyping import Array, PRNGKeyArray
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

from .walking import (
    DefaultHumanoidModel,
    HumanoidWalkingTask,
    HumanoidWalkingTaskConfig,
)


class NaiveVelocityReward(ksim.Reward):
    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return trajectory.qvel[..., 0].clip(max=5.0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GaitMatchingAuxOutputs:
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
    # TODO: remove
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
class GaitMatchingReward(ksim.Reward):
    reference_gait: xax.FrozenDict[int, xax.HashableArray]
    ctrl_dt: float
    norm: xax.NormType = attrs.field(default="l1")
    sensitivity: float = attrs.field(default=5.0)

    @property
    def num_frames(self) -> int:
        return list(self.reference_gait.values())[0].array.shape[0]

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        assert isinstance(trajectory.aux_outputs, GaitMatchingAuxOutputs)
        reference_gait: xax.FrozenDict[int, Array] = jax.tree.map(lambda x: x.array, self.reference_gait)
        step_number = jnp.int32(jnp.round(trajectory.timestep / self.ctrl_dt)) % self.num_frames
        target_pos = jax.tree.map(lambda x: jnp.take(x, step_number, axis=0), reference_gait)
        tracked_pos = trajectory.aux_outputs.tracked_pos
        error = jax.tree.map(lambda target, tracked: xax.get_norm(target - tracked, self.norm), target_pos, tracked_pos)
        mean_error_over_bodies = jax.tree.reduce(jnp.add, error) / len(error)
        mean_error = mean_error_over_bodies.mean(axis=-1)
        reward = jnp.exp(-mean_error * self.sensitivity)
        return reward


class HumanoidWalkingGaitMatchingTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards = [
            ksim.BaseHeightRangeReward(z_lower=0.8, z_upper=1.5, dropoff=10.0, scale=0.5),
            ksim.LinearVelocityPenalty(index="z", scale=-0.01),
            ksim.AngularVelocityPenalty(index="x", scale=-0.01),
            ksim.AngularVelocityPenalty(index="y", scale=-0.01),
            NaiveVelocityReward(scale=0.1),
            GaitMatchingReward(reference_gait=self.reference_gait, ctrl_dt=self.config.ctrl_dt, scale=0.1),
        ]

        return rewards

    def sample_action(
        self,
        model: DefaultHumanoidModel,
        model_carry: None,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> ksim.Action:
        action_n = super().sample_action(model, model_carry, physics_model, physics_state, observations, commands, rng)

        # Getting the local cartesian positions for all tracked bodies.
        tracked_positions: dict[int, Array] = {}
        for body_id in self.tracked_body_ids:
            body_pos = get_local_xpos(physics_state.data.xpos, body_id, self.mj_base_id)
            tracked_positions[body_id] = jnp.array(body_pos)

        return ksim.Action(
            action=action_n.action,
            aux_outputs=GaitMatchingAuxOutputs(
                tracked_pos=xax.FrozenDict(tracked_positions),
            ),
        )

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
    #   python -m examples.walking_gait_matching
    # To visualize the environment, use the following command:
    #   python -m examples.walking_gait_matching run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m examples.walking_gait_matching num_envs=8 num_batches=2
    HumanoidWalkingGaitMatchingTask.launch(
        HumanoidWalkingGaitMatchingTaskConfig(
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            valid_every_n_steps=10,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=5.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            # Gait matching parameters.
            bvh_path=str(Path(__file__).parent / "data" / "walk-relaxed_actorcore.bvh"),
            rotate_bvh_euler=(0, np.pi / 2, 0),
            bvh_scaling_factor=1 / 100,
            mj_base_name="pelvis",
            reference_base_name="CC_Base_Pelvis",
        ),
    )
