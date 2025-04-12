"""Example walking task using Adversarial Motion Priors."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Literal, TypeVar

import attrs
import chex
import glm
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
import xax
from jaxtyping import PRNGKeyArray
import equinox as eqx

try:
    import bvhio
    from bvhio.lib.hierarchy import Joint as BvhioJoint
except ImportError as e:
    raise ImportError(
        "In order to use reference motion utilities, please install Bvhio, using 'pip install bvhio'."
    ) from e


from jaxtyping import Array, PyTree
from scipy.spatial.transform import Rotation as R

import ksim
from ksim.types import PhysicsModel
from ksim.utils.reference_motion import (
    ReferenceMapping,
    get_reference_cartesian_poses,
    get_reference_joint_id,
    get_reference_qpos,
    local_to_absolute,
    visualize_reference_motion,
    visualize_reference_points,
)

from .walking import (
    DefaultHumanoidActor,
    DefaultHumanoidCritic,
    HumanoidWalkingTask,
    HumanoidWalkingTaskConfig,
    NaiveForwardReward,
    NUM_JOINTS,
)


class DefaultHumanoidDiscriminator(eqx.Module):
    """Discriminator for the walking task, returns a logit"""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
        num_frames: int,
    ) -> None:
        num_inputs = NUM_JOINTS * num_frames
        num_outputs = 1

        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=num_outputs,
            width_size=hidden_size,
            depth=depth,
            key=key,
            activation=jax.nn.relu,
        )

    def forward(self, obs_n: Array) -> Array:
        return self.mlp(obs_n)


class DefaultHumanoidAMPModel(eqx.Module):
    actor: DefaultHumanoidActor
    critic: DefaultHumanoidCritic
    discriminator: DefaultHumanoidDiscriminator

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
        num_mixtures: int,
        num_frames: int,
    ) -> None:
        self.actor = DefaultHumanoidActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=0.5,
            hidden_size=hidden_size,
            depth=depth,
            num_mixtures=num_mixtures,
        )
        self.critic = DefaultHumanoidCritic(
            key,
            hidden_size=hidden_size,
            depth=depth,
        )
        self.discriminator = DefaultHumanoidDiscriminator(
            key,
            hidden_size=hidden_size,
            depth=depth,
            num_frames=num_frames,
        )


@dataclass
class HumanoidWalkingAMPTaskConfig(HumanoidWalkingTaskConfig):
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
    discriminator_num_frames: int = xax.field(
        value=10,
        help="The number of frames to use for the discriminator.",
    )
    discriminator_imputation_strategy: Literal["zero", "repeat"] = xax.field(
        value="repeat",
        help="The imputation strategy to use for the discriminator.",
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


Config = TypeVar("Config", bound=HumanoidWalkingAMPTaskConfig)


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


# @attrs.define(frozen=True, kw_only=True)
# class AMPRewardCarry:
#     qpos_history: Array
#     did_prev_reset: Array


@jax.tree_util.register_pytree_node_class
@dataclass
class AMPAuxOutputs:
    discriminator_values: Array


@jax.tree_util.register_pytree_node_class
@dataclass
class AMPModelCarry:
    qpos_history: Array
    did_prev_reset: Array


@attrs.define(frozen=True, kw_only=True)
class AMPReward(ksim.Reward):

    # def get_discriminator_input_sequence(
    #     self, trajectory: ksim.Trajectory, initial_carry: AMPRewardCarry
    # ) -> tuple[Array, AMPRewardCarry]:
    #     chex.assert_shape(initial_carry.qpos_history, (self.discriminator_seq_len - 1, trajectory.qpos.shape[1]))

    #     def scan_fn(carry: AMPRewardCarry, transition: ksim.Trajectory) -> tuple[AMPRewardCarry, Array]:
    #         qpos_history = carry.qpos_history

    #         # Handling resets with imputation strategy
    #         if self.imputation_strategy == "zero":
    #             reset_history = jnp.zeros_like(qpos_history)
    #         elif self.imputation_strategy == "repeat":
    #             reset_history = jnp.tile(transition.qpos, self.discriminator_seq_len - 1)
    #         else:
    #             raise ValueError(f"Invalid imputation strategy: {self.imputation_strategy}")
    #         qpos_history = jnp.where(carry.did_prev_reset, reset_history, qpos_history)

    #         discriminator_input = jnp.concatenate([qpos_history, transition.qpos[None]], axis=0)
    #         next_history = discriminator_input[1:]  # input len - 1
    #         return AMPRewardCarry(qpos_history=next_history, did_prev_reset=transition.done), discriminator_input

    #     final_carry, discriminator_input = jax.lax.scan(scan_fn, initial_carry, trajectory)
    #     return discriminator_input, final_carry

    # def intial_carry(self, rng: PRNGKeyArray) -> AMPRewardCarry:
    #     # Initializing history with zeros since it will be overwritten anyways
    #     qpos_history = jnp.zeros((self.discriminator_seq_len - 1, self.reference_qpos.array.shape[1]))
    #     return AMPRewardCarry(qpos_history=qpos_history, did_prev_reset=jnp.array(True))

    def __call__(
        self,
        trajectory: ksim.Trajectory,
        carry: None,
    ) -> tuple[Array, None]:
        # All discriminator values should be between 0 and 1
        discriminator_values = trajectory.aux_outputs.discriminator_values
        reward = 1 - discriminator_values
        return reward, None


class HumanoidWalkingAMPTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards = [
            ksim.StayAliveReward(scale=1.0),
            NaiveForwardReward(scale=0.1, clip_max=2.0),
            AMPReward(scale=0.1),
        ]

        return rewards

    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidAMPModel:
        return DefaultHumanoidAMPModel(
            key,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_mixtures=self.config.num_mixtures,
            num_frames=self.config.discriminator_num_frames,
        )

    def sample_action(
        self,
        model: DefaultHumanoidAMPModel,
        model_carry: None,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> ksim.Action:
        action_dist_j = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
        )
        action_j = action_dist_j.sample(seed=rng)

        # TODO: add the discriminator values here...

        return ksim.Action(action=action_j, carry=None, aux_outputs=None)

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> AMPModelCarry:
        return AMPModelCarry(
            qpos_history=jnp.zeros((self.config.discriminator_num_frames - 1, self.reference_qpos.shape[1])),
            did_prev_reset=jnp.array(True),
        )

    def get_discriminator_inputs(self, trajectory: ksim.Trajectory) -> Array:
        chex.assert_shape(initial_carry.qpos_history, (self.discriminator_seq_len - 1, trajectory.qpos.shape[1]))

        def scan_fn(carry: AMPModelCarry, transition: ksim.Trajectory) -> tuple[AMPModelCarry, Array]:
            qpos_history = carry.qpos_history

            # Handling resets with imputation strategy
            if self.imputation_strategy == "zero":
                reset_history = jnp.zeros_like(qpos_history)
            elif self.imputation_strategy == "repeat":
                reset_history = jnp.tile(transition.qpos, self.discriminator_seq_len - 1)
            else:
                raise ValueError(f"Invalid imputation strategy: {self.imputation_strategy}")
            qpos_history = jnp.where(carry.did_prev_reset, reset_history, qpos_history)

            discriminator_input = jnp.concatenate([qpos_history, transition.qpos[None]], axis=0)
            next_history = discriminator_input[1:]  # input len - 1
            return AMPRewardCarry(qpos_history=next_history, did_prev_reset=transition.done), discriminator_input

        final_carry, discriminator_input = jax.lax.scan(scan_fn, initial_carry, trajectory)
        return discriminator_input, final_carry

    @abstractmethod
    def update_model(
        self,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectories: ksim.Trajectory,
        rewards: ksim.Rewards,
        rollout_env_states: ksim.RolloutEnvState,
        rollout_shared_state: ksim.RolloutSharedState,
        rollout_constants: ksim.RolloutConstants,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, PyTree, xax.FrozenDict[str, Array], ksim.LoggedTrajectory]:
        model_arr, opt_state, next_model_carry, train_metrics, logged_traj = super().update_model(
            optimizer,
            opt_state,
            trajectories,
            rewards,
            rollout_env_states,
            rollout_shared_state,
            rollout_constants,
            rng,
        )

        discriminator_inputs = self.get_discriminator_inputs(trajectories)

        # TODO: reformat this into some scannable loop with minibatching and epochs...
        model = eqx.combine(model_arr, rollout_constants.model_static)
        assert isinstance(model, DefaultHumanoidAMPModel)
        discriminator_values = ...

        return model_arr, opt_state, next_model_carry, train_metrics, logged_traj

    def run(self) -> None:
        mj_model: PhysicsModel = self.get_mujoco_model()
        root: BvhioJoint = bvhio.readAsHierarchy(self.config.bvh_path)
        reference_base_id = get_reference_joint_id(root, self.config.reference_base_name)
        self.mj_base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.config.mj_base_name)

        def rotation_callback(root: BvhioJoint) -> None:
            euler_rotation = np.array(self.config.rotate_bvh_euler)
            quat = R.from_euler("xyz", euler_rotation).as_quat(scalar_first=True)
            root.applyRotation(glm.quat(*quat), bake=True)

        np_reference_motion = get_reference_cartesian_poses(
            mappings=HUMANOID_REFERENCE_MAPPINGS,
            model=mj_model,
            root=root,
            reference_base_id=reference_base_id,
            root_callback=rotation_callback,
            scaling_factor=self.config.bvh_scaling_factor,
            offset=np.array(self.config.bvh_offset),
        )
        np_reference_qpos = get_reference_qpos(
            model=mj_model,
            mj_base_id=self.mj_base_id,
            bvh_root=root,
            bvh_to_mujoco_names=HUMANOID_REFERENCE_MAPPINGS,
            bvh_base_id=reference_base_id,
            bvh_offset=np.array(self.config.bvh_offset),
            bvh_root_callback=rotation_callback,
            bvh_scaling_factor=self.config.bvh_scaling_factor,
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
        self.reference_qpos = jnp.array(np_reference_qpos)

        if self.config.visualize_reference_points:
            visualize_reference_points(
                model=mj_model,
                base_id=self.mj_base_id,
                reference_motion=np_reference_motion,
            )
        elif self.config.visualize_reference_motion:
            visualize_reference_motion(
                model=mj_model,
                reference_qpos=np_reference_qpos,
            )
        else:
            super().run()


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.walking_reference_motion
    # To visualize the environment, use the following command:
    #   python -m examples.walking_reference_motion run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m examples.walking_reference_motion num_envs=8 num_batches=2
    HumanoidWalkingAMPTask.launch(
        HumanoidWalkingAMPTaskConfig(
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            valid_every_n_steps=10,
            # Simulation parameters.
            dt=0.002,
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
            bvh_path=str(Path(__file__).parent / "data" / "walk_normal_dh.bvh"),
            rotate_bvh_euler=(0, np.pi / 2, 0),
            bvh_scaling_factor=1 / 100,
            mj_base_name="pelvis",
            reference_base_name="CC_Base_Pelvis",
        ),
    )
