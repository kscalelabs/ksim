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


@jax.tree_util.register_pytree_node_class
@dataclass
class Motion:
    """A fixed-length window of motion history, including current frame.

    This is used as a carry term when rolling out such that historical frames
    are available to the discriminator.

    Naming convention by leaf dimension:
    - `motion_frame`: (num_joints,)
    - `motion`: (num_frames, num_joints)
    - `motions`: (num batches or rollout length, num_frames, num_joints)
    """

    qpos_frames: Array
    # TODO: experiment with adding xpos...


@jax.tree_util.register_pytree_node_class
@dataclass
class AMPAuxOutputs:
    """Auxiliary outputs generated during on-policy rollout.

    This lives inside the trajectory object directly because the on-policy
    discriminator outputs get incorporated via the standard reward API.
    """

    discriminator_logits: Array
    motion: Motion


class DefaultHumanoidDiscriminator(eqx.Module):
    """Discriminator for the walking task, returns logit."""

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

    def forward(self, motion: Motion) -> Array:
        return self.mlp(motion.qpos_frames)


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


@attrs.define(frozen=True, kw_only=True)
class AMPReward(ksim.Reward):

    def __call__(
        self,
        trajectory: ksim.Trajectory,
        _: None,
    ) -> tuple[Array, None]:
        assert isinstance(trajectory.aux_outputs, AMPAuxOutputs)
        discriminator_values = jax.nn.sigmoid(trajectory.aux_outputs.discriminator_logits)
        reward = 1 - discriminator_values
        return reward, None


class HumanoidWalkingAMPTask(HumanoidWalkingTask[Config], Generic[Config]):
    """Humanoid walking task with adversarial motion prior.

    NOTE: this is WIP until it trains efficiently. Should be moved into the core
    library once it's stable. The user will eventually decide what core RL
    algorithm to subclass, and this class simply adds a discriminator.
    """

    reference_motions: Motion

    def get_optimizer(self) -> optax.GradientTransformation:
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            (
                optax.adam(self.config.learning_rate)
                if self.config.adam_weight_decay == 0.0
                else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
            ),
        )
        # TODO: explore different optimizer for discriminator (use mask fn)

        return optimizer

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
        model_carry: Motion,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        action_dist_j = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)

        motion = jax.tree.map(lambda x: x[1:], model_carry)
        motion = Motion(jnp.concatenate([motion.qpos_frames, physics_state.data.qpos[None]], axis=0))
        discriminator_logits = model.discriminator.forward(motion)

        amp_aux_outputs = AMPAuxOutputs(
            discriminator_logits=discriminator_logits,
            motion=motion,
        )

        return ksim.Action(action=action_j, carry=motion, aux_outputs=amp_aux_outputs)

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> Motion:
        return self._get_initial_motion_carry()

    def _get_initial_motion_carry(self) -> Motion:
        return Motion(jnp.zeros((self.config.discriminator_num_frames, NUM_JOINTS)))

    @xax.jit(static_argnames=["self", "model_static"], jit_level=5)
    def _get_discriminator_loss_and_metrics(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        rollout_motion: Motion,
        reference_motion: Motion,
    ) -> tuple[Array, xax.FrozenDict[str, Array]]:
        """Adds the discriminator loss to the super's loss and metrics."""

        model = eqx.combine(model_arr, model_static)
        assert isinstance(model, DefaultHumanoidAMPModel)
        fake_logits = jax.vmap(model.discriminator.forward)(rollout_motion)
        real_logits = jax.vmap(model.discriminator.forward)(reference_motion)

        # Compute hinge loss for both real and fake samples
        fake_loss = jnp.mean(jnp.maximum(0.0, 1.0 + fake_logits))  # Want fake samples <= -1
        real_loss = jnp.mean(jnp.maximum(0.0, 1.0 - real_logits))  # Want real samples >= 1
        discriminator_loss = fake_loss + real_loss

        metrics = xax.FrozenDict(
            {
                "discriminator_loss": discriminator_loss,
                "discriminator_fake_loss": fake_loss,
                "discriminator_real_loss": real_loss,
                "discriminator_fake": fake_logits,
                "discriminator_real": real_logits,
            }
        )

        return discriminator_loss, metrics

    @xax.jit(static_argnames=["self", "model_static"], jit_level=3)
    def _get_discriminator_metrics_and_grads(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        rollout_motion: Motion,
        reference_motion: Motion,
    ) -> tuple[Array, xax.FrozenDict[str, Array]]:
        loss_fn = jax.grad(self._get_discriminator_loss_and_metrics, argnums=0, has_aux=True)
        loss_fn = xax.jit(static_argnums=[1], jit_level=3)(loss_fn)
        grads, metrics = loss_fn(
            model_arr,
            model_static,
            rollout_motion,
            reference_motion,
        )
        return metrics, grads

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
        super_model_arr, super_opt_state, super_model_carry, super_metrics, super_logged_traj = super().update_model(
            optimizer,
            opt_state,
            trajectories,
            rewards,
            rollout_env_states,
            rollout_shared_state,
            rollout_constants,
            rng,
        )

        rng_sample, rng_train = jax.random.split(rng)
        rollout_indices = jnp.arange(trajectories.done.shape[0]).reshape(self.num_batches, self.batch_size)
        # Randomly sample reference motion indices with replacement
        reference_indices = jax.random.randint(
            rng_sample,
            shape=(self.num_batches, self.batch_size),
            minval=0,
            maxval=self.reference_motions.qpos_frames.shape[0],
        )
        indices_by_batch = (rollout_indices, reference_indices)

        def update_discriminator_in_batch(
            carry_training_state: tuple[PyTree, optax.OptState, PRNGKeyArray], batch_indices: tuple[Array, Array]
        ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], xax.FrozenDict[str, Array]]:
            assert isinstance(trajectories.aux_outputs, AMPAuxOutputs)
            rollout_batch_indices, reference_batch_indices = batch_indices
            model_arr, opt_state, rng = carry_training_state
            reference_motion_batch = jax.tree.map(lambda x: x[reference_batch_indices], self.reference_motions)
            rollout_motion_batch = jax.tree.map(lambda x: x[rollout_batch_indices], trajectories.aux_outputs.motion)

            disc_metrics, disc_grads = self._get_discriminator_metrics_and_grads(
                model_arr=model_arr,
                model_static=rollout_constants.model_static,
                rollout_motion=rollout_motion_batch,
                reference_motion=reference_motion_batch,
            )

            new_model_arr, new_opt_state, grad_metrics = self.apply_gradients_with_clipping(
                model_arr=model_arr,
                grads=disc_grads,
                optimizer=optimizer,
                opt_state=opt_state,
            )

            metrics = xax.FrozenDict(dict(disc_metrics) | dict(grad_metrics))

            return (new_model_arr, new_opt_state, rng), metrics

        def update_discriminator_accross_batches(
            carry_training_state: tuple[PyTree, optax.OptState, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], xax.FrozenDict[str, Array]]:
            carry_training_state, metrics = jax.lax.scan(
                update_discriminator_in_batch, carry_training_state, indices_by_batch
            )

            return carry_training_state, metrics

        (model_arr, opt_state, _), discriminator_metrics = jax.lax.scan(
            update_discriminator_accross_batches,
            (super_model_arr, super_opt_state, rng_train),
            length=self.config.num_passes,
        )

        metrics = xax.FrozenDict(dict(super_metrics) | dict(discriminator_metrics))

        return model_arr, opt_state, super_model_carry, metrics, super_logged_traj

    def run(self) -> None:
        mj_model: PhysicsModel = self.get_mujoco_model()
        root: BvhioJoint = bvhio.readAsHierarchy(self.config.bvh_path)
        reference_base_id = get_reference_joint_id(root, self.config.reference_base_name)
        mj_base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, self.config.mj_base_name)

        def rotation_callback(root: BvhioJoint) -> None:
            euler_rotation = np.array(self.config.rotate_bvh_euler)
            quat = R.from_euler("xyz", euler_rotation).as_quat(scalar_first=True)
            root.applyRotation(glm.quat(*quat), bake=True)

        np_reference_cartesian_poses = get_reference_cartesian_poses(
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
            mj_base_id=mj_base_id,
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
        reference_qpos = jnp.array(np_reference_qpos)

        # Getting the reference motion from the reference qpos.
        def build_reference_motion(motion: Motion, qpos: Array) -> tuple[Motion, Motion]:
            motion = jax.tree.map(lambda x: x[1:], motion)
            motion = Motion(jnp.concatenate([motion.qpos_frames, qpos[None]], axis=0))
            return motion, motion

        _, self.reference_motions = jax.lax.scan(
            build_reference_motion, self._get_initial_motion_carry(), reference_qpos
        )

        if self.config.visualize_reference_points:
            visualize_reference_points(
                model=mj_model,
                base_id=mj_base_id,
                reference_motion=np_reference_cartesian_poses,
            )
        elif self.config.visualize_reference_motion:
            visualize_reference_motion(
                model=mj_model,
                reference_qpos=np_reference_qpos,
                cartesian_motion=np_reference_cartesian_poses,
                mj_base_id=mj_base_id,
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
