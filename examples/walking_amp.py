# mypy: disable-error-code="override"
"""Example walking task using Adversarial Motion Priors."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
import xax
from jaxtyping import PRNGKeyArray

try:
    import bvhio
    import glm
    from bvhio.lib.hierarchy import Joint as BvhioJoint
except ImportError as e:
    raise ImportError(
        "In order to use reference motion utilities, please install Bvhio, using 'pip install bvhio'."
    ) from e


from jaxtyping import Array, PyTree
from scipy.spatial.transform import Rotation as R

import ksim
from ksim.types import PhysicsModel
from ksim.utils.priors import (
    ReferenceMapping,
    get_reference_cartesian_poses,
    get_reference_joint_id,
    visualize_reference_motion,
    visualize_reference_points,
)

from .walking import (
    NUM_JOINTS,
    DefaultHumanoidActor,
    DefaultHumanoidCritic,
    HumanoidWalkingTask,
    HumanoidWalkingTaskConfig,
)


@jax.tree_util.register_dataclass
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

    def forward(self, x: Array) -> Array:
        return self.mlp(x)


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
    constrained_joint_ids: tuple[int, ...] = xax.field(
        value=(0, 1, 2, 3, 4, 5, 6),
        help="The indices of the joints to constrain. By default, freejoints.",
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
    num_discriminator_passes: int = xax.field(
        value=1,
        help="The number of times to pass the discriminator.",
    )
    w_gp: float = xax.field(
        value=10.0,
        help="Gradient penalty coefficient for discriminator (WGAN-GP style).",
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


class AMPTask(ksim.RLTask[Config], Generic[Config]):
    """Adversarial Motion Prior task."""

    reference_motions: Motion

    @xax.jit(static_argnames=["self", "model_static"], jit_level=5)
    def _get_discriminator_loss_and_metrics(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        rollout_motions: Motion,
        reference_motions: Motion,
    ) -> tuple[Array, xax.FrozenDict[str, Array]]:
        """Adds the discriminator loss to the super's loss and metrics using LSGAN objective."""
        model = eqx.combine(model_arr, model_static)

        # Vmapping over batch size and trajectory length.
        fake_logits = jax.vmap(jax.vmap(self.run_discriminator, in_axes=(None, 0)), in_axes=(None, 0))(
            model, rollout_motions
        )
        real_logits = jax.vmap(jax.vmap(self.run_discriminator, in_axes=(None, 0)), in_axes=(None, 0))(
            model, reference_motions
        )

        # Compute hinge loss for both real and fake samples
        # fake_loss = jnp.mean(jnp.maximum(0.0, 1.0 + fake_logits))  # Want fake samples <= -1
        # real_loss = jnp.mean(jnp.maximum(0.0, 1.0 - real_logits))  # Want real samples >= 1
        # discriminator_loss = fake_loss + real_loss

        # LSGAN loss:
        #   real_loss = mean((D(real) - 1)^2)
        #   fake_loss = mean((D(fake) + 1)^2)
        #   discriminator_loss = real_loss + fake_loss
        real_loss = jnp.mean(jnp.square(real_logits - 1.0))
        fake_loss = jnp.mean(jnp.square(fake_logits + 1.0))
        discriminator_loss = real_loss + fake_loss
        # TODO: add in gradient clipping

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
        rollout_motions: Motion,
        reference_motions: Motion,
    ) -> tuple[Array, xax.FrozenDict[str, Array]]:
        """Gets the discriminator metrics and gradients."""
        loss_fn = jax.grad(self._get_discriminator_loss_and_metrics, argnums=0, has_aux=True)
        loss_fn = xax.jit(static_argnums=[1], jit_level=3)(loss_fn)
        grads, metrics = loss_fn(
            model_arr,
            model_static,
            rollout_motions,
            reference_motions,
        )
        return metrics, grads

    def _get_rollout_motions(self, trajectory: ksim.Trajectory) -> Motion:
        """Gets the rollout motions from the trajectory."""

        def build_motion(carry: Motion, transition: ksim.Trajectory) -> tuple[Motion, Motion]:
            # Shifts the motion history by one frame.
            next_motion = Motion(
                qpos_frames=jnp.concatenate(
                    [carry.qpos_frames[1:], transition.qpos[None, ~self.constrained_jnt_mask]], axis=0
                )
            )
            next_motion = jax.lax.cond(
                transition.done,
                lambda: self.initial_motion_history(),
                lambda: next_motion,
            )
            return next_motion, next_motion

        _, rollout_motions = jax.lax.scan(build_motion, self.initial_motion_history(), trajectory)
        return rollout_motions

    def _get_reference_motions(self, reference_qpos: Array) -> Motion:
        """Gets the reference motions from the reference qpos."""

        def build_motion(carry: Motion, qpos: Array) -> tuple[Motion, Motion]:
            next_motion = Motion(
                qpos_frames=jnp.concatenate([carry.qpos_frames[1:], qpos[None, ~self.constrained_jnt_mask]], axis=0)
            )
            return next_motion, next_motion

        _, reference_motions = jax.lax.scan(build_motion, self.initial_motion_history(), reference_qpos)
        return reference_motions

    def post_rollout_update(
        self,
        trajectory: ksim.Trajectory,
        rollout_env_state: ksim.RolloutEnvState,
        rollout_shared_state: ksim.RolloutSharedState,
        rollout_constants: ksim.RolloutConstants,
    ) -> tuple[ksim.Trajectory, ksim.RolloutEnvState, ksim.RolloutSharedState]:
        """Adding AMP-specific variables to the aux_outputs bus."""
        model = eqx.combine(rollout_shared_state.model_arr, rollout_constants.model_static)
        rollout_motions = self._get_rollout_motions(trajectory)
        on_policy_discriminator_logits = jax.vmap(self.run_discriminator, in_axes=(None, 0))(model, rollout_motions)
        aux_outputs = {"_discriminator_logits": on_policy_discriminator_logits, "_rollout_motions": rollout_motions}
        if trajectory.aux_outputs is None:
            aux_outputs = xax.FrozenDict(aux_outputs)
        else:
            aux_outputs = dict(trajectory.aux_outputs) | aux_outputs

        trajectory = replace(trajectory, aux_outputs=aux_outputs)
        return trajectory, rollout_env_state, rollout_shared_state

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
        """Updates the model using the AMP task.

        Args:
            optimizer: The optimizer to use.
            opt_state: The optimizer state.
            trajectories: The trajectories to update the model on. (num_envs, num_steps, leaf_dim)
            rewards: The rewards for the trajectories. (num_envs, num_steps)
            rollout_env_states: The environment variables inputs into the rollout.
            rollout_shared_state: The shared state inputs into the rollout.
            rollout_constants: The constant inputs into the rollout.
            rng: A random seed.

        Returns:
            A tuple containing the updated parameters, optimizer state, next
            model carry, metrics, and the single trajectory to log.
        """
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
        assert trajectories.aux_outputs is not None
        rollout_motions = trajectories.aux_outputs["_rollout_motions"]

        # Randomly sample rollout and reference, randomly zero out reference.
        rollout_indices = jnp.arange(trajectories.done.shape[0]).reshape(self.num_batches, self.batch_size)
        reference_indices = jax.random.randint(
            rng_sample,
            shape=(self.num_batches, self.batch_size, self.rollout_length_steps),
            minval=0,
            maxval=self.reference_motions.qpos_frames.shape[0],
        )

        # During discriminator training, the motions at the start of the rollout
        # will begin mid-episode. For abstraction, we reset the motion history
        # at the start of each rollout as opposed to forcing the user to store
        # the motion history in the carry term. As such, we must match
        # the mid-episode reset distribution in the reference.
        rng_reset, rng_uniform = jax.random.split(rng_sample)
        reset_probability = (self.config.discriminator_num_frames - 1) / self.rollout_length_steps
        do_reset = jax.random.bernoulli(
            rng_reset, p=reset_probability, shape=(self.num_batches, self.batch_size, self.rollout_length_steps)
        )
        num_frames_to_reset = jax.random.randint(
            rng_uniform,
            shape=(self.num_batches, self.batch_size, self.rollout_length_steps),
            minval=1,
            maxval=self.config.discriminator_num_frames - 1,
        )
        num_frames_to_reset = jnp.where(do_reset, num_frames_to_reset, 0)
        indices_by_batch = (rollout_indices, reference_indices, num_frames_to_reset)

        def update_discriminator_in_batch(
            carry_training_state: tuple[PyTree, optax.OptState, PRNGKeyArray], batch_indices: tuple[Array, Array, Array]
        ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], xax.FrozenDict[str, Array]]:
            rollout_batch_indices, reference_batch_indices, num_frames_to_reset = batch_indices
            model_arr, opt_state, rng = carry_training_state
            rollout_motions_batch = jax.tree.map(lambda x: x[rollout_batch_indices], rollout_motions)
            reference_motions_batch = jax.tree.map(
                lambda x: x[reference_batch_indices.reshape(-1)].reshape(
                    self.batch_size, self.rollout_length_steps, *x.shape[1:]
                ),
                self.reference_motions,
            )
            # TODO: actually add in reset logic here

            disc_metrics, disc_grads = self._get_discriminator_metrics_and_grads(
                model_arr=model_arr,
                model_static=rollout_constants.model_static,
                rollout_motions=rollout_motions_batch,
                reference_motions=reference_motions_batch,
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
            length=self.config.num_discriminator_passes,
        )

        metrics = xax.FrozenDict(dict(super_metrics) | dict(discriminator_metrics))

        # TODO: update super_logged_traj with discriminator metrics
        return model_arr, opt_state, super_model_carry, metrics, super_logged_traj

    def _validate_amp_rewards(self, physics_model: ksim.PhysicsModel) -> None:
        rewards = self.get_rewards(physics_model)
        has_amp_reward = False
        for reward in rewards:
            if isinstance(reward, AMPReward):
                has_amp_reward = True
                break
        assert has_amp_reward, "AMPReward must be in the rewards list"

    def run(self) -> None:
        mj_model: PhysicsModel = self.get_mujoco_model()
        self._validate_amp_rewards(mj_model)
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
            constrained_joint_ids=self.config.constrained_joint_ids,
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
        constrained_jnt_mask = np.zeros(np_reference_qpos[0].shape, dtype=bool)
        constrained_jnt_mask[np.array(self.config.constrained_joint_ids)] = True
        self.constrained_jnt_mask = constrained_jnt_mask
        reference_qpos = jnp.array(np_reference_qpos)
        self.reference_motions = self._get_reference_motions(reference_qpos)

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


class HumanoidWalkingAMPTask(AMPTask[HumanoidWalkingAMPTaskConfig], HumanoidWalkingTask[HumanoidWalkingAMPTaskConfig]):
    """Humanoid walking task with adversarial motion prior.

    This class combines the standard humanoid walking task with AMP (Adversarial Motion Prior)
    functionality to enable learning from reference motions.
    """

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

    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidAMPModel:
        return DefaultHumanoidAMPModel(
            key,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_mixtures=self.config.num_mixtures,
            num_frames=self.config.discriminator_num_frames,
        )

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards = [
            ksim.StayAliveReward(scale=1.0),
            ksim.NaiveForwardReward(scale=0.1, clip_max=2.0),
            AMPReward(scale=0.1),
        ]
        return rewards

    def run_discriminator(self, model: DefaultHumanoidAMPModel, motion: Motion) -> Array:
        x_qpos = motion.qpos_frames.reshape(-1)
        return model.discriminator.forward(x_qpos).squeeze()

    def initial_motion_history(self) -> Motion:
        return Motion(qpos_frames=jnp.zeros((self.config.discriminator_num_frames, NUM_JOINTS)))


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
