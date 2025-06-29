# mypy: disable-error-code="override"
"""Example walking task using Adversarial Motion Priors."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
import xax
from jaxtyping import Array, PRNGKeyArray

import ksim

NUM_JOINTS = 21

NUM_INPUTS = 2 + NUM_JOINTS + NUM_JOINTS + 160 + 96 + 3 + NUM_JOINTS + 3 + 4 + 3 + 3


HUMANOID_REFERENCE_MAPPINGS = (
    ksim.MotionReferenceMapping("CC_Base_L_ThighTwist01", "thigh_left"),  # hip
    ksim.MotionReferenceMapping("CC_Base_L_CalfTwist01", "shin_left"),  # knee
    ksim.MotionReferenceMapping("CC_Base_L_Foot", "foot_left"),  # foot
    ksim.MotionReferenceMapping("CC_Base_L_UpperarmTwist01", "upper_arm_left"),  # shoulder
    ksim.MotionReferenceMapping("CC_Base_L_ForearmTwist01", "lower_arm_left"),  # elbow
    ksim.MotionReferenceMapping("CC_Base_L_Hand", "hand_left"),  # hand
    ksim.MotionReferenceMapping("CC_Base_R_ThighTwist01", "thigh_right"),  # hip
    ksim.MotionReferenceMapping("CC_Base_R_CalfTwist01", "shin_right"),  # knee
    ksim.MotionReferenceMapping("CC_Base_R_Foot", "foot_right"),  # foot
    ksim.MotionReferenceMapping("CC_Base_R_UpperarmTwist01", "upper_arm_right"),  # shoulder
    ksim.MotionReferenceMapping("CC_Base_R_ForearmTwist01", "lower_arm_right"),  # elbow
    ksim.MotionReferenceMapping("CC_Base_R_Hand", "hand_right"),  # hand
)


class DefaultHumanoidRNNActor(eqx.Module):
    """RNN-based actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    num_mixtures: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        num_mixtures: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 3 * num_mixtures,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_mixtures = num_mixtures
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs_n: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        prediction_n = self.output_proj(x_n)

        # Converts the output to a distribution.
        slice_len = self.num_outputs * self.num_mixtures
        mean_nm = prediction_n[:slice_len].reshape(NUM_JOINTS, self.num_mixtures)
        std_nm = prediction_n[slice_len : slice_len * 2].reshape(NUM_JOINTS, self.num_mixtures)
        logits_nm = prediction_n[slice_len * 2 :].reshape(NUM_JOINTS, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)
        dist_n = ksim.MixtureOfGaussians(means_nm=mean_nm, stds_nm=std_nm, logits_nm=logits_nm)
        return dist_n, jnp.stack(out_carries, axis=0)


class DefaultHumanoidRNNCritic(eqx.Module):
    """RNN-based critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for _ in range(depth)
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class DefaultHumanoidRNNModel(eqx.Module):
    actor: DefaultHumanoidRNNActor
    critic: DefaultHumanoidRNNCritic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        num_inputs: int,
        num_joints: int,
        num_mixtures: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        self.actor = DefaultHumanoidRNNActor(
            key,
            num_inputs=num_inputs,
            num_outputs=num_joints,
            num_mixtures=num_mixtures,
            min_std=min_std,
            max_std=max_std,
            var_scale=0.5,
            hidden_size=hidden_size,
            depth=depth,
        )
        self.critic = DefaultHumanoidRNNCritic(
            key,
            num_inputs=num_inputs,
            hidden_size=hidden_size,
            depth=depth,
        )


class DefaultHumanoidDiscriminator(eqx.Module):
    """Discriminator for the walking task, returns logit."""

    layers: list[Callable[[Array], Array]]

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
        num_frames: int,
    ) -> None:
        num_inputs = NUM_JOINTS
        num_outputs = 1

        layers: list[Callable[[Array], Array]] = []
        for _ in range(depth):
            key, subkey = jax.random.split(key)
            layers += [
                eqx.nn.Conv1d(
                    in_channels=num_inputs,
                    out_channels=hidden_size,
                    kernel_size=num_frames,
                    # We left-pad the input so that the discriminator output
                    # at time T can be reasonably interpretted as the logit
                    # for the motion from time T - N to T. In other words, we
                    # this means that all the reward for time T will be based
                    # on the motion from the previous N frames.
                    padding=[(num_frames - 1, 0)],
                    key=subkey,
                ),
                jax.nn.relu,
            ]
            num_inputs = hidden_size

        layers += [
            eqx.nn.Conv1d(
                in_channels=num_inputs,
                out_channels=num_outputs,
                kernel_size=1,
                key=key,
            )
        ]

        self.layers = layers

    def forward(self, x: Array) -> Array:
        x_nt = x.transpose(1, 0)
        for layer in self.layers:
            x_nt = layer(x_nt)
        return x_nt.squeeze(0)


@dataclass
class HumanoidWalkingAMPTaskConfig(ksim.AMPConfig):
    # Policy parameters.
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )
    hidden_size: int = xax.field(
        value=512,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=2,
        help="The depth for the MLPs.",
    )

    # Disciminator parameters.
    discriminator_hidden_size: int = xax.field(
        value=512,
        help="The hidden size for the discriminator.",
    )
    discriminator_depth: int = xax.field(
        value=2,
        help="The depth for the discriminator.",
    )
    num_frames: int = xax.field(
        value=10,
        help="The number of frames to use for the discriminator.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=1e-3,
        help="Learning rate for PPO.",
    )
    adam_weight_decay: float = xax.field(
        value=0.0,
        help="Weight decay for the Adam optimizer.",
    )
    max_discriminator_grad_norm: float = xax.field(
        value=2.0,
        help="Maximum gradient norm for clipping.",
    )
    discriminator_learning_rate: float = xax.field(
        value=1e-3,
        help="Learning rate for the discriminator.",
    )

    # Curriculum parameters.
    num_curriculum_levels: int = xax.field(
        value=10,
        help="The number of curriculum levels to use.",
    )
    increase_threshold: float = xax.field(
        value=3.0,
        help="Increase the curriculum level when the mean trajectory length is above this threshold.",
    )
    decrease_threshold: float = xax.field(
        value=1.0,
        help="Decrease the curriculum level when the mean trajectory length is below this threshold.",
    )
    min_level_steps: int = xax.field(
        value=50,
        help="The minimum number of steps to wait before changing the curriculum level.",
    )

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=0,
        help="The body id to track with the render camera.",
    )

    # Refernece motion parameters.
    reference_motion_path: str = xax.field(
        value=str(Path(__file__).parent / "data" / "humanoid_amp_walk_ref.npz"),
        help="The path to the reference motion file.",
    )
    bvh_path: str = xax.field(
        value=str(Path(__file__).parent / "data" / "walk_normal_dh.bvh"),
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

    # Visualization parameters.
    visualize_reference_points: bool = xax.field(
        value=False,
        help="Whether to visualize the reference points.",
    )
    visualize_reference_motion: bool = xax.field(
        value=False,
        help="Whether to visualize the reference motion after running IK.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingAMPTaskConfig)


class HumanoidWalkingAMPTask(ksim.AMPTask[Config], Generic[Config]):
    """Adversarial Motion Prior task."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reference_motion = ksim.MotionReferenceData.load(self.config.reference_motion_path)

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = (Path(__file__).parent / "data" / "scene.mjcf").resolve().as_posix()
        return mujoco.MjModel.from_xml_path(mjcf_path)

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        return ksim.Metadata.from_model(
            mj_model,
            kp=1.0,
            kd=0.1,
            armature=1e-2,
            friction=1e-6,
        )

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(
            physics_model=physics_model,
            metadata=metadata,
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.MassMultiplicationRandomizer.from_body_name(physics_model, "torso"),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(
                x_linvel=1.0,
                y_linvel=1.0,
                z_linvel=0.0,
                x_angvel=0.1,
                y_angvel=0.1,
                z_angvel=0.3,
                interval_range=(0.25, 0.75),
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset(),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(),
            ksim.JointVelocityObservation(),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="orientation",
            ),
            ksim.ActuatorAccelerationObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="local_linvel"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="upvector"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="forwardvector"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="global_linvel"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="global_angvel"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="position"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="orientation"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_global_linvel"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_global_linvel"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_upvector"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_upvector"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_pos"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_pos"),
            ksim.FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names=["foot_left"],
                foot_right_geom_names=["foot_right"],
                floor_geom_names=["floor"],
            ),
            ksim.FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_body_name="foot_left",
                foot_right_body_name="foot_right",
            ),
            ksim.FeetOrientationObservation.create_from_feet(
                physics_model=physics_model,
                foot_left_body_name="foot_left",
                foot_right_body_name="foot_right",
            ),
            ksim.TimestepObservation(),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return []

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.AMPReward(scale=1.0),
            ksim.StayAliveReward(scale=1.0),
            ksim.XYAngularVelocityPenalty(scale=-0.01),
            ksim.NaiveForwardReward(clip_max=1.0, scale=1.0),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.9, unhealthy_z_upper=1.6),
            ksim.NotUprightTermination(max_radians=math.radians(30)),
            ksim.HighVelocityTermination(),
            ksim.FarFromOriginTermination(max_dist=10.0),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.EpisodeLengthCurriculum(
            num_levels=self.config.num_curriculum_levels,
            increase_threshold=self.config.increase_threshold,
            decrease_threshold=self.config.decrease_threshold,
            min_level_steps=self.config.min_level_steps,
        )

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def get_policy_model(self, key: PRNGKeyArray) -> DefaultHumanoidRNNModel:
        return DefaultHumanoidRNNModel(
            key,
            num_inputs=NUM_INPUTS,
            num_joints=NUM_JOINTS,
            num_mixtures=self.config.num_mixtures,
            min_std=0.01,
            max_std=1.0,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
        )

    def get_discriminator_model(self, key: PRNGKeyArray) -> DefaultHumanoidDiscriminator:
        return DefaultHumanoidDiscriminator(
            key,
            hidden_size=self.config.discriminator_hidden_size,
            depth=self.config.discriminator_depth,
            num_frames=self.config.num_frames,
        )

    def get_policy_optimizer(self) -> optax.GradientTransformation:
        return (
            optax.adam(self.config.learning_rate)
            if self.config.adam_weight_decay == 0.0
            else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
        )

    def get_discriminator_optimizer(self) -> optax.GradientTransformation:
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_discriminator_grad_norm),
            optax.adam(self.config.discriminator_learning_rate),
        )
        return optimizer

    def call_discriminator(self, model: DefaultHumanoidDiscriminator, motion: Array) -> Array:
        return model.forward(motion).squeeze()

    def get_real_motions(self, mj_model: mujoco.MjModel) -> Array:
        reference_motion = ksim.MotionReferenceData.load(self.config.reference_motion_path)
        # cartesian_poses = jax.tree.map(lambda x: np.asarray(x.array), reference_motion.cartesian_poses)
        return jnp.array(reference_motion.qpos.array[None, ..., 7:])  # Remove the root joint absolute coordinates.

    def trajectory_to_motion(self, trajectory: ksim.Trajectory) -> Array:
        # xpos = trajectory.xpos
        qpos = trajectory.qpos[..., 7:]
        return qpos

    def motion_to_qpos(self, motion: Array) -> Array:
        qpos_init = jnp.array([0.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0])
        return jnp.concatenate([jnp.broadcast_to(qpos_init, (*motion.shape[:-1], 7)), motion], axis=-1)

    def run_actor(
        self,
        model: DefaultHumanoidRNNActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        # imu_acc_3 = observations["sensor_observation_imu_acc"]
        # imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def run_critic(
        self,
        model: DefaultHumanoidRNNCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        # imu_acc_3 = observations["sensor_observation_imu_acc"]
        # imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def get_ppo_variables(
        self,
        model: DefaultHumanoidRNNModel,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        def scan_fn(
            actor_critic_carry: tuple[Array, Array], transition: ksim.Trajectory
        ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
            actor_carry, critic_carry = actor_critic_carry
            actor_dist, next_actor_carry = self.run_actor(
                model=model.actor,
                observations=transition.obs,
                commands=transition.command,
                carry=actor_carry,
            )
            log_probs = actor_dist.log_prob(transition.action)
            assert isinstance(log_probs, Array)
            value, next_critic_carry = self.run_critic(
                model=model.critic,
                observations=transition.obs,
                commands=transition.command,
                carry=critic_carry,
            )

            transition_ppo_variables = ksim.PPOVariables(
                log_probs=log_probs,
                values=value.squeeze(-1),
            )

            initial_carry = self.get_initial_model_carry(rng)
            next_carry = jax.tree.map(
                lambda x, y: jnp.where(transition.done, x, y),
                initial_carry,
                (next_actor_carry, next_critic_carry),
            )

            return next_carry, transition_ppo_variables

        next_model_carry, ppo_variables = xax.scan(scan_fn, model_carry, trajectory, jit_level=ksim.JitLevel.RL_CORE)

        return ppo_variables, next_model_carry

    def sample_action(
        self,
        model: DefaultHumanoidRNNModel,
        model_carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry

        # Runs the actor model to get the action distribution.
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )

        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)

        return ksim.Action(action=action_j, carry=(actor_carry, critic_carry_in))


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.walking_reference_motion
    # To visualize the environment, use the following command:
    #   python -m examples.walking_reference_motion run_mode=view
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
            dt=0.005,
            ctrl_dt=0.02,
            iterations=3,
            ls_iterations=5,
            rollout_length_seconds=8.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            global_grad_clip=1.0,
            # Gait matching parameters.
            rotate_bvh_euler=(0, np.pi / 2, 0),
            bvh_scaling_factor=1 / 100,
            mj_base_name="pelvis",
            reference_base_name="CC_Base_Pelvis",
        ),
    )
