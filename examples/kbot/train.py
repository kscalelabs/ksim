"""Defines simple task for training a walking policy for the default humanoid."""

import asyncio
import functools
import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import mujoco_scenes
import mujoco_scenes.mjcf
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

import ksim

# These are in the order of the neural network outputs.
ZEROS: list[tuple[str, float]] = [
    ("dof_right_shoulder_pitch_03", 0.0),
    ("dof_right_shoulder_roll_03", math.radians(-10.0)),
    ("dof_right_shoulder_yaw_02", 0.0),
    ("dof_right_elbow_02", math.radians(45.0)),
    ("dof_right_wrist_00", 0.0),
    ("dof_left_shoulder_pitch_03", 0.0),
    ("dof_left_shoulder_roll_03", math.radians(10.0)),
    ("dof_left_shoulder_yaw_02", 0.0),
    ("dof_left_elbow_02", math.radians(-45.0)),
    ("dof_left_wrist_00", 0.0),
    ("dof_right_hip_pitch_04", math.radians(-10.0)),
    ("dof_right_hip_roll_03", math.radians(-0.0)),
    ("dof_right_hip_yaw_03", 0.0),
    ("dof_right_knee_04", math.radians(-30.0)),
    ("dof_right_ankle_02", math.radians(20.0)),
    ("dof_left_hip_pitch_04", math.radians(10.0)),
    ("dof_left_hip_roll_03", math.radians(0.0)),
    ("dof_left_hip_yaw_03", 0.0),
    ("dof_left_knee_04", math.radians(30.0)),
    ("dof_left_ankle_02", math.radians(-20.0)),
]

ARM_JOINTS: list[str] = [
    "dof_right_shoulder_pitch_03",
    "dof_right_shoulder_roll_03",
    "dof_right_shoulder_yaw_02",
    "dof_right_elbow_02",
    "dof_right_wrist_00",
    "dof_left_shoulder_pitch_03",
    "dof_left_shoulder_roll_03",
    "dof_left_shoulder_yaw_02",
    "dof_left_elbow_02",
    "dof_left_wrist_00",
]


@dataclass
class HumanoidWalkingTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

    # Model parameters.
    hidden_size: int = xax.field(
        value=512,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=2,
        help="The depth for the MLPs.",
    )
    num_hidden_layers: int = xax.field(
        value=2,
        help="The number of hidden layers for the MLPs.",
    )
    var_scale: float = xax.field(
        value=0.5,
        help="The scale for the standard deviations of the actor.",
    )
    start_cutoff_frequency: float = xax.field(
        value=10.0,
        help="The cutoff frequency for the low-pass filter.",
    )
    end_cutoff_frequency: float = xax.field(
        value=4.0,
        help="The cutoff frequency for the low-pass filter.",
    )

    # Reward parameters.
    linear_velocity_range: tuple[float, float] = xax.field(
        value=(1.0, 3.0),
        help="The range for the linear velocity command.",
    )
    linear_velocity_max_yaw: float = xax.field(
        value=math.radians(45.0),
        help="The maximum yaw for the linear velocity command.",
    )
    linear_velocity_zero_prob: float = xax.field(
        value=0.2,
        help="The probability of the linear velocity command being zero.",
    )
    linear_velocity_backward_prob: float = xax.field(
        value=0.0,
        help="The probability of the linear velocity command being backward.",
    )
    linear_velocity_switch_prob: float = xax.field(
        value=0.005,
        help="The probability of the linear velocity command being switched.",
    )
    angular_velocity_range: tuple[float, float] = xax.field(
        value=(-math.radians(45), math.radians(45)),
        help="The range for the angular velocity command.",
    )
    angular_velocity_zero_prob: float = xax.field(
        value=0.2,
        help="The probability of the angular velocity command being zero.",
    )
    angular_velocity_switch_prob: float = xax.field(
        value=0.005,
        help="The probability of the angular velocity command being switched.",
    )
    gait_period: float = xax.field(
        value=0.6,
        help="The target period for the gait.",
    )
    max_knee_height: float = xax.field(
        value=0.6,
        help="The maximum height of the foot.",
    )
    max_foot_height: float = xax.field(
        value=0.2,
        help="The maximum height of the foot.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=1e-3,
        help="Learning rate for PPO.",
    )
    grad_clip: float = xax.field(
        value=1.0,
        help="Gradient clipping for the optimizer.",
    )
    warmup_steps: int = xax.field(
        value=1000,
        help="Number of warmup steps for the optimizer.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Carry:
    actor_carry: Array = eqx.field()
    critic_carry: Array = eqx.field()
    lpf_params: ksim.LowPassFilterParams = eqx.field()


class Actor(eqx.Module):
    """Actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.LSTMCell, ...]
    output_proj: eqx.nn.MLP
    num_inputs: int = eqx.field()
    num_outputs: int = eqx.field()
    clip_positions: ksim.TanhPositions = eqx.field()
    min_std: float = eqx.field()
    max_std: float = eqx.field()
    var_scale: float = eqx.field()
    ctrl_dt: float = eqx.field()
    start_fc: float = eqx.field()
    end_fc: float = eqx.field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        physics_model: ksim.PhysicsModel,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
        num_hidden_layers: int,
        ctrl_dt: float,
        start_cutoff_frequency: float,
        end_cutoff_frequency: float,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layers (LSTM)
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.LSTMCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=k,
                )
                for k in rnn_keys
            ]
        )

        # Project to output
        output_proj = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=num_outputs * 2,
            width_size=hidden_size,
            depth=num_hidden_layers,
            activation=jax.nn.gelu,
            use_final_bias=True,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.clip_positions = ksim.TanhPositions.from_physics_model(physics_model)
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.ctrl_dt = ctrl_dt
        self.start_fc = start_cutoff_frequency
        self.end_fc = end_cutoff_frequency

        # Sets the bias of the output projection.
        bias = self.clip_positions.get_bias(jnp.array([v for _, v in ZEROS]))
        self.output_proj = eqx.tree_at(
            lambda m: m.layers[-1].bias,
            output_proj,
            replace_fn=lambda b: b.at[..., : bias.shape[0]].set(bias),
        )

    def forward(
        self,
        obs_n: Array,
        carry: Array,
        curriculum_level: Array,
        lpf_params: ksim.LowPassFilterParams,
    ) -> tuple[xax.Distribution, Array, ksim.LowPassFilterParams]:
        # carry shape: (2, depth, hidden_size) -> [0]=h, [1]=c
        x_n = self.input_proj(obs_n)
        new_h = []
        new_c = []
        for i, rnn in enumerate(self.rnns):
            h_i = carry[0, i]
            c_i = carry[1, i]
            h_o, c_o = rnn(x_n, (h_i, c_i))
            x_n = h_o
            new_h.append(h_o)
            new_c.append(c_o)
        out_n = self.output_proj(x_n)

        # Reshape the output to be a mixture of gaussians.
        slice_len = self.num_outputs
        mean_n = out_n[..., :slice_len]
        std_n = out_n[..., slice_len:]

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        # Clip the target positions to the minimum and maximum ranges.
        mean_n = self.clip_positions.clip(mean_n)

        # Applies a low-pass filter.
        fc = self.end_fc * curriculum_level + self.start_fc * (1.0 - curriculum_level)
        mean_n, lpf_params = ksim.lowpass_one_pole(mean_n, self.ctrl_dt, fc, lpf_params)

        # Creates a normal distribution.
        dist_n = xax.Normal(loc_n=mean_n, scale_n=std_n)

        next_carry = jnp.stack([jnp.stack(new_h, axis=0), jnp.stack(new_c, axis=0)], axis=0)
        return dist_n, next_carry, lpf_params


class Critic(eqx.Module):
    """Critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.LSTMCell, ...]
    output_proj: eqx.nn.MLP
    num_inputs: int = eqx.field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
        num_hidden_layers: int,
    ) -> None:
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layers (LSTM)
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.LSTMCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=k,
                )
                for k in rnn_keys
            ]
        )

        # Create MLP
        self.output_proj = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=num_outputs,
            width_size=hidden_size,
            depth=num_hidden_layers,
            activation=jax.nn.elu,
            use_final_bias=False,
            key=key,
        )

        self.num_inputs = num_inputs

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        # carry shape: (2, depth, hidden_size) -> [0]=h, [1]=c
        x_n = self.input_proj(obs_n)
        new_h = []
        new_c = []
        for i, rnn in enumerate(self.rnns):
            h_i = carry[0, i]
            c_i = carry[1, i]
            h_o, c_o = rnn(x_n, (h_i, c_i))
            x_n = h_o
            new_h.append(h_o)
            new_c.append(c_o)
        out_n = self.output_proj(x_n)

        next_carry = jnp.stack([jnp.stack(new_h, axis=0), jnp.stack(new_c, axis=0)], axis=0)
        return out_n, next_carry


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        physics_model: ksim.PhysicsModel,
        num_actor_inputs: int,
        num_actor_outputs: int,
        num_critic_inputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
        num_hidden_layers: int,
        ctrl_dt: float,
        start_cutoff_frequency: float,
        end_cutoff_frequency: float,
    ) -> None:
        actor_key, critic_key = jax.random.split(key)
        self.actor = Actor(
            actor_key,
            physics_model=physics_model,
            num_inputs=num_actor_inputs,
            num_outputs=num_actor_outputs,
            min_std=min_std,
            max_std=max_std,
            var_scale=var_scale,
            hidden_size=hidden_size,
            depth=depth,
            num_hidden_layers=num_hidden_layers,
            ctrl_dt=ctrl_dt,
            start_cutoff_frequency=start_cutoff_frequency,
            end_cutoff_frequency=end_cutoff_frequency,
        )
        self.critic = Critic(
            critic_key,
            hidden_size=hidden_size,
            depth=depth,
            num_inputs=num_critic_inputs,
            num_hidden_layers=num_hidden_layers,
        )


class HumanoidWalkingTask(ksim.PPOTask[HumanoidWalkingTaskConfig]):
    def get_optimizer(self) -> optax.GradientTransformation:
        scheduler = optax.warmup_constant_schedule(
            init_value=self.config.learning_rate * 0.01,
            peak_value=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
        )

        def decay_mask(pytree: PyTree) -> PyTree:
            return jax.tree.map(lambda w: w.ndim >= 2, pytree)

        return optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.masked(optax.add_decayed_weights(self.config.adam_weight_decay), decay_mask),
            optax.scale_by_adam(),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0),
        )

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path("kbot-headless", name="robot"))
        return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        metadata = asyncio.run(ksim.get_mujoco_model_metadata("kbot-headless"))
        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is not available")
        if metadata.actuator_type_to_metadata is None:
            raise ValueError("Actuator metadata is not available")
        return metadata

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

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.PhysicsRandomizer]:
        return {
            "static_friction": ksim.StaticFrictionRandomizer(),
            "floor_friction": ksim.FloorFrictionRandomizer.from_geom_name(
                physics_model,
                "floor",
                scale_lower=0.98,
                scale_upper=1.02,
            ),
            "armature": ksim.ArmatureRandomizer(),
            "all_bodies_mass_multiplication": ksim.AllBodiesMassMultiplicationRandomizer(
                scale_lower=0.5,
                scale_upper=1.5,
            ),
            "joint_damping": ksim.JointDampingRandomizer(),
            "joint_zero_position": ksim.JointZeroPositionRandomizer(
                scale_lower=math.radians(-2),
                scale_upper=math.radians(2),
            ),
        }

    def get_events(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Event]:
        return {
            "push": ksim.LinearPushEvent(
                linvel=1.0,
                vel_range=(0.0, 1.0),
                interval_range=(4.0, 8.0),
                curriculum_range=(0.0, 1.0),
            ),
            "jump": ksim.JumpEvent(
                jump_height_range=(0.1, 0.3),
                interval_range=(4.0, 8.0),
                curriculum_range=(0.0, 1.0),
            ),
        }

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v in ZEROS}, scale=0.1),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Observation]:
        return {
            "joint_position": ksim.JointPositionObservation(noise=ksim.AdditiveUniformNoise(mag=math.radians(2))),
            "joint_velocity": ksim.JointVelocityObservation(noise=ksim.AdditiveUniformNoise(mag=math.radians(30))),
            "actuator_force": ksim.ActuatorForceObservation(),
            "center_of_mass_inertia": ksim.CenterOfMassInertiaObservation(),
            "center_of_mass_velocity": ksim.CenterOfMassVelocityObservation(),
            "base_position": ksim.BasePositionObservation(),
            "base_orientation": ksim.BaseOrientationObservation(),
            "base_linear_velocity": ksim.BaseLinearVelocityObservation(),
            "base_angular_velocity": ksim.BaseAngularVelocityObservation(),
            "base_linear_acceleration": ksim.BaseLinearAccelerationObservation(),
            "base_angular_acceleration": ksim.BaseAngularAccelerationObservation(),
            "actuator_acceleration": ksim.ActuatorAccelerationObservation(),
            "imu_projected_gravity": ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
                noise=ksim.AdditiveGaussianNoise(std=0.01),
                min_lag=0.001,
                max_lag=0.005,
                bias=math.radians(2.0),
            ),
            "projected_gravity": ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="imu_site_quat",
            ),
            "imu_acc": ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=ksim.AdditiveUniformNoise(mag=0.01),
            ),
            "imu_gyro": ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=ksim.AdditiveUniformNoise(mag=math.radians(10)),
            ),
            "feet_contact": ksim.FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names=[
                    "KB_D_501L_L_LEG_FOOT_collision_capsule_0",
                    "KB_D_501L_L_LEG_FOOT_collision_capsule_1",
                ],
                foot_right_geom_names=[
                    "KB_D_501R_R_LEG_FOOT_collision_capsule_0",
                    "KB_D_501R_R_LEG_FOOT_collision_capsule_1",
                ],
                floor_geom_names=["floor"],
            ),
            "feet_position": ksim.BodyPositionObservation.create(
                physics_model=physics_model,
                body_names=("KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"),
            ),
            "foot_position": ksim.BodyPositionObservation.create(
                physics_model=physics_model,
                body_names=("KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"),
            ),
            "knee_position": ksim.BodyPositionObservation.create(
                physics_model=physics_model,
                body_names=("KC_D_301R_R_Femur_Lower_Drive", "KC_D_301L_L_Femur_Lower_Drive"),
            ),
            "feet_force": ksim.FeetForceObservation.create(
                physics_model=physics_model,
                foot_left_body_name="KB_D_501L_L_LEG_FOOT",
                foot_right_body_name="KB_D_501R_R_LEG_FOOT",
            ),
        }

    def get_commands(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Command]:
        return {
            "linvel": ksim.LinearVelocityCommand(
                min_vel=self.config.linear_velocity_range[0],
                max_vel=self.config.linear_velocity_range[1],
                max_yaw=self.config.linear_velocity_max_yaw,
                zero_prob=self.config.linear_velocity_zero_prob,
                backward_prob=self.config.linear_velocity_backward_prob,
                switch_prob=self.config.linear_velocity_switch_prob,
            ),
            "angvel": ksim.AngularVelocityCommand(
                min_vel=self.config.angular_velocity_range[0],
                max_vel=self.config.angular_velocity_range[1],
                zero_prob=self.config.angular_velocity_zero_prob,
                switch_prob=self.config.angular_velocity_switch_prob,
            ),
            "arm_position": ksim.JointPositionCommand.create(
                physics_model=physics_model,
                joint_names=ARM_JOINTS,
                ctrl_dt=self.config.ctrl_dt,
                min_time=0.3,
                max_time=2.0,
            ),
        }

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Reward]:
        return {
            "stay_alive": ksim.StayAliveReward(scale=100.0),
            # Command tracking rewards.
            "linvel": ksim.LinearVelocityReward(cmd="linvel", scale=1.0),
            "angvel": ksim.AngularVelocityReward(cmd="angvel", scale=0.5),
            "arm_position": ksim.JointPositionReward.create(
                physics_model=physics_model,
                joint_names=ARM_JOINTS,
                command_name="arm_position",
                length_scale=0.25,
                sq_scale=0.1,
                abs_scale=0.1,
                scale=1.0,
                scale_by_curriculum=True,
            ),
            # Gait rewards.
            "hip_deviation": ksim.JointDeviationPenalty.create(
                physics_model=physics_model,
                joint_names=("dof_right_hip_roll_03", "dof_left_hip_roll_03"),
                joint_targets=(0.0, 0.0),
                scale=-1.0,
            ),
            "foot_airtime": ksim.FeetAirTimeReward(
                ctrl_dt=self.config.ctrl_dt,
                max_air_time=self.config.gait_period * 0.4,
                max_ground_time=self.config.gait_period * 0.6,
                contact_obs="feet_contact",
                scale=10.0,
            ),
            "foot_grounded": ksim.FeetGroundedAtRestReward(
                ctrl_dt=self.config.ctrl_dt,
                max_ground_time=self.config.gait_period * 0.6,
                contact_obs="feet_contact",
                scale=1.0,
            ),
            "knee_height": ksim.TargetHeightReward(
                position_obs="knee_position",
                height=self.config.max_knee_height,
                scale=1.0,
            ),
            "foot_height": ksim.TargetHeightReward(
                position_obs="foot_position",
                height=self.config.max_foot_height,
                scale=1.0,
            ),
            "foot_contact": ksim.ForcePenalty(
                force_obs="feet_force",
                ctrl_dt=self.config.ctrl_dt,
                bias=350.0,  # Weight of the robot, in Newtons.
                scale=-1.0,
            ),
            "upright": ksim.UprightReward(scale=1.0),
            # Normalization penalties.
            "ctrl": ksim.SmallCtrlReward.create(model=physics_model, scale=0.1),
            "joint_velocity": ksim.SmallJointVelocityReward(scale=0.1, kernel_scale=0.25),
            "joint_acceleration": ksim.SmallJointAccelerationReward(scale=0.1, kernel_scale=0.25),
            "joint_jerk": ksim.SmallJointJerkReward(scale=0.1, kernel_scale=0.25),
        }

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Termination]:
        return {
            "bad_z": ksim.BadZTermination(min_z=0.3, final_min_z=0.6, max_z=1.2),
            "far_from_origin": ksim.FarFromOriginTermination(max_dist=20.0),
        }

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.DistanceFromOriginCurriculum(
            min_level=0.001,
            min_level_steps=25,
            increase_threshold=8.0,
            decrease_threshold=8.0,
        )

    def get_model(self, params: ksim.InitParams) -> Model:
        return Model(
            params.key,
            physics_model=params.physics_model,
            num_actor_inputs=59,
            num_actor_outputs=len(ZEROS),
            num_critic_inputs=473,
            min_std=0.01,
            max_std=1.0,
            var_scale=self.config.var_scale,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_hidden_layers=self.config.num_hidden_layers,
            ctrl_dt=self.config.ctrl_dt,
            start_cutoff_frequency=self.config.start_cutoff_frequency,
            end_cutoff_frequency=self.config.end_cutoff_frequency,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, PyTree],
        commands: xax.FrozenDict[str, PyTree],
        carry: Array,
        curriculum_level: Array,
        lpf_params: ksim.LowPassFilterParams,
    ) -> tuple[xax.Distribution, Array, ksim.LowPassFilterParams]:
        joint_pos_n = observations["noisy_joint_position"]
        joint_vel_n = observations["noisy_joint_velocity"]
        proj_grav_3 = observations["noisy_imu_projected_gravity"]
        imu_gyro_3 = observations["noisy_imu_gyro"]

        # Command tensors.
        linvel_cmd: ksim.LinearVelocityCommandValue = commands["linvel"]
        angvel_cmd: ksim.AngularVelocityCommandValue = commands["angvel"]
        joint_pos_cmd: ksim.JointPositionCommandValue = commands["arm_position"]

        # Stacks into tensors.
        linvel_cmd_2 = jnp.stack([linvel_cmd.vel, linvel_cmd.yaw], axis=-1)
        angvel_cmd_1 = jnp.stack([angvel_cmd.vel], axis=-1)

        obs = [
            joint_pos_n,  # NUM_JOINTS
            joint_vel_n / 10.0,  # NUM_JOINTS
            proj_grav_3,  # 3
            imu_gyro_3,  # 3
            linvel_cmd_2,  # 2
            angvel_cmd_1,  # 1
            joint_pos_cmd.current_position,  # 10
        ]

        obs_n = jnp.concatenate(obs, axis=-1)
        action, carry, lpf_params = model.forward(obs_n, carry, curriculum_level, lpf_params)

        return action, carry, lpf_params

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, PyTree],
        commands: xax.FrozenDict[str, PyTree],
        carry: Array,
    ) -> tuple[Array, Array]:
        dh_joint_pos_j = observations["joint_position"]
        dh_joint_vel_j = observations["joint_velocity"]
        com_inertia_n = observations["center_of_mass_inertia"]
        com_vel_n = observations["center_of_mass_velocity"]
        imu_acc_3 = observations["imu_acc"]
        imu_gyro_3 = observations["imu_gyro"]
        proj_grav_3 = observations["projected_gravity"]
        act_frc_obs_n = observations["actuator_force"]
        base_pos_3 = observations["base_position"]
        base_quat_4 = observations["base_orientation"]
        lin_vel_obs_3 = observations["base_linear_velocity"]
        ang_vel_obs_3 = observations["base_angular_velocity"]
        feet_contact_2 = observations["feet_contact"].any(axis=-2)
        feet_height_2 = observations["feet_position"][..., 2]
        feet_force_obs_23 = observations["feet_force"]

        # Flattens the last two dimensions.
        feet_force_obs_6 = feet_force_obs_23.reshape(*feet_force_obs_23.shape[:-2], 6)

        # Command tensors.
        linvel_cmd: ksim.LinearVelocityCommandValue = commands["linvel"]
        angvel_cmd: ksim.AngularVelocityCommandValue = commands["angvel"]
        joint_pos_cmd: ksim.JointPositionCommandValue = commands["arm_position"]

        # Stacks into tensors.
        linvel_cmd_2 = jnp.stack([linvel_cmd.vel, linvel_cmd.yaw], axis=-1)
        angvel_cmd_1 = jnp.stack([angvel_cmd.vel], axis=-1)

        obs_n = jnp.concatenate(
            [
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,
                com_vel_n,
                imu_acc_3,
                imu_gyro_3,
                proj_grav_3,
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,
                base_quat_4,
                lin_vel_obs_3,
                ang_vel_obs_3,
                feet_contact_2,
                feet_height_2,
                feet_force_obs_6 / 100.0,
                linvel_cmd_2,
                angvel_cmd_1,
                joint_pos_cmd.current_position,
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def _model_scan_fn(
        self,
        actor_critic_carry: Carry,
        xs: tuple[ksim.Trajectory, PRNGKeyArray],
        model: Model,
    ) -> tuple[Carry, ksim.PPOVariables]:
        transition, rng = xs

        actor_dist, next_actor_carry, lpf_params = self.run_actor(
            model=model.actor,
            observations=transition.obs,
            commands=transition.command,
            carry=actor_critic_carry.actor_carry,
            curriculum_level=transition.curriculum_level,
            lpf_params=actor_critic_carry.lpf_params,
        )

        # Gets the log probabilities of the action.
        log_probs = actor_dist.log_prob(transition.action)
        assert isinstance(log_probs, Array)

        value, next_critic_carry = self.run_critic(
            model=model.critic,
            observations=transition.obs,
            commands=transition.command,
            carry=actor_critic_carry.critic_carry,
        )

        transition_ppo_variables = ksim.PPOVariables(
            log_probs=log_probs,
            values=value.squeeze(-1),
        )

        next_carry = jax.tree.map(
            lambda x, y: jnp.where(transition.done, x, y),
            self.get_initial_model_carry(model, rng),
            Carry(
                actor_carry=next_actor_carry,
                critic_carry=next_critic_carry,
                lpf_params=lpf_params,
            ),
        )

        return next_carry, transition_ppo_variables

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: Carry,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, Carry]:
        scan_fn = functools.partial(self._model_scan_fn, model=model)
        next_model_carry, ppo_variables = xax.scan(
            scan_fn,
            model_carry,
            (trajectory, jax.random.split(rng, len(trajectory.done))),
            jit_level=4,
        )
        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, model: Model, rng: PRNGKeyArray) -> Carry:
        return Carry(
            actor_carry=jnp.zeros(shape=(2, self.config.depth, self.config.hidden_size)),
            critic_carry=jnp.zeros(shape=(2, self.config.depth, self.config.hidden_size)),
            lpf_params=ksim.LowPassFilterParams.initialize(len(ZEROS)),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: Carry,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        curriculum_level: Array,
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        action_dist_j, actor_carry, lpf_params = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=model_carry.actor_carry,
            curriculum_level=curriculum_level,
            lpf_params=model_carry.lpf_params,
        )
        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(key=rng)
        return ksim.Action(
            action=action_j,
            carry=Carry(
                actor_carry=actor_carry,
                critic_carry=model_carry.critic_carry,
                lpf_params=lpf_params,
            ),
        )


if __name__ == "__main__":
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=4096,
            batch_size=512,
            num_passes=4,
            rollout_length_frames=24,
            # Simulation parameters.
            dt=0.004,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            action_latency_range=(0.001, 0.01),  # Simulate 1-10ms of latency.
            drop_action_prob=0.05,  # Drop 5% of commands.
            # Visualization parameters.
            # If running this on Mac and you are getting segaults,
            # you might need to disable `render_markers`
            render_markers=True,
            render_track_body_id=0,
            disable_multiprocessing=True,
        ),
    )
