"""Defines simple task for training a walking policy for the default humanoid."""

import functools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypedDict, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

import ksim

ZEROS = [
    ("abdomen_z", 0.0),
    ("abdomen_y", 0.0),
    ("abdomen_x", 0.0),
    ("hip_x_right", 0.0),
    ("hip_z_right", 0.0),
    ("hip_y_right", math.radians(-25.0)),
    ("knee_right", math.radians(-50.0)),
    ("hip_x_left", 0.0),
    ("hip_z_left", 0.0),
    ("hip_y_left", math.radians(-25.0)),
    ("knee_left", math.radians(-50.0)),
    ("shoulder1_right", 0.0),
    ("shoulder2_right", 0.0),
    ("elbow_right", 0.0),
    ("shoulder1_left", 0.0),
    ("shoulder2_left", 0.0),
    ("elbow_left", 0.0),
]


class Actor(eqx.Module):
    """RNN-based actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.MLP
    clip_positions: ksim.ClipPositions
    num_joints: int = eqx.field()
    num_inputs: int = eqx.field()
    num_outputs: int = eqx.field()
    min_std: float = eqx.field()
    max_std: float = eqx.field()
    var_scale: float = eqx.field()
    num_mixtures: int = eqx.field()
    ctrl_dt: float = eqx.field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        physics_model: ksim.PhysicsModel,
        num_inputs: int,
        num_outputs: int,
        num_joints: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
        num_hidden_layers: int,
        num_mixtures: int,
        ctrl_dt: float,
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
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
            ]
        )

        # Project to output
        self.output_proj = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=num_outputs * 3 * num_mixtures,
            width_size=hidden_size,
            depth=num_hidden_layers,
            activation=jax.nn.gelu,
            use_final_bias=False,
            key=key,
        )

        self.num_joints = num_joints
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.clip_positions = ksim.ClipPositions.from_physics_model(physics_model)
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.num_mixtures = num_mixtures
        self.ctrl_dt = ctrl_dt

    def forward(
        self,
        obs_n: Array,
        carry: Array,
        filter_params: ksim.ClipAccelerationParams,
    ) -> tuple[xax.Distribution, Array, ksim.ClipAccelerationParams]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        prediction_n = self.output_proj(x_n)

        # Splits the predictions into means, standard deviations, and logits.
        slice_len = self.num_joints * self.num_mixtures
        mean_nm = prediction_n[:slice_len].reshape(self.num_joints, self.num_mixtures)
        std_nm = prediction_n[slice_len : 2 * slice_len].reshape(self.num_joints, self.num_mixtures)
        logits_nm = prediction_n[2 * slice_len :].reshape(self.num_joints, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip((jax.nn.softplus(std_nm) + self.min_std) * self.var_scale, max=self.max_std)

        # Adds zero bias.
        mean_nm = mean_nm + jnp.array([v for _, v in ZEROS])[:, None]

        # Clip the target positions to the minimum and maximum ranges.
        mean_nm = jax.vmap(self.clip_positions.clip, in_axes=-1, out_axes=-1)(mean_nm)

        dist_n = xax.MixtureOfGaussians(mean_nm, std_nm, logits_nm)

        return dist_n, jnp.stack(out_carries, axis=0), filter_params


class Critic(eqx.Module):
    """RNN-based critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnns: tuple[eqx.nn.GRUCell, ...]
    output_proj: eqx.nn.MLP

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

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        rnn_keys = jax.random.split(rnn_key, depth)
        self.rnns = tuple(
            [
                eqx.nn.GRUCell(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    key=rnn_key,
                )
                for rnn_key in rnn_keys
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

    def forward(self, obs_n: Array, carry: Array) -> tuple[Array, Array]:
        x_n = self.input_proj(obs_n)
        out_carries = []
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, carry[i])
            out_carries.append(x_n)
        out_n = self.output_proj(x_n)

        return out_n, jnp.stack(out_carries, axis=0)


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        physics_model: ksim.PhysicsModel,
        min_std: float,
        max_std: float,
        var_scale: float,
        num_actor_inputs: int,
        num_critic_inputs: int,
        num_joints: int,
        hidden_size: int,
        depth: int,
        num_hidden_layers: int,
        num_mixtures: int,
        ctrl_dt: float,
    ) -> None:
        self.actor = Actor(
            key,
            physics_model=physics_model,
            num_inputs=num_actor_inputs,
            num_outputs=num_joints,
            num_joints=num_joints,
            min_std=min_std,
            max_std=max_std,
            var_scale=var_scale,
            hidden_size=hidden_size,
            depth=depth,
            num_hidden_layers=num_hidden_layers,
            num_mixtures=num_mixtures,
            ctrl_dt=ctrl_dt,
        )
        self.critic = Critic(
            key,
            num_inputs=num_critic_inputs,
            hidden_size=hidden_size,
            depth=depth,
            num_hidden_layers=num_hidden_layers,
        )


@dataclass
class WalkingConfig(ksim.PPOConfig):
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
    num_mixtures: int = xax.field(
        value=10,
        help="The number of mixtures for the actor.",
    )

    # Reward parameters.
    linear_velocity_range: tuple[float, float] = xax.field(
        value=(1.0, 3.0),
        help="The range for the linear velocity command.",
    )
    linear_velocity_max_yaw: float = xax.field(
        value=math.pi / 4.0,
        help="The maximum yaw for the linear velocity command.",
    )
    linear_velocity_zero_prob: float = xax.field(
        value=0.2,
        help="The probability of the linear velocity command being zero.",
    )
    linear_velocity_switch_prob: float = xax.field(
        value=0.001,
        help="The probability of the linear velocity command being switched.",
    )
    angular_velocity_range: tuple[float, float] = xax.field(
        value=(-0.2, 0.2),
        help="The range for the angular velocity command.",
    )
    angular_velocity_zero_prob: float = xax.field(
        value=0.2,
        help="The probability of the angular velocity command being zero.",
    )
    angular_velocity_switch_prob: float = xax.field(
        value=0.001,
        help="The probability of the angular velocity command being switched.",
    )
    gait_period: float = xax.field(
        value=0.6,
        help="The period for the sinusoidal gait command.",
    )
    max_foot_height: float = xax.field(
        value=0.2,
        help="The maximum height for the sinusoidal gait command.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    warmup_steps: int = xax.field(
        value=100,
        help="The number of steps to warm up the learning rate.",
    )
    adam_weight_decay: float = xax.field(
        value=1e-5,
        help="Weight decay for the Adam optimizer.",
    )
    grad_clip: float = xax.field(
        value=2.0,
        help="Gradient clip for the Adam optimizer.",
    )

    # Curriculum parameters.
    curriculum_step_size: float = xax.field(
        value=0.01,
        help="The step size for the curriculum.",
    )
    curriculum_step_every_n_epochs: int = xax.field(
        value=100,
        help="The number of steps to take before updating the curriculum.",
    )
    curriculum_delay_steps: int = xax.field(
        value=5000,
        help="The number of steps to delay the curriculum.",
    )

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=0,
        help="The body id to track with the render camera.",
    )


class Carry(TypedDict):
    actor: Array
    critic: Array
    filter_params: ksim.ClipAccelerationParams


Config = TypeVar("Config", bound=WalkingConfig)


class WalkingTask(ksim.PPOTask[Config], Generic[Config]):
    def get_optimizer(self) -> optax.GradientTransformation:
        scheduler = optax.warmup_constant_schedule(
            init_value=self.config.learning_rate * 0.01,
            peak_value=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
        )

        return optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.add_decayed_weights(self.config.adam_weight_decay),
            optax.scale_by_adam(),
            optax.scale_by_schedule(scheduler),
            optax.scale(-1.0),
        )

    def get_mujoco_model(self) -> mujoco.MjModel:  # pyright: ignore[reportAttributeAccessIssue]
        mjcf_path = (Path(__file__).parent / "data" / "scene.mjcf").resolve().as_posix()
        return mujoco.MjModel.from_xml_path(mjcf_path)  # pyright: ignore[reportAttributeAccessIssue]

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:  # pyright: ignore[reportAttributeAccessIssue]
        return ksim.Metadata.from_model(
            mj_model,
            kp=50.0,
            kd=1.0,
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

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.PhysicsRandomizer]:
        return {
            "static_friction": ksim.StaticFrictionRandomizer(),
            "armature": ksim.ArmatureRandomizer(),
            "mass_multiplication": ksim.MassMultiplicationRandomizer.from_body_name(physics_model, "torso"),
            "joint_damping": ksim.JointDampingRandomizer(),
            "joint_zero_position": ksim.JointZeroPositionRandomizer(),
        }

    def get_events(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Event]:
        return {
            "push": ksim.LinearPushEvent(
                linvel=1.0,
                interval_range=(2.0, 5.0),
            ),
            "jump": ksim.JumpEvent(
                jump_height_range=(0.1, 0.5),
                interval_range=(2.0, 5.0),
            ),
        }

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, zeros={"abdomen_z": 0.0}),
            ksim.RandomJointVelocityReset(),
            ksim.RandomHeadingReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Observation]:
        return {
            "joint_position": ksim.JointPositionObservation(
                noise=ksim.AdditiveUniformNoise(mag=0.1),
            ),
            "joint_velocity": ksim.JointVelocityObservation(
                noise=ksim.MultiplicativeUniformNoise(mag=0.3),
            ),
            "actuator_force": ksim.ActuatorForceObservation(),
            "center_of_mass_inertia": ksim.CenterOfMassInertiaObservation(),
            "center_of_mass_velocity": ksim.CenterOfMassVelocityObservation(),
            "base_position": ksim.BasePositionObservation(),
            "base_orientation": ksim.BaseOrientationObservation(),
            "base_linear_velocity": ksim.BaseLinearVelocityObservation(),
            "base_angular_velocity": ksim.BaseAngularVelocityObservation(),
            "imu_projected_gravity": ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="orientation",
                noise=ksim.AdditiveGaussianNoise(std=0.01),
                min_lag=0.001,
                max_lag=0.005,
                bias=math.radians(2.0),
            ),
            "projected_gravity": ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="orientation",
            ),
            "imu_acc": ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_acc",
                noise=ksim.AdditiveGaussianNoise(std=0.01),
            ),
            "imu_gyro": ksim.SensorObservation.create(
                physics_model=physics_model,
                sensor_name="imu_gyro",
                noise=ksim.AdditiveGaussianNoise(std=0.01),
            ),
            "feet_contact": ksim.FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names=["foot_left"],
                foot_right_geom_names=["foot_right"],
                floor_geom_names=["floor"],
            ),
            "feet_position": ksim.BodyPositionObservation.create(
                physics_model=physics_model,
                body_names=("foot_left", "foot_right"),
            ),
            "feet_force": ksim.FeetForceObservation.create(
                physics_model=physics_model,
                foot_left_body_name="foot_left",
                foot_right_body_name="foot_right",
            ),
        }

    def get_commands(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Command]:
        return {
            "linvel": ksim.LinearVelocityCommand(
                min_vel=self.config.linear_velocity_range[0],
                max_vel=self.config.linear_velocity_range[1],
                max_yaw=self.config.linear_velocity_max_yaw,
                zero_prob=self.config.linear_velocity_zero_prob,
                switch_prob=self.config.linear_velocity_switch_prob,
            ),
            "angvel": ksim.AngularVelocityCommand(
                min_vel=self.config.angular_velocity_range[0],
                max_vel=self.config.angular_velocity_range[1],
                zero_prob=self.config.angular_velocity_zero_prob,
                switch_prob=self.config.angular_velocity_switch_prob,
            ),
        }

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Reward]:
        return {
            "stay_alive": ksim.StayAliveReward(scale=100.0),
            "upright": ksim.UprightReward(scale=5.0),
            "linvel": ksim.LinearVelocityReward(
                cmd="linvel",
                scale=0.1,
            ),
            "angvel": ksim.AngularVelocityReward(
                cmd="angvel",
                scale=0.01,
            ),
            "foot_airtime": ksim.FeetAirTimeReward(
                ctrl_dt=self.config.ctrl_dt,
                max_air_time=self.config.gait_period * 0.4,
                max_ground_time=self.config.gait_period * 0.6,
                contact_obs="feet_contact",
                scale=1.0,
            ),
            "foot_height": ksim.SparseTargetHeightReward(
                contact_obs="feet_contact",
                position_obs="feet_position",
                height=self.config.max_foot_height,
                scale=1.0,
            ),
            "foot_force": ksim.ForcePenalty(
                force_obs="feet_force",
                ctrl_dt=self.config.ctrl_dt,
                bias=100.0,
                scale=-1e-1,
            ),
            "motionless_at_rest": ksim.MotionlessAtRestPenalty(scale=-1e-2),
        }

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Termination]:
        return {
            "bad_z": ksim.BadZTermination(min_z=0.5, max_z=3.0),
            "bad_velocity": ksim.BadVelocityTermination(max_vel=100.0),
            "far_from_origin": ksim.FarFromOriginTermination(max_dist=10.0),
        }

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.LinearCurriculum(
            step_size=self.config.curriculum_step_size,
            step_every_n_epochs=self.config.curriculum_step_every_n_epochs,
            delay_steps=self.config.curriculum_delay_steps,
        )

    def get_model(self, params: ksim.InitParams) -> Model:
        return Model(
            params.key,
            physics_model=params.physics_model,
            num_actor_inputs=43,
            num_critic_inputs=340,
            num_joints=17,
            min_std=0.01,
            max_std=1.0,
            var_scale=0.5,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_hidden_layers=self.config.num_hidden_layers,
            num_mixtures=self.config.num_mixtures,
            ctrl_dt=self.config.ctrl_dt,
        )

    def get_initial_model_carry(
        self,
        model: Model,
        rng: PRNGKeyArray,
    ) -> Carry:
        return {
            "actor": jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            "critic": jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            "filter_params": ksim.ClipAccelerationParams.initialize_from(jnp.array([v for _, v in ZEROS])),
        }

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, PyTree],
        commands: xax.FrozenDict[str, PyTree],
        carry: Array,
        filter_params: ksim.ClipAccelerationParams,
    ) -> tuple[xax.Distribution, Array, ksim.ClipAccelerationParams]:
        dh_joint_pos_j = observations["noisy_joint_position"]
        dh_joint_vel_j = observations["noisy_joint_velocity"]
        proj_grav_3 = observations["noisy_imu_projected_gravity"]
        imu_gyro_3 = observations["noisy_imu_gyro"]

        # Command tensors.
        linvel_cmd: ksim.LinearVelocityCommandValue = commands["linvel"]
        angvel_cmd: ksim.AngularVelocityCommandValue = commands["angvel"]

        # Stacks into tensors.
        linvel_cmd_2 = jnp.stack([linvel_cmd.vel, linvel_cmd.yaw], axis=-1)
        angvel_cmd_1 = jnp.stack([angvel_cmd.vel], axis=-1)

        obs_n = jnp.concatenate(
            [
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                proj_grav_3,  # 3
                imu_gyro_3,  # 3
                linvel_cmd_2,  # 2
                angvel_cmd_1,  # 1
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry, filter_params)

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
        feet_force_obs_23 = observations["feet_force"]

        # Flattens the last two dimensions.
        feet_force_obs_6 = feet_force_obs_23.reshape(*feet_force_obs_23.shape[:-2], 6)

        # Command tensors.
        linvel_cmd: ksim.LinearVelocityCommandValue = commands["linvel"]
        angvel_cmd: ksim.AngularVelocityCommandValue = commands["angvel"]

        # Stacks into tensors.
        linvel_cmd_4 = jnp.stack([linvel_cmd.vel, linvel_cmd.yaw, linvel_cmd.xvel, linvel_cmd.yvel], axis=-1)
        angvel_cmd_1 = jnp.stack([angvel_cmd.vel], axis=-1)

        obs_n = jnp.concatenate(
            [
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
                feet_force_obs_6,  # 6
                linvel_cmd_4,  # 4
                angvel_cmd_1,  # 1
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def _scan_fn(
        self,
        actor_critic_carry: Carry,
        xs: tuple[ksim.Trajectory, PRNGKeyArray],
        model: Model,
    ) -> tuple[Carry, ksim.PPOVariables]:
        transition, rng = xs
        actor_carry = actor_critic_carry["actor"]
        critic_carry = actor_critic_carry["critic"]
        filter_params = actor_critic_carry["filter_params"]
        actor_dist, next_actor_carry, filter_params = self.run_actor(
            model=model.actor,
            observations=transition.obs,
            commands=transition.command,
            carry=actor_carry,
            filter_params=filter_params,
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

        next_carry = jax.tree.map(
            lambda x, y: jnp.where(transition.done, x, y),
            self.get_initial_model_carry(model, rng),
            {"actor": next_actor_carry, "critic": next_critic_carry, "filter_params": filter_params},
        )

        return next_carry, transition_ppo_variables

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: Carry,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, Carry]:
        scan_fn = functools.partial(self._scan_fn, model=model)
        rngs = jax.random.split(rng, trajectory.done.shape[0])
        next_model_carry, ppo_variables = xax.scan(
            scan_fn,
            model_carry,
            (trajectory, rngs),
            jit_level=ksim.JitLevel.RL_CORE,
        )
        return ppo_variables, next_model_carry

    def sample_action(
        self,
        model: Model,
        model_carry: Carry,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, PyTree],
        commands: xax.FrozenDict[str, PyTree],
        curriculum_level: Array,
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in = model_carry["actor"]
        critic_carry_in = model_carry["critic"]
        filter_params = model_carry["filter_params"]

        # Runs the actor model to get the action distribution.
        action_dist_j, actor_carry, filter_params = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
            filter_params=filter_params,
        )

        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(rng)

        return ksim.Action(
            action=action_j,
            carry={"actor": actor_carry, "critic": critic_carry_in, "filter_params": filter_params},
        )


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.walking
    # To visualize the environment, use the following command:
    #   python -m examples.walking run_mode=view
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m examples.walking num_envs=8 batch_size=4
    WalkingTask.launch(
        WalkingConfig(
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
        ),
    )
