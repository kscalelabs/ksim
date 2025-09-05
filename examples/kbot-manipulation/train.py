"""Bimanual reaching task on a welded tabletop torso.

Changes vs your file:
- Adds a 6D command (xyz per wrist) with piecewise-constant resampling.
- Observations now include end-effector positions and the 6D target.
- Reward = -||ee - target||^2 for both hands (plus small torque penalty).
- Removes locomotion events; keeps simple resets; no-op curriculum.
"""

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
import attrs
from ksim.vis import Marker
from ksim.utils.mujoco import get_joint_names_in_order

import ksim

# ---------------------------------------------------------------------------
# End-effector body names from your MJCF
RIGHT_EE_BODY = "PRT0001"
LEFT_EE_BODY = "PRT0001_2"
# ---------------------------------------------------------------------------

# These are in the order of the neural network outputs.
ZEROS: list[tuple[str, float]] = [
    # Right arm
    ("dof_right_shoulder_pitch_03", 0.0),
    ("dof_right_shoulder_roll_03", math.radians(-10.0)),
    ("dof_right_shoulder_yaw_02", 0.0),
    ("dof_right_elbow_02", math.radians(45.0)),
    ("dof_right_wrist_00", 0.0),
    # Left arm
    ("dof_left_shoulder_pitch_03", 0.0),
    ("dof_left_shoulder_roll_03", math.radians(10.0)),
    ("dof_left_shoulder_yaw_02", 0.0),
    ("dof_left_elbow_02", math.radians(-45.0)),
    ("dof_left_wrist_00", 0.0),
]


@attrs.define(frozen=True, kw_only=True)
class FixedPeriodJointTargetCommand(ksim.Command):
    """Piecewise-constant joint targets for specific joints, resampled every period_s seconds.

    State encoding: [targets..., elapsed_time]
    """

    joint_indices: tuple[int, ...] = attrs.field()
    ranges: tuple[tuple[float, float], ...] = attrs.field()
    dt: float = attrs.field()
    period_s: float = attrs.field()

    def _sample_targets(self, rng: PRNGKeyArray) -> Array:
        ranges = jnp.array(self.ranges)  # (N, 2)
        return jax.random.uniform(rng, (ranges.shape[0],), minval=ranges[:, 0], maxval=ranges[:, 1])

    def initial_command(
        self,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        targets = self._sample_targets(rng)
        t = jnp.array(0.0)
        return jnp.concatenate([targets, jnp.array([t])])

    def __call__(
        self,
        prev_command: Array,
        physics_data: ksim.PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        n = len(self.joint_indices)
        targets = prev_command[..., :n]
        t = prev_command[..., n]
        t_new = t + self.dt
        resample = t_new >= self.period_s
        new_targets = self._sample_targets(rng)
        targets_next = jnp.where(resample, new_targets, targets)
        t_next = jnp.where(resample, jnp.array(0.0), t_new)
        return jnp.concatenate([targets_next, jnp.array([t_next])])

    def get_markers(self, name: str) -> list[Marker]:
        # No markers for joint-space targets.
        return []




@dataclass
class HumanoidManipulationTaskConfig(ksim.PPOConfig):
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
        value=2.0,
        help="The cutoff frequency for the low-pass filter.",
    )
    mujoco_scene: str = xax.field(
        value="smooth",
        help="The MuJoCo scene to use.",
    )

    # Command timing for left-arm joint targets
    left_arm_target_period_s: float = xax.field(
        value=1.0,
        help="Seconds between resampling left-arm joint targets.",
    )

    # Local model override
    local_model_dir: str | None = xax.field(
        value=None,
        help="Path to local robot directory containing metadata.json and *.mjcf/*.xml to bypass K API",
    )


    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    grad_clip: float = xax.field(
        value=2.0,
        help="Gradient clipping for the optimizer.",
    )
    warmup_steps: int = xax.field(
        value=100,
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
    """Actor for the reaching task: outputs desired joint positions (same as before)."""

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
    fc_scale: ksim.Scale = eqx.field()

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
        fc_scale: ksim.Scale,
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
        self.fc_scale = fc_scale

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
        # carry shape: (depth, hidden_size)
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
        fc = self.fc_scale.get_scale(curriculum_level)
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
        # carry shape: (depth, hidden_size)
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
        fc_scale: ksim.Scale,
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
            fc_scale=fc_scale,
        )
        self.critic = Critic(
            critic_key,
            hidden_size=hidden_size,
            depth=depth,
            num_inputs=num_critic_inputs,
            num_hidden_layers=num_hidden_layers,
        )


class HumanoidManipulationTask(ksim.PPOTask[HumanoidManipulationTaskConfig]):
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
        model_name_or_dir = self.config.local_model_dir or "kbot-headless"
        mjcf_path = asyncio.run(ksim.get_mujoco_model_path(model_name_or_dir, name="robot"))
        return mujoco_scenes.mjcf.load_mjmodel(mjcf_path, scene="smooth")

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        model_name_or_dir = self.config.local_model_dir or "kbot-headless"
        metadata = asyncio.run(ksim.get_mujoco_model_metadata(model_name_or_dir))
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
            torque_noise=ksim.AdditiveGaussianNoise(std=0.01),
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
                scale_lower=1.0,
                scale_upper=1.0,
            ),
            "joint_damping": ksim.JointDampingRandomizer(),
            "joint_zero_position": ksim.JointZeroPositionRandomizer(
                scale_lower=math.radians(-2),
                scale_upper=math.radians(2),
            ),
        }

    def get_events(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Event]:
        return {
            # "linear_push": ksim.LinearPushEvent(
            #     linvel=1.0,
            #     vel_range=(0.0, 1.0),
            #     interval_range=(1.0, 2.0),
            #     scale=ksim.QuadraticScale.from_endpoints(0.1, 1.0),
            # ),
            # "angular_push": ksim.AngularPushEvent(
            #     angvel=math.radians(90.0),
            #     vel_range=(0.0, 1.0),
            #     interval_range=(1.0, 2.0),
            #     scale=ksim.QuadraticScale.from_endpoints(0.1, 1.0),
            # ),
            # "jump": ksim.JumpEvent(
            #     jump_height_range=(0.1, 0.3),
            #     interval_range=(1.0, 2.0),
            #     scale=ksim.QuadraticScale.from_endpoints(0.1, 1.0),
            # ),
        }

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset.create(physics_model, {k: v for k, v in ZEROS}, scale=0.1),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Observation]:
        obs: dict[str, ksim.Observation] = {
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
        }

        # End-effector positions (left first, then right) => shape (.., 6)
        obs["ee_position"] = ksim.BodyPositionObservation.create(
            physics_model=physics_model,
            body_names=(LEFT_EE_BODY, RIGHT_EE_BODY),
            noise=ksim.AdditiveGaussianNoise(std=0.005),
        )
        return obs
    
    def get_commands(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Command]:
        # Command: piecewise-constant joint targets for the left arm; resample every fixed period.
        left_joint_names = (
            # "dof_left_shoulder_pitch_03",
            # "dof_left_shoulder_roll_03",
            # "dof_left_shoulder_yaw_02",
            # "dof_left_elbow_02",
            # "dof_left_wrist_00",
            "dof_right_elbow_02",
        )

        all_names = tuple(get_joint_names_in_order(physics_model))
        name_to_idx = {name: idx for idx, name in enumerate(all_names)}
        left_joint_indices = tuple(int(name_to_idx[n]) for n in left_joint_names)

        # Build ranges in the same order as indices
        all_ranges = physics_model.jnt_range.tolist()
        ranges_list = [(float(minv), float(maxv)) for (minv, maxv) in all_ranges]
        joint_ranges = tuple(ranges_list[name_to_idx[n]] for n in left_joint_names)

        return {
            "left_arm_joint_targets": FixedPeriodJointTargetCommand(
                joint_indices=left_joint_indices,
                ranges=joint_ranges,
                dt=self.config.ctrl_dt,
                period_s=self.config.left_arm_target_period_s,
            ),
        }

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Reward]:
        rewards: dict[str, ksim.Reward] = {}
        rewards["stay_alive"] = ksim.StayAliveReward(scale=100.0)

        # Reward: track the commanded left arm joint targets.
        left_joint_names = (
            # "dof_left_shoulder_pitch_03",
            # "dof_left_shoulder_roll_03",
            # "dof_left_shoulder_yaw_02",
            # "dof_left_elbow_02",
            # "dof_left_wrist_00",
            "dof_right_elbow_02",
        )

        # Precompute indices to avoid capturing the model in the reward.
        all_names = tuple(get_joint_names_in_order(physics_model))
        name_to_idx = {name: idx for idx, name in enumerate(all_names)}
        left_joint_indices = tuple(int(name_to_idx[n]) for n in left_joint_names)

        @attrs.define(frozen=True, kw_only=True)
        class TrackLeftArmToCommandReward(ksim.Reward):
            joint_indices: tuple[int, ...] = attrs.field()

            def get_reward(self, trajectory: ksim.Trajectory) -> Array:
                idxs = jnp.array(self.joint_indices) + 7
                qpos_sel = trajectory.qpos[..., idxs]

                # Command target vector for these joints is the first N entries
                cmd_state = trajectory.command["left_arm_joint_targets"]
                n = qpos_sel.shape[-1]
                target = cmd_state[..., :n]

                diff = qpos_sel - target
                return -jnp.sum(jnp.square(diff), axis=-1)

        rewards["track_left_arm_to_command"] = TrackLeftArmToCommandReward(
            joint_indices=left_joint_indices,
            scale=1.0,
        )

        return rewards

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> dict[str, ksim.Termination]:
        return {
            "far_from_origin": ksim.FarFromOriginTermination(max_dist=5.0),
        }

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.ConstantCurriculum(level=1.0)

    def get_model(self, params: ksim.InitParams) -> Model:
        pm = params.physics_model
        num_joints = pm.nq - 7  # exclude floating base (pos 3 + quat 4); welded, but model still has freejoint in file.
        nu = pm.nu
        nbody = pm.nbody

        # Actor obs: joint_pos (N) + joint_vel (N) + proj_grav (3) + imu_gyro (3)
        num_actor_inputs = (2 * num_joints) + 3 + 3

        # Critic obs matches run_critic concatenation (no EE/target, no commands)
        num_critic_inputs = (
            (2 * num_joints)
            + ((nbody - 1) * 10)
            + ((nbody - 1) * 6)
            + 3  # imu_acc
            + 3  # imu_gyro
            + 3  # projected_gravity
            + nu  # actuator_force
            + 3  # base_pos (we skip; welded; but if your collector adds it, harmless)
            + 4  # base_quat
            + 3  # lin_vel
            + 3  # ang_vel
        )

        return Model(
            params.key,
            physics_model=params.physics_model,
            num_actor_inputs=num_actor_inputs,
            num_actor_outputs=len(ZEROS),
            num_critic_inputs=num_critic_inputs,
            min_std=0.0001,
            max_std=1.0,
            var_scale=self.config.var_scale,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_hidden_layers=self.config.num_hidden_layers,
            ctrl_dt=self.config.ctrl_dt,
            fc_scale=ksim.QuadraticScale.from_endpoints(
                start=self.config.start_cutoff_frequency,
                end=self.config.end_cutoff_frequency,
            ),
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


        obs_n = jnp.concatenate(
            [
                joint_pos_n,            # N
                joint_vel_n / 10.0,     # N
                proj_grav_3,            # 3
                imu_gyro_3,             # 3
            ],
            axis=-1,
        )
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


        obs_n = jnp.concatenate(
            [
                dh_joint_pos_j,                 # N
                dh_joint_vel_j / 10.0,         # N
                com_inertia_n,
                com_vel_n,
                imu_acc_3,
                imu_gyro_3,
                proj_grav_3,
                act_frc_obs_n / 100.0,         # scale
                base_pos_3,
                base_quat_4,
                lin_vel_obs_3,
                ang_vel_obs_3,
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
    HumanoidManipulationTask.launch(
        HumanoidManipulationTaskConfig(
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
            drop_action_prob=0.05,               # Drop 5% of commands.
            # Visualization parameters.
            render_markers=True,
            render_track_body_id=0,
            disable_multiprocessing=True,
            render_azimuth=135,
            render_distance=1.5,
            valid_every_n_steps=5,
            log_all_images=True,
            max_values_per_plot=50,
        ),
    )
