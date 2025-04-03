"""Defines simple task for training a walking policy for the default humanoid."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import optax
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx

import ksim

NUM_JOINTS = 21

NUM_INPUTS = NUM_JOINTS + NUM_JOINTS + 160 + 96 + 3 + 3 + NUM_JOINTS + 3 + 4 + 3 + 3 + 1 + 1 + 1

ACTION_RANGES = [
    [-0.7853981633974483, 0.7853981633974483],
    [-1.3089969389957472, 0.5235987755982988],
    [-0.6108652381980153, 0.6108652381980153],
    [-0.5235987755982988, 0.17453292519943295],
    [-1.0471975511965976, 0.6108652381980153],
    [-2.6179938779914944, 0.3490658503988659],
    [-2.792526803190927, 0.03490658503988659],
    [-0.8726646259971648, 0.8726646259971648],
    [-0.8726646259971648, 0.8726646259971648],
    [-0.5235987755982988, 0.17453292519943295],
    [-1.0471975511965976, 0.6108652381980153],
    [-2.6179938779914944, 0.3490658503988659],
    [-2.792526803190927, 0.03490658503988659],
    [-0.8726646259971648, 0.8726646259971648],
    [-0.8726646259971648, 0.8726646259971648],
    [-1.4835298641951802, 1.0471975511965976],
    [-1.4835298641951802, 1.0471975511965976],
    [-1.7453292519943295, 0.8726646259971648],
    [-1.4835298641951802, 1.0471975511965976],
    [-1.4835298641951802, 1.0471975511965976],
    [-1.7453292519943295, 0.8726646259971648],
]


def map_normal_distribution(dist: distrax.Distribution) -> distrax.Distribution:
    action_ranges = jnp.array(ACTION_RANGES)
    action_min, action_max = action_ranges[..., 0], action_ranges[..., 1]
    dist = distrax.Transformed(dist, distrax.Tanh())
    dist = distrax.Transformed(dist, ksim.DoubleUnitIntervalToRangeBijector(min=action_min, max=action_max))
    return dist


@attrs.define(frozen=True, kw_only=True)
class NaiveForwardReward(ksim.Reward):
    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return trajectory.qvel[..., 0].clip(max=5.0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array


class DefaultHumanoidActor(eqx.Module):
    """Actor for the walking task."""

    mlp: eqx.nn.MLP
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_inputs = NUM_INPUTS
        num_outputs = NUM_JOINTS

        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=num_outputs * 2,
            width_size=hidden_size,
            depth=depth,
            key=key,
            activation=jax.nn.relu,
        )
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(
        self,
        dh_joint_pos_n: Array,
        dh_joint_vel_n: Array,
        com_inertia_n: Array,
        com_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        act_frc_obs_n: Array,
        base_pos_3: Array,
        base_quat_4: Array,
        lin_vel_obs_3: Array,
        ang_vel_obs_3: Array,
        lin_vel_cmd_x_1: Array,
        lin_vel_cmd_y_1: Array,
        ang_vel_cmd_z_1: Array,
    ) -> distrax.Distribution:
        obs_n = jnp.concatenate(
            [
                dh_joint_pos_n,  # NUM_JOINTS
                dh_joint_vel_n,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                act_frc_obs_n,  # 21
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
                lin_vel_cmd_x_1,  # 1
                lin_vel_cmd_y_1,  # 1
                ang_vel_cmd_z_1,  # 1
            ],
            axis=-1,
        )

        prediction_n = self.mlp(obs_n)
        mean_n = prediction_n[..., :NUM_JOINTS]
        std_n = prediction_n[..., NUM_JOINTS:]

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        dist = distrax.Normal(mean_n, std_n)
        dist = map_normal_distribution(dist)
        return dist


class DefaultHumanoidCritic(eqx.Module):
    """Critic for the walking task."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_inputs = NUM_INPUTS
        num_outputs = 1

        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=num_outputs,
            width_size=hidden_size,
            depth=depth,
            key=key,
            activation=jax.nn.relu,
        )

    def forward(
        self,
        dh_joint_pos_n: Array,
        dh_joint_vel_n: Array,
        com_inertia_n: Array,
        com_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        act_frc_obs_n: Array,
        base_pos_3: Array,
        base_quat_4: Array,
        lin_vel_obs_3: Array,
        ang_vel_obs_3: Array,
        lin_vel_cmd_x_1: Array,
        lin_vel_cmd_y_1: Array,
        ang_vel_cmd_z_1: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [
                dh_joint_pos_n,  # NUM_JOINTS
                dh_joint_vel_n,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                act_frc_obs_n,  # 21
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
                lin_vel_cmd_x_1,  # 1
                lin_vel_cmd_y_1,  # 1
                ang_vel_cmd_z_1,  # 1
            ],
            axis=-1,
        )
        return self.mlp(x_n)


class DefaultHumanoidModel(eqx.Module):
    actor: DefaultHumanoidActor
    critic: DefaultHumanoidCritic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
    ) -> None:
        self.actor = DefaultHumanoidActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=0.5,
            hidden_size=hidden_size,
            depth=depth,
        )
        self.critic = DefaultHumanoidCritic(
            key,
            hidden_size=hidden_size,
            depth=depth,
        )


@dataclass
class HumanoidWalkingTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

    # Model parameters.
    hidden_size: int = xax.field(
        value=256,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=5,
        help="The depth for the MLPs.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=3e-4,
        help="Learning rate for PPO.",
    )
    max_grad_norm: float = xax.field(
        value=2.0,
        help="Maximum gradient norm for clipping.",
    )
    adam_weight_decay: float = xax.field(
        value=0.0,
        help="Weight decay for the Adam optimizer.",
    )

    # Mujoco parameters.
    kp: float = xax.field(
        value=1.0,
        help="The Kp for the actuators",
    )
    kd: float = xax.field(
        value=0.1,
        help="The Kd for the actuators",
    )
    armature: float = xax.field(
        value=1e-2,
        help="A value representing the effective inertia of the actuator armature",
    )
    friction: float = xax.field(
        value=1e-6,
        help="The dynamic friction loss for the actuator",
    )

    # Curriculum parameters.
    num_curriculum_levels: int = xax.field(
        value=10,
        help="The number of curriculum levels to use.",
    )
    increase_threshold: float = xax.field(
        value=1.0,
        help="Increase the curriculum level when the deaths per episode is below this threshold.",
    )
    decrease_threshold: float = xax.field(
        value=5.0,
        help="Decrease the curriculum level when the deaths per episode is above this threshold.",
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

    # Checkpointing parameters.
    export_for_inference: bool = xax.field(
        value=False,
        help="Whether to export the model for inference.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingTaskConfig)


class HumanoidWalkingTask(ksim.PPOTask[Config], Generic[Config]):
    def get_optimizer(self) -> optax.GradientTransformation:
        """Builds the optimizer.

        This provides a reasonable default optimizer for training PPO models,
        but can be overridden by subclasses who want to do something different.
        """
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            (
                optax.adam(self.config.learning_rate)
                if self.config.adam_weight_decay == 0.0
                else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
            ),
        )

        return optimizer

    def get_mujoco_model(self) -> tuple[mujoco.MjModel, dict[str, JointMetadataOutput]]:
        mjcf_path = (Path(__file__).parent / "data" / "scene.mjcf").resolve().as_posix()
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)

        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 4
        mj_model.opt.ls_iterations = 8
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG

        # Observed NaNs in qpos with Newton solver...
        # mj_model.opt.solver = mjx.SolverType.NEWTON

        return mj_model

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> dict[str, JointMetadataOutput]:
        return ksim.get_joint_metadata(
            mj_model,
            kp=self.config.kp,
            kd=self.config.kd,
            armature=self.config.armature,
            friction=self.config.friction,
        )

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.MITPositionActuators(
            physics_model=physics_model,
            joint_name_to_metadata=metadata,
        )

    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.Randomization]:
        return [
            ksim.StaticFrictionRandomization(),
            ksim.ArmatureRandomization(),
            ksim.MassMultiplicationRandomization.from_body_name(physics_model, "torso"),
            ksim.JointDampingRandomization(),
            ksim.JointZeroPositionRandomization(),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(
                x_force=1.0,
                y_force=1.0,
                z_force=0.0,
                x_angular_force=0.1,
                y_angular_force=0.1,
                z_angular_force=0.3,
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
            ksim.ActuatorAccelerationObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro"),
            ksim.FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names=["foot1_left", "foot2_left"],
                foot_right_geom_names=["foot1_right", "foot2_right"],
                floor_geom_names=["floor"],
            ),
            ksim.FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_body_name="foot_left",
                foot_right_body_name="foot_right",
            ),
            ksim.FeetOrientationObservation.create(
                physics_model=physics_model,
                foot_left_body_name="foot_left",
                foot_right_body_name="foot_right",
            ),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        switch_prob = self.config.ctrl_dt / 5
        return [
            ksim.LinearVelocityCommand(index="x", range=(-1.0, 2.5), zero_prob=0.1, switch_prob=switch_prob),
            ksim.LinearVelocityCommand(index="y", range=(-0.3, 0.3), zero_prob=0.9, switch_prob=switch_prob),
            ksim.AngularVelocityCommand(index="z", scale=0.2, zero_prob=0.9, switch_prob=switch_prob),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.LinearVelocityTrackingReward(index="x", command_name="linear_velocity_command_x", scale=1.0),
            ksim.LinearVelocityTrackingReward(index="y", command_name="linear_velocity_command_y", scale=0.1),
            ksim.AngularVelocityTrackingReward(index="z", command_name="angular_velocity_command_z", scale=0.01),
            # NaiveForwardReward(scale=1.0),
            ksim.StayAliveReward(scale=1.0),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.9, unhealthy_z_upper=1.6),
            ksim.PitchTooGreatTermination(max_pitch=math.pi / 3),
            ksim.RollTooGreatTermination(max_roll=math.pi / 3),
            ksim.FastAccelerationTermination(),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.StepWhenSaturated(
            num_levels=self.config.num_curriculum_levels,
            increase_threshold=self.config.increase_threshold,
            decrease_threshold=self.config.decrease_threshold,
            min_level_steps=self.config.min_level_steps,
        )

    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidModel:
        return DefaultHumanoidModel(key, hidden_size=self.config.hidden_size, depth=self.config.depth)

    def get_initial_carry(self, rng: PRNGKeyArray) -> None:
        return None

    def _run_actor(
        self,
        model: DefaultHumanoidActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> distrax.Distribution:
        dh_joint_pos_n = observations["joint_position_observation"]  # 26
        dh_joint_vel_n = observations["joint_velocity_observation"]  # 27
        com_inertia_n = observations["center_of_mass_inertia_observation"]  # 160
        com_vel_n = observations["center_of_mass_velocity_observation"]  # 96
        imu_acc_3 = observations["sensor_observation_imu_acc"]  # 3
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]  # 3
        act_frc_obs_n = observations["actuator_force_observation"] / 100.0  # 21
        base_pos_3 = observations["base_position_observation"]  # 3
        base_quat_4 = observations["base_orientation_observation"]  # 4
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]  # 3
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]  # 3
        lin_vel_cmd_x_1 = commands["linear_velocity_command_x"]  # 1
        lin_vel_cmd_y_1 = commands["linear_velocity_command_y"]  # 1
        ang_vel_cmd_z_1 = commands["angular_velocity_command_z"]  # 1

        return model.forward(
            dh_joint_pos_n=dh_joint_pos_n,
            dh_joint_vel_n=dh_joint_vel_n / 10.0,
            com_inertia_n=com_inertia_n,
            com_vel_n=com_vel_n,
            imu_acc_3=imu_acc_3 / 50.0,
            imu_gyro_3=imu_gyro_3 / 3.0,
            act_frc_obs_n=act_frc_obs_n,
            base_pos_3=base_pos_3,
            base_quat_4=base_quat_4,
            lin_vel_obs_3=lin_vel_obs_3,
            ang_vel_obs_3=ang_vel_obs_3,
            lin_vel_cmd_x_1=lin_vel_cmd_x_1,
            lin_vel_cmd_y_1=lin_vel_cmd_y_1,
            ang_vel_cmd_z_1=ang_vel_cmd_z_1,
        )

    def _run_critic(
        self,
        model: DefaultHumanoidCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        dh_joint_pos_n = observations["joint_position_observation"]  # 26
        dh_joint_vel_n = observations["joint_velocity_observation"]  # 27
        com_inertia_n = observations["center_of_mass_inertia_observation"]  # 160
        com_vel_n = observations["center_of_mass_velocity_observation"]  # 96
        imu_acc_3 = observations["sensor_observation_imu_acc"]  # 3
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]  # 3
        act_frc_obs_n = observations["actuator_force_observation"] / 100.0  # 21
        base_pos_3 = observations["base_position_observation"]  # 3
        base_quat_4 = observations["base_orientation_observation"]  # 4
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]  # 3
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]  # 3
        lin_vel_cmd_x_1 = commands["linear_velocity_command_x"]  # 1
        lin_vel_cmd_y_1 = commands["linear_velocity_command_y"]  # 1
        ang_vel_cmd_z_1 = commands["angular_velocity_command_z"]  # 1

        return model.forward(
            dh_joint_pos_n=dh_joint_pos_n,
            dh_joint_vel_n=dh_joint_vel_n / 10.0,
            com_inertia_n=com_inertia_n,
            com_vel_n=com_vel_n,
            imu_acc_3=imu_acc_3 / 50.0,
            imu_gyro_3=imu_gyro_3 / 3.0,
            act_frc_obs_n=act_frc_obs_n,
            base_pos_3=base_pos_3,
            base_quat_4=base_quat_4,
            lin_vel_obs_3=lin_vel_obs_3,
            ang_vel_obs_3=ang_vel_obs_3,
            lin_vel_cmd_x_1=lin_vel_cmd_x_1,
            lin_vel_cmd_y_1=lin_vel_cmd_y_1,
            ang_vel_cmd_z_1=ang_vel_cmd_z_1,
        )

    def get_on_policy_variables(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> ksim.PPOVariables:
        # Use cached log probabilities from training.
        if not isinstance(trajectories.aux_outputs, AuxOutputs):
            raise ValueError("No aux outputs found in trajectories")

        # Compute the values online.
        par_critic_fn = jax.vmap(self._run_critic, in_axes=(None, 0, 0))
        values_t1 = par_critic_fn(model.critic, trajectories.obs, trajectories.command)

        return ksim.PPOVariables(
            log_probs_tn=trajectories.aux_outputs.log_probs,
            values_t=values_t1.squeeze(-1),
        )

    def get_off_policy_variables(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> ksim.PPOVariables:
        # Vectorize over the time dimensions.
        par_actor_fn = jax.vmap(self._run_actor, in_axes=(None, 0, 0))
        action_dist_tn = par_actor_fn(model.actor, trajectories.obs, trajectories.command)
        log_probs_tj = action_dist_tn.log_prob(trajectories.action)

        # Vectorize over the time dimensions.
        par_critic_fn = jax.vmap(self._run_critic, in_axes=(None, 0, 0))
        values_t1 = par_critic_fn(model.critic, trajectories.obs, trajectories.command)

        return ksim.PPOVariables(
            log_probs_tn=log_probs_tj,
            values_t=values_t1.squeeze(-1),
        )

    def sample_action(
        self,
        model: DefaultHumanoidModel,
        carry: None,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, None, AuxOutputs]:
        action_dist_j = self._run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
        )
        action_j = action_dist_j.sample(seed=rng)
        action_log_prob_j = action_dist_j.log_prob(action_j)

        return action_j, None, AuxOutputs(log_probs=action_log_prob_j)


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.default_humanoid.walking
    # To visualize the environment, use the following command:
    #   python -m examples.default_humanoid.walking run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m examples.default_humanoid.walking num_envs=8 batch_size=4
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=32,
            epochs_per_log_step=1,
            rollout_length_seconds=2.0,
            num_rollout_levels=3,
            # Logging parameters.
            # log_full_trajectory_every_n_seconds=60,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
        ),
    )
