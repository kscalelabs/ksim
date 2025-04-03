"""Pseudo-Inverse Kinematics task for the default humanoid."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

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
from ksim.utils.mujoco import remove_joints_except

NUM_JOINTS = 3  # disabling all DoFs except for the right arm.


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


class DefaultHumanoidActor(eqx.Module):
    """Actor for the walking task."""

    mlp: eqx.nn.MLP
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()
    mean_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
        mean_scale: float,
    ) -> None:
        num_inputs = NUM_JOINTS + NUM_JOINTS + 3 + 3 + 4
        num_outputs = NUM_JOINTS

        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=num_outputs * 2,
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.mean_scale = mean_scale

    def __call__(
        self,
        dh_joint_pos_n: Array,
        dh_joint_vel_n: Array,
        imu_acc_3: Array,
        xyz_target_3: Array,
        quat_target_4: Array,
    ) -> distrax.Normal:
        obs_n = jnp.concatenate(
            [
                dh_joint_pos_n,  # NUM_JOINTS
                dh_joint_vel_n,  # NUM_JOINTS
                imu_acc_3,  # 3
                xyz_target_3,  # 3
                quat_target_4,  # 4
            ],
            axis=-1,
        )

        prediction_n = self.mlp(obs_n)
        mean_n = prediction_n[..., :NUM_JOINTS]
        std_n = prediction_n[..., NUM_JOINTS:]

        # Scale the mean.
        mean_n = jnp.tanh(mean_n) * self.mean_scale

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        return distrax.Normal(mean_n, std_n)


class DefaultHumanoidCritic(eqx.Module):
    """Critic for the walking task."""

    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray) -> None:
        num_inputs = NUM_JOINTS + NUM_JOINTS + NUM_JOINTS + 3 + 3 + 4
        num_outputs = 1

        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=num_outputs,
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )

    def __call__(
        self,
        joint_pos_n: Array,
        joint_vel_n: Array,
        actuator_force_n: Array,
        imu_acc_3: Array,
        xyz_target_3: Array,
        quat_target_4: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [
                joint_pos_n,  # NUM_JOINTS
                joint_vel_n,  # NUM_JOINTS
                actuator_force_n,  # NUM_JOINTS
                imu_acc_3,  # 3
                xyz_target_3,  # 3
                quat_target_4,  # 4
            ],
            axis=-1,
        )
        return self.mlp(x_n)


class DefaultHumanoidModel(eqx.Module):
    actor: DefaultHumanoidActor
    critic: DefaultHumanoidCritic

    def __init__(self, key: PRNGKeyArray) -> None:
        self.actor = DefaultHumanoidActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=1.0,
            mean_scale=1.0,
        )
        self.critic = DefaultHumanoidCritic(key)


@dataclass
class HumanoidPseudoIKTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

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
    use_mit_actuators: bool = xax.field(
        value=False,
        help="Whether to use the MIT actuator model, where the actions are position commands",
    )
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


Config = TypeVar("Config", bound=HumanoidPseudoIKTaskConfig)


class HumanoidPseudoIKTask(ksim.PPOTask[Config], Generic[Config]):
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
        mj_model_joint_removed = remove_joints_except(mjcf_path, ["shoulder1_right", "shoulder2_right", "elbow_right"])
        mj_model = mujoco.MjModel.from_xml_string(mj_model_joint_removed)

        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG

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
        if self.config.use_mit_actuators:
            if metadata is None:
                raise ValueError("Metadata is required for MIT actuators")
            return ksim.MITPositionActuators(physics_model, metadata)
        else:
            return ksim.TorqueActuators()

    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.Randomization]:
        return [
            # ksim.StaticFrictionRandomization(),
            # ksim.ArmatureRandomization(),
            # ksim.MassMultiplicationRandomization.from_body_name(physics_model, "upper_arm_right"),
            # ksim.JointDampingRandomization(),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return []

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return [
            ksim.RandomJointPositionReset(),
            ksim.RandomJointVelocityReset(),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(freejoint_first=False),
            ksim.JointVelocityObservation(freejoint_first=False),
            ksim.ActuatorForceObservation(),
            # ksim.ActuatorAccelerationObservation(freejoint_first=False),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc"),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.CartesianBodyTargetCommand.create(
                model=physics_model,
                command_name="cartesian_body_target_command_upper_arm_right",
                pivot_name="upper_arm_right",
                base_name="pelvis",
                sample_sphere_radius=0.5,
                positive_x=True,  # only sample in the positive x direction
                positive_y=False,
                positive_z=False,
                switch_prob=self.config.ctrl_dt / 1,  # will last 1 seconds in expectation
                vis_radius=0.05,
                vis_color=(1.0, 0.0, 0.0, 0.8),
            ),
            ksim.GlobalBodyQuaternionCommand.create(
                model=physics_model,
                command_name="global_body_quaternion_command_hand_right",
                base_name="hand_right",
                switch_prob=self.config.ctrl_dt / 1,  # will last 1 seconds in expectation
                null_prob=0.5,
                vis_magnitude=0.5,
                vis_size=0.05,
                vis_color=(0.0, 0.0, 1.0, 0.8),
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.ContinuousCartesianBodyTargetReward.create(
                model=physics_model,
                tracked_body_name="hand_right",
                base_body_name="pelvis",
                norm="l2",
                scale=1.0,
                sensitivity=1.0,
                threshold=0.0001,  # with l2 norm, this is 1cm of error
                time_bonus_scale=0.1,
                command_name="cartesian_body_target_command_upper_arm_right",
            ),
            ksim.GlobalBodyQuaternionReward.create(
                model=physics_model,
                tracked_body_name="hand_right",
                base_body_name="pelvis",
                norm="l2",
                scale=0.1,
                sensitivity=1.0,
                command_name="global_body_quaternion_command_hand_right",
            ),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.FastAccelerationTermination(),
            # TODO: add for collisions
        ]

    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidModel:
        return DefaultHumanoidModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> None:
        return None

    def _run_actor(
        self,
        model: DefaultHumanoidModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> distrax.Normal:
        dh_joint_pos_n = observations["joint_position_observation"]
        dh_joint_vel_n = observations["joint_velocity_observation"] / 50.0
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        xyz_target_3 = commands["cartesian_body_target_command_upper_arm_right"]
        quat_target_4 = commands["global_body_quaternion_command_hand_right"]
        return model.actor(
            dh_joint_pos_n=dh_joint_pos_n,
            dh_joint_vel_n=dh_joint_vel_n,
            imu_acc_3=imu_acc_3,
            xyz_target_3=xyz_target_3,
            quat_target_4=quat_target_4,
        )

    def _run_critic(
        self,
        model: DefaultHumanoidModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        joint_pos_n = observations["joint_position_observation"]  # 26
        joint_vel_n = observations["joint_velocity_observation"] / 100.0  # 27
        actuator_force_n = observations["actuator_force_observation"]  # 27
        imu_acc_3 = observations["sensor_observation_imu_acc"]  # 3
        xyz_target_3 = commands["cartesian_body_target_command_upper_arm_right"]  # 3
        quat_target_4 = commands["global_body_quaternion_command_hand_right"]  # 4
        return model.critic(
            joint_pos_n=joint_pos_n,
            joint_vel_n=joint_vel_n,
            actuator_force_n=actuator_force_n,
            imu_acc_3=imu_acc_3,
            xyz_target_3=xyz_target_3,
            quat_target_4=quat_target_4,
        )

    def get_on_policy_variables(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        if not isinstance(trajectories.aux_outputs, AuxOutputs):
            raise ValueError("No aux outputs found in trajectories")
        return trajectories.aux_outputs.log_probs

    def get_on_policy_values(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        if not isinstance(trajectories.aux_outputs, AuxOutputs):
            raise ValueError("No aux outputs found in trajectories")
        return trajectories.aux_outputs.values

    def get_off_policy_variables(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        # Vectorize over both batch and time dimensions.
        par_fn = jax.vmap(self._run_actor, in_axes=(None, 0, 0))
        action_dist_btn = par_fn(model, trajectories.obs, trajectories.command)

        # Compute the log probabilities of the trajectory's actions according
        # to the current policy, along with the entropy of the distribution.
        action_btn = trajectories.action / model.actor.mean_scale
        log_probs_btn = action_dist_btn.log_prob(action_btn)
        entropy_btn = action_dist_btn.entropy()

        return log_probs_btn, entropy_btn

    def get_values(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        # Vectorize over both batch and time dimensions.
        par_fn = jax.vmap(self._run_critic, in_axes=(None, 0, 0))
        values_bt1 = par_fn(model, trajectories.obs, trajectories.command)

        # Remove the last dimension.
        return values_bt1.squeeze(-1)

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
        action_dist_n = self._run_actor(model, observations, commands)
        action_n = action_dist_n.sample(seed=rng)
        action_log_prob_n = action_dist_n.log_prob(action_n)

        critic_n = self._run_critic(model, observations, commands)
        value_n = critic_n.squeeze(-1)

        return action_n, None, AuxOutputs(log_probs=action_log_prob_n, values=value_n)


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.default_humanoid.walking
    # To visualize the environment, use the following command:
    #   python -m examples.default_humanoid.walking run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m examples.default_humanoid.walking num_envs=8 batch_size=4
    HumanoidPseudoIKTask.launch(
        HumanoidPseudoIKTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            # Logging parameters.
            log_full_trajectory_every_n_seconds=60,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=4.0,
            # If you experience segfaults, try disabling the markers.
            # render_markers=True,
        ),
    )
