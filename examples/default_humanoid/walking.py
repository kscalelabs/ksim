"""Defines simple task for training a walking policy for the default humanoid."""

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

NUM_JOINTS = 21


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


class NaiveVelocityReward(ksim.Reward):
    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return trajectory.qvel[..., 0].clip(max=5.0)


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
        num_inputs = NUM_JOINTS + NUM_JOINTS + 160 + 96 + NUM_JOINTS + 2 + 1
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
        com_inertia_n: Array,
        com_vel_n: Array,
        act_frc_obs_n: Array,
        lin_vel_cmd_2: Array,
        ang_vel_cmd_1: Array,
    ) -> distrax.Normal:
        obs_n = jnp.concatenate(
            [
                dh_joint_pos_n,  # NUM_JOINTS
                dh_joint_vel_n,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                act_frc_obs_n,  # 21
                lin_vel_cmd_2,  # 2
                ang_vel_cmd_1,  # 1
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

        # return distrax.Transformed(distrax.Normal(mean_n, std_n), distrax.Tanh())
        return distrax.Normal(mean_n, std_n)


class DefaultHumanoidCritic(eqx.Module):
    """Critic for the walking task."""

    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray) -> None:
        num_inputs = NUM_JOINTS + NUM_JOINTS + 160 + 96 + NUM_JOINTS + 3 + 3 + 2 + 1
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
        dh_joint_pos_n: Array,
        dh_joint_vel_n: Array,
        com_inertia_n: Array,
        com_vel_n: Array,
        act_frc_obs_n: Array,
        lin_vel_obs_3: Array,
        ang_vel_obs_3: Array,
        lin_vel_cmd_2: Array,
        ang_vel_cmd_1: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [
                dh_joint_pos_n,  # NUM_JOINTS
                dh_joint_vel_n,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                act_frc_obs_n,  # 21
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
                lin_vel_cmd_2,  # 2
                ang_vel_cmd_1,  # 1
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
class HumanoidWalkingTaskConfig(ksim.PPOConfig):
    """Config for the humanoid walking task."""

    # Reward parameters.
    use_naive_reward: bool = xax.field(
        value=False,
        help="Whether to use the naive velocity reward.",
    )
    domain_randomize: bool = xax.field(
        value=True,
        help="Whether to domain randomize the model.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=1e-4,
        help="Learning rate for PPO.",
    )
    max_grad_norm: float = xax.field(
        value=0.5,
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
        mjcf_path = (Path(__file__).parent / "scene.mjcf").resolve().as_posix()
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)

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
        if self.config.domain_randomize:
            return [
                ksim.StaticFrictionRandomization(),
                ksim.ArmatureRandomization(),
                ksim.MassMultiplicationRandomization.from_body_name(physics_model, "torso"),
                ksim.JointDampingRandomization(),
                ksim.JointZeroPositionRandomization(),
            ]
        else:
            return []

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        if self.config.domain_randomize:
            return [
                ksim.PushEvent(
                    x_force=1.0,
                    y_force=1.0,
                    z_force=0.0,
                    interval_range=(1.0, 2.0),
                ),
            ]
        else:
            return []

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        scale = 0.0 if self.config.domain_randomize else 0.01
        return [
            ksim.RandomJointPositionReset(scale=scale),
            ksim.RandomJointVelocityReset(scale=scale),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        noise = 0.0 if self.config.domain_randomize else 0.01
        return [
            ksim.JointPositionObservation(noise=noise),
            ksim.JointVelocityObservation(noise=noise),
            ksim.ActuatorForceObservation(noise=noise),
            ksim.CenterOfMassInertiaObservation(noise=noise),
            ksim.CenterOfMassVelocityObservation(noise=noise),
            ksim.BaseLinearVelocityObservation(noise=noise),
            ksim.BaseAngularVelocityObservation(noise=noise),
            ksim.BaseLinearAccelerationObservation(noise=noise),
            ksim.BaseAngularAccelerationObservation(noise=noise),
            ksim.ActuatorAccelerationObservation(noise=noise),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            # ksim.LinearVelocityStepCommand(
            #     x_range=(0.0, 3.0),
            #     y_range=(0.0, 0.0),
            #     x_fwd_prob=0.8,
            #     y_fwd_prob=0.5,
            #     x_zero_prob=0.2,
            #     y_zero_prob=0.8,
            # ),
            ksim.LinearVelocityCommand(
                x_range=(0.0, 2.5),
                y_range=(0.0, 0.0),
                x_zero_prob=0.1,
                y_zero_prob=1.0,
                switch_prob=self.config.ctrl_dt / 5,
            ),
            ksim.AngularVelocityStepCommand(
                scale=0.2,
                zero_prob=0.2,
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards = [
            ksim.BaseHeightRangeReward(z_lower=0.8, z_upper=1.5, scale=0.5),
            ksim.LinearVelocityZPenalty(scale=-0.01),
            ksim.AngularVelocityXYPenalty(scale=-0.01),
        ]

        # Use this to toggle the "naive" mode, where the model just learns to
        # move forward as quickly as possible.
        if self.config.use_naive_reward:
            rewards += [
                NaiveVelocityReward(scale=0.1),
            ]
        else:
            rewards += [
                ksim.LinearVelocityTrackingPenalty(
                    command_name="linear_velocity_command",
                    scale=-0.1,
                ),
                ksim.AngularVelocityTrackingPenalty(
                    command_name="angular_velocity_step_command",
                    scale=-0.01,
                ),
            ]

        return rewards

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.6, unhealthy_z_upper=1.5),
            ksim.FastAccelerationTermination(),
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
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"] / 50.0
        act_frc_obs_n = observations["actuator_force_observation"] / 100.0
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        ang_vel_cmd_1 = commands["angular_velocity_step_command"]
        return model.actor(
            dh_joint_pos_n,
            dh_joint_vel_n,
            com_inertia_n,
            com_vel_n,
            act_frc_obs_n,
            lin_vel_cmd_2,
            ang_vel_cmd_1,
        )

    def _run_critic(
        self,
        model: DefaultHumanoidModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        dh_joint_pos_n = observations["joint_position_observation"]  # 26
        dh_joint_vel_n = observations["joint_velocity_observation"]  # 27
        com_inertia_n = observations["center_of_mass_inertia_observation"]  # 160
        com_vel_n = observations["center_of_mass_velocity_observation"]  # 96
        act_frc_obs_n = observations["actuator_force_observation"] / 100.0  # 21
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]  # 3
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]  # 3
        lin_vel_cmd_2 = commands["linear_velocity_command"]  # 2
        ang_vel_cmd_1 = commands["angular_velocity_step_command"]  # 1
        return model.critic(
            dh_joint_pos_n,
            dh_joint_vel_n,
            com_inertia_n,
            com_vel_n,
            act_frc_obs_n,
            lin_vel_obs_3,
            ang_vel_obs_3,
            lin_vel_cmd_2,
            ang_vel_cmd_1,
        )

    def get_on_policy_log_probs(
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

    def get_log_probs(
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
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=4.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            use_mit_actuators=True,
            valid_every_n_steps=50,
            domain_randomize=False,
            use_naive_reward=True,
        ),
    )
