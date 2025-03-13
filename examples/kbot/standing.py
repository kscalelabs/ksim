"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp
import xax
from jaxtyping import PRNGKeyArray

from ksim.builders.commands import AngularVelocityCommand, LinearVelocityCommand
from ksim.builders.observation import (
    ActuatorForceObservation,
    BaseAngularVelocityObservation,
    BaseLinearVelocityObservation,
    BaseOrientationObservation,
    CenterOfMassInertiaObservation,
    CenterOfMassVelocityObservation,
    JointPositionObservation,
    JointVelocityObservation,
    SensorObservationBuilder,
)
from ksim.builders.resets import (
    RandomizeJointPositions,
    RandomizeJointVelocities,
    XYPositionResetBuilder,
)
from ksim.builders.rewards import (
    ActionSmoothnessPenalty,
    AngularVelocityXYPenalty,
    DefaultPoseDeviationPenaltyBuilder,
    HeightReward,
    JointAccelerationPenalty,
    LinearVelocityZPenalty,
    OrientationPenalty,
    TrackAngularVelocityZReward,
    TrackLinearVelocityXYReward,
)
from ksim.builders.terminations import PitchTooGreatTermination, RollTooGreatTermination
from ksim.env.mjx.mjx_env import MjxEnv, MjxEnvConfig
from ksim.model.base import ActorCriticAgent
from ksim.model.factory import mlp_actor_critic_agent
from ksim.task.ppo import PPOConfig, PPOTask

NUM_OUTPUTS = 14  # No shoulders

######################
# Static Definitions #
######################


@dataclass
class KBotStandingConfig(PPOConfig, MjxEnvConfig):
    """Combining configs for the KBot standing task and fixing params."""

    robot_model_name: str = xax.field(value="examples/kbot/")


####################
# Task Definitions #
####################


class KBotStandingTask(PPOTask[KBotStandingConfig]):
    def get_environment(self) -> MjxEnv:
        """Get the environment."""
        return MjxEnv(
            self.config,
            terminations=[
                RollTooGreatTermination(max_roll=1.04),
                PitchTooGreatTermination(max_pitch=1.04),
            ],
            resets=[
                XYPositionResetBuilder(),
                RandomizeJointVelocities(scale=0.01),
                RandomizeJointPositions(scale=0.01),
            ],
            rewards=[
                LinearVelocityZPenalty(scale=-0.0),
                AngularVelocityXYPenalty(scale=-0.15),
                TrackLinearVelocityXYReward(scale=1.0),
                HeightReward(scale=1.0, height_target=1.0),
                TrackAngularVelocityZReward(scale=1.0),
                ActionSmoothnessPenalty(scale=-0.0),
                OrientationPenalty(scale=-0.5, target_orientation=[0.0, 0.0, 0.0]),
                JointAccelerationPenalty(scale=0.0),  # -2e-7),
                DefaultPoseDeviationPenaltyBuilder(
                    scale=-0.1,
                    default_positions={
                        # "left_shoulder_pitch_03": 0.0,
                        # "left_shoulder_roll_03": 0.0,
                        # "left_shoulder_yaw_02": 0.0,
                        "left_elbow_02": 0.0,
                        "left_wrist_02": 0.0,
                        # "right_shoulder_pitch_03": 0.0,
                        # "right_shoulder_roll_03": 0.0,
                        # "right_shoulder_yaw_02": 0.0,
                        "right_elbow_02": 0.0,
                        "right_wrist_02": 0.0,
                        "left_hip_pitch_04": 0.0,
                        "left_hip_roll_03": 0.0,
                        "left_hip_yaw_03": 0.0,
                        "left_knee_04": 0.0,
                        "left_ankle_02": 0.0,
                        "right_hip_pitch_04": 0.0,
                        "right_hip_roll_03": 0.0,
                        "right_hip_yaw_03": 0.0,
                        "right_knee_04": 0.0,
                        "right_ankle_02": 0.0,
                    },
                    deviation_weights={
                        # "left_shoulder_pitch_03": 1.0,
                        # "left_shoulder_roll_03": 1.0,
                        # "left_shoulder_yaw_02": 1.0,
                        "left_elbow_02": 1.0,
                        "left_wrist_02": 1.0,
                        # "right_shoulder_pitch_03": 1.0,
                        # "right_shoulder_roll_03": 1.0,
                        # "right_shoulder_yaw_02": 1.0,
                        "right_elbow_02": 1.0,
                        "right_wrist_02": 1.0,
                        "left_hip_pitch_04": 2.0,
                        "left_hip_roll_03": 2.0,
                        "left_hip_yaw_03": 2.0,
                        "left_knee_04": 1.0,
                        "left_ankle_02": 1.0,
                        "right_hip_pitch_04": 2.0,
                        "right_hip_roll_03": 2.0,
                        "right_hip_yaw_03": 2.0,
                        "right_knee_04": 1.0,
                        "right_ankle_02": 1.0,
                    },
                ),
            ],
            observations=[
                BaseOrientationObservation(noise_type="gaussian"),
                BaseLinearVelocityObservation(noise_type="gaussian"),
                BaseAngularVelocityObservation(noise_type="gaussian"),
                JointPositionObservation(noise_type="gaussian"),
                JointVelocityObservation(noise_type="gaussian"),
                CenterOfMassInertiaObservation(noise_type="gaussian"),
                CenterOfMassVelocityObservation(noise_type="gaussian"),
                ActuatorForceObservation(noise_type="gaussian"),
                SensorObservationBuilder(sensor_name="imu_acc"),  # Sensor has noise already.
                SensorObservationBuilder(sensor_name="imu_gyro"),  # Sensor has noise already.
            ],
            commands=[
                LinearVelocityCommand(x_scale=0.0, y_scale=0.0, switch_prob=0.02, zero_prob=0.3),
                AngularVelocityCommand(scale=0.0, switch_prob=0.02, zero_prob=0.8),
            ],
        )

    def get_model(self, key: PRNGKeyArray) -> ActorCriticAgent:
        """Get the model."""
        return mlp_actor_critic_agent(
            num_actions=NUM_OUTPUTS,
            prediction_type="mean_std",
            distribution_type="tanh_gaussian",
            actor_hidden_dims=(64,) * 5,
            critic_hidden_dims=(64,) * 5,
            kernel_initialization=nn.initializers.lecun_normal(),
            post_process_kwargs={"min_std": 0.01, "max_std": 1.0, "var_scale": 1.0},
        )

    def get_init_actor_carry(self) -> jnp.ndarray | None:
        """Get the initial actor carry."""
        return None

    def get_init_critic_carry(self) -> None:
        """Get the initial critic carry."""
        return None


if __name__ == "__main__":
    # python -m examples.kbot.standing
    KBotStandingTask.launch(
        KBotStandingConfig(
            num_learning_epochs=8,
            num_env_states_per_minibatch=8192,
            num_minibatches=32,
            num_envs=2048,
            dt=0.001,
            ctrl_dt=0.005,
            learning_rate=1e-5,
            save_every_n_steps=50,
            only_save_most_recent=False,
            reward_scaling_alpha=0.0,
            obs_norm_alpha=0.0,
            solver_iterations=6,
            solver_ls_iterations=6,
            actuator_type="mit",
            scale_rewards=False,
            gamma=0.97,
            lam=0.95,
            normalize_advantage=True,
            normalize_advantage_in_minibatch=True,
            entropy_coef=0.001,
            clip_param=0.2,
            use_clipped_value_loss=False,
            max_grad_norm=1.0,
            max_action_latency=0.0,
            min_action_latency=0.0,
        ),
    )
