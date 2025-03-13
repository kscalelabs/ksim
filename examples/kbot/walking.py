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
    EnergyPenalty,
    FeetClearancePenaltyBuilder,
    FootContactPenaltyBuilder,
    FootSlipPenaltyBuilder,
    HeightReward,
    JointAccelerationPenalty,
    LinearVelocityZPenalty,
    OrientationPenalty,
    TorquePenalty,
    TrackAngularVelocityZReward,
    TrackLinearVelocityXYReward,
)
from ksim.builders.terminations import PitchTooGreatTermination, RollTooGreatTermination
from ksim.env.mjx.mjx_env import MjxEnv, MjxEnvConfig
from ksim.model.base import ActorCriticAgent
from ksim.model.factory import mlp_actor_critic_agent
from ksim.task.ppo import PPOConfig, PPOTask

######################
# Static Definitions #
######################

NUM_OUTPUTS = 20


@dataclass
class KBotWalkingConfig(PPOConfig, MjxEnvConfig):
    # Robot model name to use.
    robot_model_name: str = xax.field(value="kbot-v1-feet")

    ####################
    # Task Definitions #
    ####################


class KBotWalkingTask(PPOTask[KBotWalkingConfig]):
    def get_environment(self) -> MjxEnv:
        return MjxEnv(
            self.config,
            terminations=[
                # IllegalContactTerminationBuilder(
                #     body_names=[
                #         "shoulder",
                #         "shoulder_2",
                #         "hand_shell",
                #         "hand_shell_2",
                #         "leg0_shell",
                #         "leg0_shell_2",
                #     ],
                # ),
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
                TrackLinearVelocityXYReward(scale=10.0),
                HeightReward(scale=0.2, height_target=1.0),
                TrackAngularVelocityZReward(scale=7.5),
                ActionSmoothnessPenalty(scale=-0.0),
                OrientationPenalty(scale=-0.5, target_orientation=[0.0, 0.0, 0.0]),
                TorquePenalty(scale=-0.0),
                EnergyPenalty(scale=-0.0),
                JointAccelerationPenalty(scale=-0.0),
                FootSlipPenaltyBuilder(
                    scale=-0.25,
                    foot_geom_names=[
                        "foot1_collision_sphere_1",
                        "foot1_collision_sphere_2",
                        "foot1_collision_sphere_3",
                        "foot1_collision_sphere_4",
                        "foot1_collision_box",
                        "foot3_collision_sphere_1",
                    ],
                ),
                FeetClearancePenaltyBuilder(
                    scale=-0.0,
                    foot_geom_names=[
                        "foot1_collision_sphere_1",
                        "foot1_collision_sphere_2",
                        "foot1_collision_sphere_3",
                        "foot1_collision_sphere_4",
                        "foot1_collision_box",
                        "foot3_collision_sphere_1",
                    ],
                    max_foot_height=0.2,
                ),
                FootContactPenaltyBuilder(
                    scale=-0.1,
                    foot_geom_names=[
                        "foot3_collision_sphere_1",
                        "foot3_collision_sphere_2",
                        "foot3_collision_sphere_3",
                        "foot3_collision_sphere_4",
                        "foot3_collision_box",
                        "foot1_collision_sphere_1",
                        "foot1_collision_sphere_2",
                        "foot1_collision_sphere_3",
                        "foot1_collision_sphere_4",
                        "foot1_collision_box",
                    ],
                    allowed_contact_prct=0.7,
                    skip_if_zero_command=("linear_velocity_command", "angular_velocity_command"),
                ),
                DefaultPoseDeviationPenaltyBuilder(
                    scale=-0.1,
                    default_positions={
                        "left_shoulder_pitch_03": 0.0,
                        "left_shoulder_roll_03": 0.0,
                        "left_shoulder_yaw_02": 0.0,
                        "left_elbow_02": 0.0,
                        "left_wrist_02": 0.0,
                        "right_shoulder_pitch_03": 0.0,
                        "right_shoulder_roll_03": 0.0,
                        "right_shoulder_yaw_02": 0.0,
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
                        "left_shoulder_pitch_03": 1.0,
                        "left_shoulder_roll_03": 1.0,
                        "left_shoulder_yaw_02": 1.0,
                        "left_elbow_02": 1.0,
                        "left_wrist_02": 1.0,
                        "right_shoulder_pitch_03": 1.0,
                        "right_shoulder_roll_03": 1.0,
                        "right_shoulder_yaw_02": 1.0,
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
        """No need to initialize actor carry."""
        return None

    def get_init_critic_carry(self) -> None:
        """No need to initialize critic carry."""
        return None


if __name__ == "__main__":
    # python -m examples.kbot.walking action=env
    KBotWalkingTask.launch(
        KBotWalkingConfig(
            num_learning_epochs=8,
            num_env_states_per_minibatch=8192,
            num_minibatches=32,
            num_envs=2048,
            dt=0.001,
            ctrl_dt=0.005,
            learning_rate=1e-5,
            save_every_n_steps=25,
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
            eval_rollout_length=1000,
        ),
    )
