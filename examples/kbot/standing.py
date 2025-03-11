"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

from ksim.builders.commands import AngularVelocityCommand, LinearVelocityCommand
from ksim.builders.observation import (
    BaseAngularVelocityObservation,
    BaseLinearVelocityObservation,
    BaseOrientationObservation,
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
from ksim.builders.terminations import (
    PitchTooGreatTermination,
    RollTooGreatTermination,
)
from ksim.env.mjx.mjx_env import MjxEnv, MjxEnvConfig
from ksim.model.formulations import ActionModel, ActorCriticAgent, GaussianActionModel
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask

NUM_OUTPUTS = 14  # No shoulders


class KBotActorModel(GaussianActionModel):
    mlp: MLP
    action_clipping: float = 20.0
    action_scale: float = 0.5

    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        lin_vel_cmd_2 = cmd["linear_velocity_command"]
        ang_vel_cmd_1 = cmd["angular_velocity_command"]
        joint_pos_j = obs["joint_position_observation_noisy"]
        joint_vel_j = obs["joint_velocity_observation_noisy"]
        imu_acc_3 = obs["imu_acc_sensor_observation"]
        imu_gyro_3 = obs["imu_gyro_sensor_observation"]

        x_n = jnp.concatenate(
            [
                lin_vel_cmd_2,
                ang_vel_cmd_1,
                imu_acc_3,
                imu_gyro_3,
                joint_pos_j,
                joint_vel_j,
            ],
            axis=-1,
        )

        actions_n = self.mlp(x_n)

        actions_n = jnp.clip(actions_n, -self.action_clipping, self.action_clipping)
        actions_n = actions_n * self.action_scale
        return actions_n


class KBotZeroActions(GaussianActionModel):
    mlp: MLP

    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        return jnp.zeros_like(NUM_OUTPUTS)

    def sample_and_log_prob(
        self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array], rng: PRNGKeyArray
    ) -> tuple[Array, Array]:
        mean = self(obs, cmd)
        return mean, mean


class KBotCriticModel(nn.Module):
    mlp: MLP

    @nn.compact
    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> jax.Array:
        # Concatenate all observations and commands (critic has privileged information)
        clean_obs: FrozenDict[str, Array] = FrozenDict(
            {k: v for k, v in obs.items() if "_noisy" not in k}
        )
        x_n = jnp.concatenate([obs_array for obs_array in clean_obs.values()], axis=-1)
        cmd_n = jnp.concatenate([cmd_array for cmd_array in cmd.values()], axis=-1)
        x_n = jnp.concatenate([x_n, cmd_n], axis=-1)

        value_estimate = self.mlp(x_n)

        return value_estimate


@dataclass
class KBotStandingConfig(PPOConfig, MjxEnvConfig):
    # Robot model name to use.
    robot_model_name: str = xax.field(value="examples/kbot/")

    # ML model parameters.
    actor_hidden_dims: int = xax.field(value=512)
    actor_num_layers: int = xax.field(value=2)
    critic_hidden_dims: int = xax.field(value=512)
    critic_num_layers: int = xax.field(value=4)
    init_noise_std: float = xax.field(value=1.0)

    # Termination conditions.
    max_episode_length: float = xax.field(value=10.0)
    max_pitch: float = xax.field(value=0.1)
    max_roll: float = xax.field(value=0.1)
    action_clipping: float = xax.field(value=10.0)
    action_scale: float = xax.field(value=0.3)

    actuator_type: str = xax.field(value="mit", help="The type of actuator to use.")


class KBotStandingTask(PPOTask[KBotStandingConfig]):
    def get_environment(self) -> MjxEnv:
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
                HeightReward(scale=5.0, height_target=0.98),
                TrackAngularVelocityZReward(scale=1.0),
                ActionSmoothnessPenalty(scale=-0.0),
                OrientationPenalty(scale=-0.5, target_orientation=[0.0, 0.0, -1.0]),
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
                BaseOrientationObservation(noise_type="gaussian", noise=0.01),
                JointPositionObservation(noise_type="gaussian", noise=0.01),
                JointVelocityObservation(noise_type="gaussian", noise=0.01),
                SensorObservationBuilder(sensor_name="imu_acc"),  # Sensor has noise already.
                SensorObservationBuilder(sensor_name="imu_gyro"),  # Sensor has noise already.
                # Clean observations
                # NOTE: Depending on much we value flexibility vs cleanliness
                # we might want to abstract this `clean` logic in `MjxEnv`
                BaseOrientationObservation(noise_type="gaussian", noise=0.0),
                BaseLinearVelocityObservation(noise_type="gaussian", noise=0.0),
                BaseAngularVelocityObservation(noise_type="gaussian", noise=0.0),
                JointPositionObservation(noise_type="gaussian", noise=0.0),
                JointVelocityObservation(noise_type="gaussian", noise=0.0),
            ],
            commands=[
                LinearVelocityCommand(
                    x_scale=0.0,
                    y_scale=0.0,
                    switch_prob=0.02,
                    zero_prob=0.3,
                ),
                AngularVelocityCommand(
                    scale=0.0,
                    switch_prob=0.02,
                    zero_prob=0.8,
                ),
            ],
        )

    def get_model(self, key: PRNGKeyArray) -> ActorCriticAgent:
        return ActorCriticAgent(
            actor_module=KBotActorModel(
                mlp=MLP(
                    num_hidden_layers=self.config.actor_num_layers,
                    hidden_features=self.config.actor_hidden_dims,
                    out_features=NUM_OUTPUTS,
                ),
                init_log_std=-0.7,
                num_outputs=NUM_OUTPUTS,
                action_clipping=self.config.action_clipping,
                action_scale=self.config.action_scale,
            ),
            critic_module=KBotCriticModel(
                mlp=MLP(
                    num_hidden_layers=self.config.critic_num_layers,
                    hidden_features=self.config.critic_hidden_dims,
                    out_features=1,
                ),
            ),
        )

    def get_init_actor_carry(self) -> jnp.ndarray | None:
        # return jnp.zeros((self.config.actor_num_layers, self.config.actor_hidden_dims))
        return None

    def get_init_critic_carry(self) -> None:
        return None

    # Overloading to run KBotZeroActions instead of default Actor model
    def run(self) -> None:
        """Highest level entry point for RL tasks, determines what to run."""
        match self.config.action:
            case "train":
                self.run_training()

            case "env":
                mlp = MLP(
                    num_hidden_layers=self.config.actor_num_layers,
                    hidden_features=self.config.actor_hidden_dims,
                    out_features=NUM_OUTPUTS,
                )
                actor: ActionModel
                match self.config.viz_action:
                    case "policy":
                        actor = KBotActorModel(num_outputs=NUM_OUTPUTS, mlp=mlp, init_log_std=-0.7)
                    case "zero":
                        actor = KBotZeroActions(num_outputs=NUM_OUTPUTS, mlp=mlp, init_log_std=-0.7)
                    case _:
                        raise ValueError(
                            f"Invalid action: {self.config.viz_action}."
                            f" Should be one of `policy` or `zero`."
                        )

                model = ActorCriticAgent(
                    actor_module=actor,
                    critic_module=KBotCriticModel(
                        mlp=MLP(
                            num_hidden_layers=self.config.critic_num_layers,
                            hidden_features=self.config.critic_hidden_dims,
                            out_features=1,
                        ),
                    ),
                )

                self.run_environment(model)

            case _:
                raise ValueError(
                    f"Invalid action: {self.config.action}. Should be one of `train` or `env`."
                )


if __name__ == "__main__":
    # python -m examples.kbot.standing
    KBotStandingTask.launch(
        KBotStandingConfig(
            num_learning_epochs=8,
            num_env_states_per_minibatch=8192,
            num_minibatches=64,
            num_envs=2048,
            dt=0.001,
            max_episode_length=20.0,
            ctrl_dt=0.008,
            learning_rate=5e-5,
            save_every_n_seconds=60 * 4,
            only_save_most_recent=False,
            reward_scaling_alpha=0.0,
            obs_norm_alpha=0.0,
            scale_rewards=True,
            solver_iterations=6,
            solver_ls_iterations=6,
            actuator_type="mit",
        ),
    )
