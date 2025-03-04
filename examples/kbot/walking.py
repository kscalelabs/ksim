"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass
from typing import Tuple

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
    BasePositionObservation,
    JointPositionObservation,
    JointVelocityObservation,
    SensorObservationBuilder,
)
from ksim.builders.resets import XYPositionResetKbotBuilder
from ksim.builders.rewards import (
    ActionSmoothnessPenalty,
    AngularVelocityXYPenalty,
    FootContactPenaltyBuilder,
    LinearVelocityZPenalty,
    TrackAngularVelocityZReward,
    TrackLinearVelocityXYReward,
)
from ksim.builders.terminations import IllegalContactTerminationBuilder
from ksim.env.mjx.mjx_env import MjxEnv, MjxEnvConfig
from ksim.model.formulations import ActionModel, ActorCriticAgent
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask

NUM_INPUTS = 49
NUM_OUTPUTS = 20


class KBotActorModel(ActionModel):
    mlp: MLP

    def setup(self) -> None:
        self.log_std = self.param("log_std", nn.initializers.constant(-0.7), (NUM_OUTPUTS,))

    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        x_n = jnp.concatenate([obs_array for obs_array in obs.values()], axis=-1)
        cmd_n = jnp.concatenate([cmd_array for cmd_array in cmd.values()], axis=-1)
        x_n = jnp.concatenate([x_n, cmd_n], axis=-1)
        actions_n = self.mlp(x_n)

        return actions_n

    def calc_log_prob(self, prediction: Array, action: Array) -> Array:
        mean = prediction
        std = jnp.exp(self.log_std)

        log_prob = (
            -0.5 * jnp.square((action - mean) / std) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
        )
        return jnp.sum(log_prob, axis=-1)

    def sample_and_log_prob(
        self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array], rng: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        mean = self(obs, cmd)
        std = jnp.exp(self.log_std)

        noise = jax.random.normal(rng, mean.shape)
        action = mean + noise * std
        log_prob = self.calc_log_prob(mean, action)

        return action, log_prob


class KBotZeroActions(ActionModel):
    mlp: MLP

    def setup(self) -> None:
        self.log_std = self.param("log_std", nn.initializers.constant(-0.7), (NUM_OUTPUTS,))

    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        x_n = jnp.concatenate([obs_array for obs_array in obs.values()], axis=-1)
        cmd_n = jnp.concatenate([cmd_array for cmd_array in cmd.values()], axis=-1)
        x_n = jnp.concatenate([x_n, cmd_n], axis=-1)
        actions_n = self.mlp(x_n)

        return actions_n

    def calc_log_prob(self, prediction: Array, action: Array) -> Array:
        mean = prediction
        std = jnp.exp(self.log_std)

        log_prob = (
            -0.5 * jnp.square((action - mean) / std) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
        )
        return jnp.sum(log_prob, axis=-1)

    def sample_and_log_prob(
        self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array], rng: PRNGKeyArray
    ) -> Tuple[Array, Array]:
        zeros = self(obs, cmd) * 0.0

        return zeros, zeros


class KBotCriticModel(nn.Module):
    mlp: MLP

    @nn.compact
    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> jax.Array:
        lin_vel_cmd_2 = cmd["linear_velocity_command"]
        ang_vel_cmd_1 = cmd["angular_velocity_command"]
        joint_pos_j = obs["joint_position_observation"]
        joint_vel_j = obs["joint_velocity_observation"]

        base_pos_3 = obs["base_position_observation"]
        base_ang_vel_3 = obs["base_angular_velocity_observation"]
        base_lin_vel_3 = obs["base_linear_velocity_observation"]
        base_quat_4 = obs["base_orientation_observation"]
        imu_acc_3 = obs["imu_acc_sensor_observation"]
        imu_gyro_3 = obs["imu_gyro_sensor_observation"]

        x_n = jnp.concatenate(
            [
                lin_vel_cmd_2,
                ang_vel_cmd_1,
                imu_acc_3,
                imu_gyro_3,
                base_pos_3,
                base_ang_vel_3,
                base_lin_vel_3,
                base_quat_4,
                joint_pos_j,
                joint_vel_j,
            ],
            axis=-1,
        )
        value_estimate = self.mlp(x_n)

        return value_estimate


@dataclass
class KBotWalkingConfig(PPOConfig, MjxEnvConfig):
    # Robot model name to use.
    robot_model_name: str = xax.field(value="kbot-v1-feet")

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


class KBotWalkingTask(PPOTask[KBotWalkingConfig]):
    def get_environment(self) -> MjxEnv:
        return MjxEnv(
            self.config,
            terminations=[
                IllegalContactTerminationBuilder(
                    body_names=[
                        "shoulder",
                        "shoulder_2",
                        "hand_shell",
                        "hand_shell_2",
                        "leg0_shell",
                        "leg0_shell_2",
                    ],
                ),
            ],
            resets=[
                XYPositionResetKbotBuilder(),
            ],
            rewards=[
                LinearVelocityZPenalty(scale=-0.1),
                AngularVelocityXYPenalty(scale=-0.1),
                TrackLinearVelocityXYReward(scale=0.1),
                TrackAngularVelocityZReward(scale=0.1),
                ActionSmoothnessPenalty(scale=-0.1),
                FootContactPenaltyBuilder(
                    scale=-0.1,
                    foot_body_names=["KB_D_501R_R_LEG_FOOT"],
                    allowed_contact_prct=0.7,
                    skip_if_zero_command=[
                        "linear_velocity_command",
                        "angular_velocity_command",
                    ],
                ),
                FootContactPenaltyBuilder(
                    scale=-0.1,
                    foot_body_names=["KB_D_501L_L_LEG_FOOT"],
                    allowed_contact_prct=0.7,
                    skip_if_zero_command=[
                        "linear_velocity_command",
                        "angular_velocity_command",
                    ],
                ),
            ],
            observations=[
                BasePositionObservation(noise_type="gaussian", noise=0.01),
                BaseOrientationObservation(noise_type="gaussian", noise=0.01),
                BaseLinearVelocityObservation(noise_type="gaussian", noise=0.01),
                BaseAngularVelocityObservation(noise_type="gaussian", noise=0.01),
                JointPositionObservation(noise_type="gaussian", noise=0.01),
                JointVelocityObservation(noise_type="gaussian", noise=0.01),
                SensorObservationBuilder(sensor_name="imu_acc"),  # Sensor has noise already.
                SensorObservationBuilder(sensor_name="imu_gyro"),  # Sensor has noise already.
            ],
            commands=[
                LinearVelocityCommand(
                    x_scale=1.0,
                    y_scale=0.0,
                    switch_prob=0.02,
                    zero_prob=0.3,
                ),
                AngularVelocityCommand(
                    scale=1.0,
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
                match self.config.viz_action:
                    case "policy":
                        actor = KBotActorModel(mlp=mlp)
                    case "zero":
                        actor = KBotZeroActions(mlp=mlp)
                    case _:
                        raise ValueError(
                            f"Invalid action: {self.config.viz_action}. Should be one of `policy` or `zero`."
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
    # python -m examples.kbot.walking action=env
    KBotWalkingTask.launch(
        KBotWalkingConfig(
            num_envs=1,
            num_steps_per_trajectory=100,
        ),
    )
