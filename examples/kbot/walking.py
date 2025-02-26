"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.base_env import EnvState
from ksim.env.builders.commands import AngularVelocityCommand, LinearVelocityCommand
from ksim.env.builders.observation import (
    BaseAngularVelocityObservation,
    BaseLinearVelocityObservation,
    BaseOrientationObservation,
    BasePositionObservation,
    JointPositionObservation,
    JointVelocityObservation,
    SensorObservationBuilder,
)
from ksim.env.builders.resets import XYPositionResetBuilder
from ksim.env.builders.rewards import (
    ActionSmoothnessPenalty,
    AngularVelocityXYPenalty,
    FootContactPenaltyBuilder,
    LinearVelocityZPenalty,
    TrackAngularVelocityZReward,
    TrackLinearVelocityXYReward,
)
from ksim.env.builders.terminations import IllegalContactTerminationBuilder
from ksim.env.mjx.mjx_env import MjxEnv
from ksim.model.formulations import (
    ActionModel,
    ActorCriticModel,
)
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask

NUM_INPUTS = 49
NUM_OUTPUTS = 20

# NOTE: implement after MLP is working.
# class RNNCell(eqx.Module):
#     num_inputs: int
#     num_hidden: int
#     num_layers: int
#     num_outputs: int

#     rnns: list[eqx.nn.GRUCell]
#     output_layer: eqx.nn.Linear

#     def __init__(
#         self,
#         num_inputs: int,
#         num_hidden: int,
#         num_layers: int,
#         num_outputs: int,
#         *,
#         key: PRNGKeyArray,
#     ) -> None:
#         super().__init__()

#         self.num_inputs = num_inputs
#         self.num_hidden = num_hidden
#         self.num_layers = num_layers
#         self.num_outputs = num_outputs

#         keys = jax.random.split(key, num_layers + 1)

#         self.rnns = [
#             eqx.nn.GRUCell(
#                 input_size=num_inputs if i == 0 else num_hidden,
#                 hidden_size=num_hidden,
#                 use_bias=True,
#                 key=keys[i],
#             )
#             for i in range(num_layers)
#         ]

#         self.output_layer = eqx.nn.Linear(
#             in_features=num_hidden,
#             out_features=num_outputs,
#             use_bias=True,
#             key=keys[-1],
#         )

#     @eqx.filter_jit
#     def __call__(
#         self,
#         x_n: jnp.ndarray,
#         state_ln: jnp.ndarray | None = None,
#     ) -> tuple[jnp.ndarray, jnp.ndarray]:
#         next_state_n_list = []
#         if state_ln is None:
#             state_ln = jnp.zeros((self.num_layers, self.num_hidden))
#         for i, rnn in enumerate(self.rnns):
#             x_n = rnn(x_n, state_ln[i])
#             next_state_n_list.append(x_n)
#         x_n = self.output_layer(x_n)
#         next_state_ln = jnp.stack(next_state_n_list, axis=0)
#         return x_n, next_state_ln


# class MLPCell(eqx.Module):
#     num_inputs: int
#     num_hidden: int
#     num_layers: int
#     num_outputs: int

#     layers: list[eqx.nn.Linear]

#     def __init__(
#         self,
#         num_inputs: int,
#         num_hidden: int,
#         num_layers: int,
#         num_outputs: int,
#         *,
#         key: PRNGKeyArray,
#     ) -> None:
#         super().__init__()

#         self.num_inputs = num_inputs
#         self.num_hidden = num_hidden
#         self.num_layers = num_layers
#         self.num_outputs = num_outputs

#         keys = jax.random.split(key, num_layers)

#         self.layers = [
#             eqx.nn.Linear(
#                 in_features=num_inputs if i == 0 else num_hidden,
#                 out_features=num_outputs if i == num_layers else num_hidden,
#                 use_bias=True,
#                 key=keys[i],
#             )
#             for i in range(num_hidden)
#         ]

#     @eqx.filter_jit
#     def __call__(self, x_n: jnp.ndarray) -> jnp.ndarray:
#         for layer in self.layers:
#             x_n = layer(x_n)
#         return x_n


# class ActorModel(eqx.Module):
#     rnn: RNNCell

#     def __init__(
#         self,
#         num_hidden: int,
#         num_layers: int,
#         *,
#         key: PRNGKeyArray,
#     ) -> None:
#         super().__init__()

#         num_joints = 20
#         num_inputs = 2 + 1 + 3 + 3 + num_joints + num_joints
#         num_outputs = num_joints

#         self.rnn = RNNCell(
#             num_inputs=num_inputs,
#             num_hidden=num_hidden,
#             num_layers=num_layers,
#             num_outputs=num_outputs,
#             key=key,
#         )

#     @eqx.filter_jit
#     def __call__(
#         self,
#         lin_vel_cmd_2: jnp.ndarray,  # The XY linear velocity command.
#         ang_vel_cmd_1: jnp.ndarray,  # The Z angular velocity command.
#         imu_acc_3: jnp.ndarray,  # The IMU acceleration.
#         imu_gyro_3: jnp.ndarray,  # The IMU gyroscope.
#         joint_pos_j: jnp.ndarray,  # The joint angular positions.
#         joint_vel_j: jnp.ndarray,  # The joint angular velocities.
#         state_ln: jnp.ndarray | None = None,  # The state of the RNN.
#     ) -> tuple[jnp.ndarray, jnp.ndarray]:
#         x_n = jnp.concatenate(
#             [
#                 lin_vel_cmd_2,
#                 ang_vel_cmd_1,
#                 imu_acc_3,
#                 imu_gyro_3,
#                 joint_pos_j,
#                 joint_vel_j,
#             ],
#             axis=-1,
#         )

#         x_n, next_state_ln = self.rnn(x_n, state_ln)

#         return x_n, next_state_ln


# class CriticModel(eqx.Module):
#     mlp: MLPCell

#     def __init__(
#         self,
#         num_hidden: int,
#         num_layers: int,
#         *,
#         key: PRNGKeyArray,
#     ) -> None:
#         super().__init__()

#         num_joints = 20
#         num_inputs = 2 + 1 + 3 + 3 + 3 + 3 + 3 + 4 + num_joints + num_joints
#         num_outputs = 1

#         self.mlp = MLPCell(
#             num_inputs=num_inputs,
#             num_hidden=num_hidden,
#             num_layers=num_layers,
#             num_outputs=num_outputs,
#             key=key,
#         )

#     @eqx.filter_jit
#     def __call__(
#         self,
#         lin_vel_cmd_2: jnp.ndarray,  # The XY linear velocity command.
#         ang_vel_cmd_1: jnp.ndarray,  # The Z angular velocity command.
#         imu_acc_3: jnp.ndarray,  # The IMU acceleration.
#         imu_gyro_3: jnp.ndarray,  # The IMU gyroscope.
#         base_pos_3: jnp.ndarray,  # The base position.
#         base_ang_vel_3: jnp.ndarray,  # The base angular velocity.
#         base_lin_vel_3: jnp.ndarray,  # The base linear velocity.
#         base_quat_4: jnp.ndarray,  # The base orientation.
#         joint_pos_j: jnp.ndarray,  # The joint angular positions.
#         joint_vel_j: jnp.ndarray,  # The joint angular velocities.
#     ) -> jnp.ndarray:
#         x_n = jnp.concatenate(
#             [
#                 lin_vel_cmd_2,
#                 ang_vel_cmd_1,
#                 imu_acc_3,
#                 imu_gyro_3,
#                 base_pos_3,
#                 base_ang_vel_3,
#                 base_lin_vel_3,
#                 base_quat_4,
#                 joint_pos_j,
#                 joint_vel_j,
#             ],
#             axis=-1,
#         )
#         x_n = self.mlp(x_n)
#         return x_n


# class ActorCriticModel(eqx.Module):
#     actor: ActorModel
#     critic: CriticModel

#     def __init__(self, actor: ActorModel, critic: CriticModel) -> None:
#         super().__init__()

#         self.actor = actor
#         self.critic = critic


class KBotActorModel(ActionModel):
    mlp: MLP

    def setup(self) -> None:
        self.log_std = self.param("log_std", nn.initializers.constant(-0.7), (NUM_OUTPUTS,))

    def input_vec_from_state(self, state: PyTree) -> jax.Array:
        lin_vel_cmd_2 = state["info"]["commands"]["linear_velocity_command"]
        ang_vel_cmd_1 = state["info"]["commands"]["angular_velocity_command"]
        joint_pos_j = state["obs"]["joint_position_observation"]
        joint_vel_j = state["obs"]["joint_velocity_observation"]

        imu_acc_3 = state["obs"]["imu_acc_sensor_observation"]
        imu_gyro_3 = state["obs"]["imu_gyro_sensor_observation"]

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
        return x_n

    def out_from_input_vec(self, x_n: jax.Array) -> jax.Array:
        actions_n = self.mlp(x_n)
        return actions_n

    def __call__(self, state: PyTree) -> jax.Array:
        x_n = self.input_vec_from_state(state)
        actions_n = self.out_from_input_vec(x_n)

        return actions_n

    def calc_log_prob(self, prediction: jax.Array, action: jax.Array) -> jax.Array:
        mean = prediction
        std = jnp.exp(self.log_std)

        log_prob = -0.5 * jnp.square((action - mean) / std) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
        return jnp.sum(log_prob, axis=-1)

    def sample_and_log_prob(self, obs: Array, rng: PRNGKeyArray) -> Tuple[Array, Array]:
        mean = self(obs)
        std = jnp.exp(self.log_std)

        noise = jax.random.normal(rng, mean.shape)
        action = mean + noise * std
        log_prob = self.calc_log_prob(mean, action)

        return action, log_prob


class KBotCriticModel(nn.Module):
    mlp: MLP

    @nn.compact
    def __call__(self, state: PyTree) -> jax.Array:
        lin_vel_cmd_2 = state["info"]["commands"]["linear_velocity_command"]
        ang_vel_cmd_1 = state["info"]["commands"]["angular_velocity_command"]
        joint_pos_j = state["obs"]["joint_position_observation"]
        joint_vel_j = state["obs"]["joint_velocity_observation"]

        base_pos_3 = state["obs"]["base_position_observation"]
        base_ang_vel_3 = state["obs"]["base_angular_velocity_observation"]
        base_lin_vel_3 = state["obs"]["base_linear_velocity_observation"]
        base_quat_4 = state["obs"]["base_orientation_observation"]
        imu_acc_3 = state["obs"]["imu_acc_sensor_observation"]
        imu_gyro_3 = state["obs"]["imu_gyro_sensor_observation"]

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
class KBotWalkingConfig(PPOConfig):
    # Robot model name to use.
    model_name: str = xax.field(value="kbot-v1-feet")

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
                XYPositionResetBuilder(),
            ],
            rewards=[
                LinearVelocityZPenalty(scale=-0.1),
                AngularVelocityXYPenalty(scale=-0.1),
                TrackLinearVelocityXYReward(scale=0.1),
                TrackAngularVelocityZReward(scale=0.1),
                ActionSmoothnessPenalty(scale=-0.1),
                FootContactPenaltyBuilder(
                    scale=-0.1,
                    foot_name="foot1",
                    allowed_contact_prct=0.7,
                    skip_if_zero_command=[
                        "linear_velocity_command",
                        "angular_velocity_command",
                    ],
                ),
                FootContactPenaltyBuilder(
                    scale=-0.1,
                    foot_name="foot3",
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

    def get_model_obs_from_state(self, state: EnvState) -> PyTree:
        return {
            "obs": state.obs,
            "info": state.info,
        }

    def get_model(self, key: PRNGKeyArray) -> ActorCriticModel:
        return ActorCriticModel(
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

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size


if __name__ == "__main__":
    # python -m examples.kbot.walking train
    KBotWalkingTask.launch(
        KBotWalkingConfig(
            num_envs=32,
            max_trajectory_seconds=10.0,
            robot_model_name="kbot-v1-feet",
        ),
    )
