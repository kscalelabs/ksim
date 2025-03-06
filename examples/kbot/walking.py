"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import attrs
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
from ksim.builders.resets import JointVelocityResetBuilder, XYPositionResetBuilder
from ksim.builders.rewards import (
    ActionSmoothnessPenalty,
    AngularVelocityXYPenalty,
    FootContactPenaltyBuilder,
    HeightReward,
    LinearVelocityZPenalty,
    TrackAngularVelocityZReward,
    TrackLinearVelocityXYReward,
)
from ksim.builders.terminations import (
    IllegalContactTerminationBuilder,
    PitchTooGreatTermination,
    RollTooGreatTermination,
)
from ksim.env.mjx.mjx_env import MjxEnv, MjxEnvConfig
from ksim.model.formulations import ActionModel, ActorCriticAgent, GaussianActionModel
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask

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


class KBotActorModel(GaussianActionModel):
    mlp: MLP
    action_clipping: float = 20.0

    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        lin_vel_cmd_2 = cmd["linear_velocity_command"]
        ang_vel_cmd_1 = cmd["angular_velocity_command"]
        joint_pos_j = obs["joint_position_observation"]
        joint_vel_j = obs["joint_velocity_observation"]
        # imu_acc_3 = obs["imu_acc_sensor_observation"]
        # imu_gyro_3 = obs["imu_gyro_sensor_observation"]

        x_n = jnp.concatenate(
            [
                lin_vel_cmd_2,
                ang_vel_cmd_1,
                # imu_acc_3,
                # imu_gyro_3,
                joint_pos_j,
                joint_vel_j,
            ],
            axis=-1,
        )

        actions_n = self.mlp(x_n)

        actions_n = jnp.clip(actions_n, -self.action_clipping, self.action_clipping)

        return actions_n


class KBotZeroActions(GaussianActionModel):
    mlp: MLP

    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        lin_vel_cmd_2 = cmd["linear_velocity_command"]
        ang_vel_cmd_1 = cmd["angular_velocity_command"]
        joint_pos_j = obs["joint_position_observation"]
        joint_vel_j = obs["joint_velocity_observation"]
        # imu_acc_3 = obs["imu_acc_sensor_observation"]
        # imu_gyro_3 = obs["imu_gyro_sensor_observation"]

        x_n = jnp.concatenate(
            [
                lin_vel_cmd_2,
                ang_vel_cmd_1,
                # imu_acc_3,
                # imu_gyro_3,
                joint_pos_j,
                joint_vel_j,
            ],
            axis=-1,
        )

        actions_n = self.mlp(x_n)

        return jnp.zeros_like(actions_n)

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
        x_n = jnp.concatenate([obs_array for obs_array in obs.values()], axis=-1)
        cmd_n = jnp.concatenate([cmd_array for cmd_array in cmd.values()], axis=-1)
        x_n = jnp.concatenate([x_n, cmd_n], axis=-1)

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
    action_clipping: float = xax.field(value=20.0)

    actuator_type: str = xax.field(value="mit", help="The type of actuator to use.")

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
                # JointVelocityResetBuilder(max_velocity=1.0),
            ],
            rewards=[
                # LinearVelocityZPenalty(scale=-0.1),
                # AngularVelocityXYPenalty(scale=-0.1),
                TrackLinearVelocityXYReward(scale=0.1),
                HeightReward(scale=0.1, height_target=1.5),
                # TrackAngularVelocityZReward(scale=0.1),
                # ActionSmoothnessPenalty(scale=-0.1),
                # FootContactPenaltyBuilder(
                #     scale=-0.1,
                #     foot_body_names=["KB_D_501R_R_LEG_FOOT"],
                #     allowed_contact_prct=0.7,
                #     skip_if_zero_command=[
                #         "linear_velocity_command",
                #         "angular_velocity_command",
                #     ],
                # ),
                # FootContactPenaltyBuilder(
                #     scale=-0.1,
                #     foot_body_names=["KB_D_501L_L_LEG_FOOT"],
                #     allowed_contact_prct=0.7,
                #     skip_if_zero_command=[
                #         "linear_velocity_command",
                #         "angular_velocity_command",
                #     ],
                # ),
            ],
            observations=[
                BaseOrientationObservation(noise_type="gaussian", noise=0.01),
                BaseLinearVelocityObservation(noise_type="gaussian", noise=0.01),
                BaseAngularVelocityObservation(noise_type="gaussian", noise=0.01),
                JointPositionObservation(noise_type="gaussian", noise=0.01),
                JointVelocityObservation(noise_type="gaussian", noise=0.01),
                # SensorObservationBuilder(sensor_name="imu_acc"),  # Sensor has noise already.
                # SensorObservationBuilder(sensor_name="imu_gyro"),  # Sensor has noise already.
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
                init_log_std=-0.7,
                num_outputs=NUM_OUTPUTS,
                action_clipping=self.config.action_clipping,
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
    # python -m examples.kbot.walking action=env
    KBotWalkingTask.launch(
        KBotWalkingConfig(
            num_envs=2048,
            num_steps_per_trajectory=600,
            minibatch_size=1024,
            # num_learning_epochs=10,
            # normalize_advantage=True,
            # obs_norm_alpha=0.01,
        ),
    )
