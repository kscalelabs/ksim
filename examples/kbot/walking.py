"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from brax.base import System
from brax.envs.base import State as BraxState
from jaxtyping import PRNGKeyArray

from ksim.commands import AngularVelocityCommand, LinearVelocityCommand
from ksim.env.brax import KScaleEnv
from ksim.observation import (
    BaseAngularVelocityObservation,
    BaseLinearVelocityObservation,
    BaseOrientationObservation,
    BasePositionObservation,
    JointPositionObservation,
    JointVelocityObservation,
    SensorObservationBuilder,
)
from ksim.resets import XYPositionResetBuilder
from ksim.rewards import (
    AngularVelocityXYPenalty,
    FootContactPenaltyBuilder,
    FootSlipPenaltyBuilder,
    LinearVelocityZPenalty,
    TrackAngularVelocityZReward,
    TrackLinearVelocityXYReward,
)
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.terminations import IllegalContactTerminationBuilder


class RNNCell(eqx.Module):
    num_inputs: int
    num_hidden: int
    num_layers: int
    num_outputs: int

    rnns: list[eqx.nn.GRUCell]
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        num_layers: int,
        num_outputs: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_outputs = num_outputs

        keys = jax.random.split(key, num_layers + 1)

        self.rnns = [
            eqx.nn.GRUCell(
                input_size=num_inputs if i == 0 else num_hidden,
                hidden_size=num_hidden,
                use_bias=True,
                key=keys[i],
            )
            for i in range(num_layers)
        ]

        self.output_layer = eqx.nn.Linear(
            in_features=num_hidden,
            out_features=num_outputs,
            use_bias=True,
            key=keys[-1],
        )

    def __call__(
        self,
        x_n: jnp.ndarray,
        state_ln: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        next_state_n_list = []
        if state_ln is None:
            state_ln = jnp.zeros((self.num_layers, self.num_hidden))
        for i, rnn in enumerate(self.rnns):
            x_n = rnn(x_n, state_ln[i])
            next_state_n_list.append(x_n)
        x_n = self.output_layer(x_n)
        next_state_ln = jnp.stack(next_state_n_list, axis=0)
        return x_n, next_state_ln


class ActorModel(eqx.Module):
    rnn: RNNCell

    def __init__(
        self,
        num_hidden: int,
        num_layers: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        num_joints = 20
        num_inputs = 2 + 1 + 3 + num_joints + num_joints
        num_outputs = num_joints

        self.rnn = RNNCell(
            num_inputs=num_inputs,
            num_hidden=num_hidden,
            num_layers=num_layers,
            num_outputs=num_outputs,
            key=key,
        )

    def __call__(
        self,
        lin_vel_cmd_2: jnp.ndarray,  # The XY linear velocity command.
        ang_vel_cmd_1: jnp.ndarray,  # The Z angular velocity command.
        proj_grav_3: jnp.ndarray,  # The projected gravity vector.
        joint_pos_j: jnp.ndarray,  # The joint angular positions.
        joint_vel_j: jnp.ndarray,  # The joint angular velocities.
        state_ln: jnp.ndarray | None = None,  # The state of the RNN.
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x_n = jnp.concatenate(
            [
                lin_vel_cmd_2,
                ang_vel_cmd_1,
                proj_grav_3,
                joint_pos_j,
                joint_vel_j,
            ],
            axis=-1,
        )

        x_n, next_state_ln = self.rnn(x_n, state_ln)

        return x_n, next_state_ln


class CriticModel(eqx.Module):
    rnn: RNNCell

    def __init__(
        self,
        num_hidden: int,
        num_layers: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        num_joints = 20
        num_inputs = 2 + 1 + 3 + num_joints + num_joints
        num_outputs = num_joints

        self.rnn = RNNCell(
            num_inputs=num_inputs,
            num_hidden=num_hidden,
            num_layers=num_layers,
            num_outputs=num_outputs,
            key=key,
        )

    def __call__(
        self,
        lin_vel_cmd_2: jnp.ndarray,  # The XY linear velocity command.
        ang_vel_cmd_1: jnp.ndarray,  # The Z angular velocity command.
        proj_grav_3: jnp.ndarray,  # The projected gravity vector.
        joint_pos_j: jnp.ndarray,  # The joint angular positions.
        joint_vel_j: jnp.ndarray,  # The joint angular velocities.
        state_ln: jnp.ndarray | None = None,  # The state of the RNN.
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x_n = jnp.concatenate(
            [
                lin_vel_cmd_2,
                ang_vel_cmd_1,
                proj_grav_3,
                joint_pos_j,
                joint_vel_j,
            ],
            axis=-1,
        )

        x_n, next_state_ln = self.rnn(x_n, state_ln)

        return x_n, next_state_ln


class ActorCriticModel(eqx.Module):
    actor: ActorModel
    critic: CriticModel

    def __init__(self, actor: ActorModel, critic: CriticModel) -> None:
        super().__init__()

        self.actor = actor
        self.critic = critic


@dataclass
class KBotWalkingConfig(PPOConfig):
    # Robot model name to use.
    model_name: str = xax.field(value="kbot-v1-feet")

    # ML model parameters.
    actor_hidden_dims: int = xax.field(value=512)
    actor_num_layers: int = xax.field(value=2)
    critic_hidden_dims: int = xax.field(value=512)
    critic_num_layers: int = xax.field(value=2)
    init_noise_std: float = xax.field(value=1.0)

    # Termination conditions.
    max_episode_length: float = xax.field(value=10.0)
    max_pitch: float = xax.field(value=0.1)
    max_roll: float = xax.field(value=0.1)


class KBotWalkingTask(PPOTask[KBotWalkingConfig]):
    def get_environment(self) -> KScaleEnv:
        return KScaleEnv(
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
                FootSlipPenaltyBuilder(scale=0.1, foot_names=["foot1", "foot3"]),
                FootContactPenaltyBuilder(scale=0.1, foot_names=["foot1", "foot3"]),
            ],
            observations=[
                BasePositionObservation(noise=0.01),
                BaseOrientationObservation(noise=0.01),
                BaseLinearVelocityObservation(noise=0.01),
                BaseAngularVelocityObservation(noise=0.01),
                JointPositionObservation(noise=0.01),
                JointVelocityObservation(noise=0.01),
                SensorObservationBuilder(sensor_name="imu_acc", noise=0.01),
                SensorObservationBuilder(sensor_name="imu_gyro", noise=0.01),
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

    def get_model(self, key: PRNGKeyArray) -> ActorCriticModel:
        return ActorCriticModel(
            actor=ActorModel(
                num_hidden=self.config.actor_hidden_dims,
                num_layers=self.config.actor_num_layers,
                key=key,
            ),
            critic=CriticModel(
                num_hidden=self.config.critic_hidden_dims,
                num_layers=self.config.critic_num_layers,
                key=key,
            ),
        )

    def get_init_carry(self) -> jnp.ndarray:
        return jnp.zeros((self.config.actor_num_layers, self.config.actor_hidden_dims))

    def get_actor_output(
        self,
        model: ActorCriticModel,
        sys: System,
        state: BraxState,
        rng: PRNGKeyArray,
        carry: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        breakpoint()

        lin_vel_cmd_2 = state.info["commands"]["linear_velocity_command"]
        ang_vel_cmd_1 = state.info["commands"]["angular_velocity_command"]
        joint_pos_j = state.obs["joint_position_observation"]
        joint_vel_j = state.obs["joint_velocity_observation"]

        actions_n, next_carry = model.actor(
            lin_vel_cmd_2=lin_vel_cmd_2,
            ang_vel_cmd_1=ang_vel_cmd_1,
            proj_grav_3=jnp.array([0.0, 0.0, -9.81]),
            joint_pos_j=joint_pos_j,
            joint_vel_j=joint_vel_j,
            state_ln=carry,
        )

        return actions_n, next_carry


if __name__ == "__main__":
    # python -m examples.kbot.walking train
    KBotWalkingTask.launch(
        KBotWalkingConfig(
            num_envs=32,
            max_trajectory_seconds=10.0,
        ),
    )
