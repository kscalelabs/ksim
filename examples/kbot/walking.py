"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from brax.base import System
from brax.envs.base import State as BraxState
from jaxtyping import PRNGKeyArray, PyTree

from ksim.commands import AngularVelocityCommand, LinearVelocityCommand
from ksim.env.brax import KScaleEnv
from ksim.observation.mjcf import (
    BaseAngularVelocityObservation,
    BaseLinearVelocityObservation,
    BaseOrientationObservation,
    BasePositionObservation,
    JointPositionObservation,
    JointVelocityObservation,
)
from ksim.resets.mjcf import XYPositionReset
from ksim.rewards.mjcf import LinearVelocityZPenalty
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.terminations import IllegalContactTerminationBuilder


class Model(eqx.Module):
    input_layer: eqx.nn.Linear
    rnn: eqx.nn.GRUCell
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_hidden: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        # Split the PRNG key into four keys for the four layers.
        key1, key2, key3 = jax.random.split(key, 3)

        self.input_layer = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=num_hidden,
            use_bias=True,
            key=key1,
        )

        self.rnn = eqx.nn.GRUCell(
            input_size=num_hidden,
            hidden_size=num_hidden,
            use_bias=True,
            key=key2,
        )

        self.output_layer = eqx.nn.Linear(
            in_features=num_hidden,
            out_features=num_outputs,
            use_bias=True,
            key=key3,
        )

    def __call__(self, x_tn: jnp.ndarray) -> jnp.ndarray:
        x_tn = self.input_layer(x_tn)

        def scan_fn(state_n: jnp.ndarray, x_n: jnp.ndarray) -> tuple[jnp.ndarray, None]:
            state_n = self.rnn(x_n, state_n)
            return state_n, None

        init_state_n = jnp.zeros(self.rnn.hidden_size)
        x_tn, _ = jax.lax.scan(scan_fn, init_state_n, x_tn)

        x_tn = self.output_layer(x_tn)
        return x_tn


class ActorCriticModel(eqx.Module):
    actor: Model
    critic: Model

    def __init__(self, num_hidden: int = 512, *, key: PRNGKeyArray) -> None:
        super().__init__()

        actor_key, critic_key = jax.random.split(key, 2)

        self.actor = Model(
            num_inputs=32,
            num_outputs=32,
            num_hidden=num_hidden,
            key=actor_key,
        )

        self.critic = Model(
            num_inputs=32,
            num_outputs=1,
            num_hidden=num_hidden,
            key=critic_key,
        )


@dataclass
class KBotWalkingConfig(PPOConfig):
    # Robot model name to use.
    model_name: str = xax.field(value="kbot-v1")
    kp: float = xax.field(value=100.0)
    kd: float = xax.field(value=10.0)

    # ML model parameters.
    actor_hidden_dims: list[int] = xax.field(value=[512, 256, 128])
    critic_hidden_dims: list[int] = xax.field(value=[512, 256, 128])
    init_noise_std: float = xax.field(value=1.0)

    # Termination conditions.
    max_episode_length: float = xax.field(value=20.0)
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
                XYPositionReset(x_range=(-0.5, 0.5), y_range=(-0.5, 0.5)),
            ],
            rewards=[
                LinearVelocityZPenalty(scale=-1.0),
            ],
            observations=[
                BasePositionObservation(noise=0.01),
                BaseOrientationObservation(noise=0.01),
                BaseLinearVelocityObservation(noise=0.01),
                BaseAngularVelocityObservation(noise=0.01),
                JointPositionObservation(noise=0.01),
                JointVelocityObservation(noise=0.01),
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
        return ActorCriticModel(key=key)

    def get_actor_output(
        self,
        model: ActorCriticModel,
        sys: System,
        state: BraxState,
        rng: PRNGKeyArray,
        carry: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        pos = state.obs['base_position_observation']
        ori = state.obs['base_orientation_observation']
        lin_vel = state.obs['base_linear_velocity_observation']
        ang_vel = state.obs['base_angular_velocity_observation']
        joint_pos = state.obs['joint_position_observation']
        joint_vel = state.obs['joint_velocity_observation']

        breakpoint()

        raise NotImplementedError


if __name__ == "__main__":
    # python -m examples.kbot.walking train
    KBotWalkingTask.launch(
        KBotWalkingConfig(
            num_envs=32,
            max_trajectory_seconds=10.0,
        ),
    )
