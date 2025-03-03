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
)
from ksim.builders.resets import XYPositionResetBuilder
from ksim.builders.rewards import (
    ActionSmoothnessPenalty,
    AngularVelocityXYPenalty,
    FootContactPenaltyBuilder,
    LinearVelocityZPenalty,
    TrackAngularVelocityZReward,
    TrackLinearVelocityXYReward,
)
from ksim.builders.terminations import MinimumHeightTermination
from ksim.env.mjx.mjx_env import MjxEnv, MjxEnvConfig
from ksim.model.formulations import ActorCriticModel, GaussianActionModel
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask

NUM_INPUTS = 24
NUM_OUTPUTS = 17


class HumanoidActorModel(GaussianActionModel):
    mlp: MLP

    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        x_n = jnp.concatenate([obs_array for obs_array in obs.values()], axis=-1)
        cmd_n = jnp.concatenate([cmd_array for cmd_array in cmd.values()], axis=-1)
        x_n = jnp.concatenate([x_n, cmd_n], axis=-1)
        actions_n = self.mlp(x_n)

        return actions_n


class HumanoidCriticModel(nn.Module):
    mlp: MLP

    @nn.compact
    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> jax.Array:
        x_n = jnp.concatenate([obs_array for obs_array in obs.values()], axis=-1)
        cmd_n = jnp.concatenate([cmd_array for cmd_array in cmd.values()], axis=-1)
        x_n = jnp.concatenate([x_n, cmd_n], axis=-1)
        value_estimate = self.mlp(x_n)

        return value_estimate


@dataclass
class HumanoidWalkingConfig(PPOConfig, MjxEnvConfig):
    # Robot model name to use.
    robot_model_name: str = xax.field(value="examples/default_humanoid/")

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


class HumanoidWalkingTask(PPOTask[HumanoidWalkingConfig]):
    def get_environment(self) -> MjxEnv:
        return MjxEnv(
            self.config,
            terminations=[
                MinimumHeightTermination(min_height=0.4),
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
                # TODO: default humanoid doesn't have sensors, add them later
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
            actor_module=HumanoidActorModel(
                mlp=MLP(
                    num_hidden_layers=self.config.actor_num_layers,
                    hidden_features=self.config.actor_hidden_dims,
                    out_features=NUM_OUTPUTS,
                ),
                init_log_std=-0.7,
                num_outputs=NUM_OUTPUTS,
            ),
            critic_module=HumanoidCriticModel(
                mlp=MLP(
                    num_hidden_layers=self.config.critic_num_layers,
                    hidden_features=self.config.critic_hidden_dims,
                    out_features=1,
                ),
            ),
        )

    def get_init_actor_carry(self) -> jnp.ndarray | None:
        return None

    def get_init_critic_carry(self) -> None:
        return None


if __name__ == "__main__":
    # python -m examples.default_humanoid.walking action=train
    HumanoidWalkingTask.launch(
        HumanoidWalkingConfig(
            num_envs=32,
            num_steps_per_trajectory=500,
            minibatch_size=500 * 4,
        ),
    )
