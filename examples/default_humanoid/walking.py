"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

from ksim.builders.commands import LinearVelocityCommand
from ksim.builders.observation import (
    BaseAngularVelocityObservation,
    BaseLinearVelocityObservation,
    BaseOrientationObservation,
    JointPositionObservation,
    JointVelocityObservation,
)
from ksim.builders.resets import (
    RandomizeJointPositions,
    RandomizeJointVelocities,
)
from ksim.builders.rewards import DHControlPenalty, DHForwardReward, DHHealthyReward
from ksim.builders.terminations import UnhealthyTermination
from ksim.env.mjx.mjx_env import MjxEnv, MjxEnvConfig
from ksim.model.formulations import ActorCriticAgent, GaussianActionModel
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask

NUM_OUTPUTS = 21


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
    actor_num_layers: int = xax.field(value=5)
    critic_hidden_dims: int = xax.field(value=512)
    critic_num_layers: int = xax.field(value=5)

    # Termination conditions.
    max_pitch: float = xax.field(value=0.1)
    max_roll: float = xax.field(value=0.1)
    pretrained: str | None = xax.field(value=None)
    checkpoint_num: int | None = xax.field(value=None)

    actuator_type: str = xax.field(value="scaled_torque", help="The type of actuator to use.")


class HumanoidWalkingTask(PPOTask[HumanoidWalkingConfig]):
    def get_environment(self) -> MjxEnv:
        return MjxEnv(
            self.config,
            terminations=[
                UnhealthyTermination(
                    unhealthy_z_lower=0.5,
                    unhealthy_z_upper=1.5,
                ),
            ],
            resets=[
                RandomizeJointPositions(scale=0.01),
                RandomizeJointVelocities(scale=0.01),
            ],
            rewards=[
                DHForwardReward(scale=0.125),
                DHHealthyReward(
                    scale=0.5,
                    healthy_z_lower=0.5,
                    healthy_z_upper=1.5,
                ),
                DHControlPenalty(scale=0.01),
            ],
            observations=[
                BaseOrientationObservation(noise_type="gaussian", noise=0.01),
                BaseLinearVelocityObservation(noise_type="gaussian", noise=0.01),
                BaseAngularVelocityObservation(noise_type="gaussian", noise=0.01),
                JointPositionObservation(noise_type="gaussian", noise=0.01),
                JointVelocityObservation(noise_type="gaussian", noise=0.01),
                # TODO: default humanoid doesn't have sensors, add them later
                # Legacy Ksim observation setup
                # LegacyPositionObservation(exclude_xy=True),
                # LegacyVelocityObservation(),
                # CenterOfMassInertiaObservation(),
                # CenterOfMassVelocityObservation(),
                # ActuatorForceObservation(),
            ],
            commands=[
                LinearVelocityCommand(
                    x_scale=0.0,
                    y_scale=0.0,
                    switch_prob=0.02,
                    zero_prob=0.3,
                ),
                # AngularVelocityCommand(
                #     scale=1.0,
                #     switch_prob=0.02,
                #     zero_prob=0.8,
                # ),
            ],
        )

    def get_model(self, key: PRNGKeyArray) -> ActorCriticAgent:
        return ActorCriticAgent(
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
    # python -m examples.default_humanoid.walking
    HumanoidWalkingTask.launch(
        HumanoidWalkingConfig(
            num_learning_epochs=8,
            num_env_states_per_minibatch=8192,
            num_minibatches=32,
            num_envs=2048,
            dt=0.005,
            ctrl_dt=0.02,
            learning_rate=0.00001,
            save_every_n_steps=50,
            only_save_most_recent=False,
            reward_scaling_alpha=0.0,
            obs_norm_alpha=0.0,
            # ksim-legacy original setup was dt=0.003 and ctrl_dt=0.012 ~ 83.33 hz
            solver_iterations=6,
            solver_ls_iterations=6,
            actuator_type="scaled_torque",
            scale_rewards=True,
            gamma=0.97,
            lam=0.95,
            normalize_advantage=True,
            normalize_advantage_in_minibatch=True,
            entropy_coef=1e-4,
            actor_num_layers=5,
            critic_num_layers=5,
            clip_param=0.3,
        ),
    )
