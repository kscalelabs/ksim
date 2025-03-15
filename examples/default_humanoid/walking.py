"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

from ksim.actuators import TorqueActuators
from ksim.commands import LinearVelocityCommand
from ksim.env.mjx_engine import MjxEnv, MjxEnvConfig
from ksim.model.base import ActorCriticAgent, KSimModule
from ksim.model.distributions import TanhGaussianDistribution
from ksim.model.types import ModelInput
from ksim.normalization import Normalizer, PassThrough, Standardize
from ksim.observation import (
    ActuatorForceObservation,
    CenterOfMassInertiaObservation,
    CenterOfMassVelocityObservation,
    LegacyPositionObservation,
    LegacyVelocityObservation,
)
from ksim.resets import RandomizeJointPositions, RandomizeJointVelocities
from ksim.rewards import DHForwardReward, DHHealthyReward
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.terminations import UnhealthyTermination

NUM_OUTPUTS = 21


class DefaultHumanoidActor(eqx.Module, KSimModule):
    """Actor for the walking task."""

    mlp: eqx.nn.MLP
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=338,  # TODO: use similar pattern when dummy data gets passed in to populate
            out_size=NUM_OUTPUTS * 2,
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, model_input: ModelInput) -> Array:
        obs_vec = jnp.concatenate([v for v in model_input.obs.values()], axis=-1)
        command_vec = jnp.concatenate([v for v in model_input.command.values()], axis=-1)
        x = jnp.concatenate([obs_vec, command_vec], axis=-1)
        prediction = self.mlp(x)

        mean = prediction[..., :NUM_OUTPUTS]
        std = prediction[..., NUM_OUTPUTS:]

        # softplus and clipping for stability
        std = (jax.nn.softplus(std) + self.min_std) * self.var_scale
        std = jnp.clip(std, self.min_std, self.max_std)

        # concat because Gaussian-like distributions expect the parameters
        # to be mean concat std
        parametrization = jnp.concatenate([mean, std], axis=-1)

        return parametrization


class DefaultHumanoidCritic(eqx.Module, KSimModule):
    """Critic for the walking task."""

    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=338,  # TODO: is there a nice way of inferring this?
            out_size=1,
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )

    def forward(self, model_input: ModelInput) -> Array:
        obs_vec = jnp.concatenate([v for v in model_input.obs.values()], axis=-1)
        command_vec = jnp.concatenate([v for v in model_input.command.values()], axis=-1)
        x = jnp.concatenate([obs_vec, command_vec], axis=-1)
        return self.mlp(x)


@dataclass
class HumanoidWalkingConfig(PPOConfig, MjxEnvConfig):
    """Combining configs for the humanoid walking task and fixing params."""

    robot_model_path: str = "examples/default_humanoid/scene.mjcf"


class HumanoidWalkingTask(PPOTask[HumanoidWalkingConfig]):
    def get_environment(self) -> MjxEnv:
        return MjxEnv(
            self.config,
            robot_model_path=self.config.robot_model_path,
            robot_metadata_path=None,
            actuators=TorqueActuators(),
            terminations=[
                UnhealthyTermination(
                    unhealthy_z_lower=0.8,
                    unhealthy_z_upper=2.0,
                ),
            ],
            resets=[
                RandomizeJointPositions(scale=0.01),
                RandomizeJointVelocities(scale=0.01),
            ],
            rewards=[
                DHHealthyReward(
                    scale=0.5,
                ),
                DHForwardReward(scale=0.125),
            ],
            observations=[
                LegacyPositionObservation(exclude_xy=True),
                LegacyVelocityObservation(),
                CenterOfMassInertiaObservation(),
                CenterOfMassVelocityObservation(),
                ActuatorForceObservation(),
            ],
            commands=[
                LinearVelocityCommand(x_scale=0.0, y_scale=0.0, switch_prob=0.02, zero_prob=0.3),
            ],
        )

    def get_model(self, key: PRNGKeyArray) -> ActorCriticAgent:
        return ActorCriticAgent(
            critic_model=DefaultHumanoidCritic(key),
            actor_model=DefaultHumanoidActor(key, min_std=0.01, max_std=1.0, var_scale=1.0),
            action_distribution=TanhGaussianDistribution(action_dim=NUM_OUTPUTS),
        )

    def get_obs_normalizer(self, dummy_obs: FrozenDict[str, Array]) -> Normalizer:
        return Standardize(dummy_obs, alpha=1.0)

    def get_cmd_normalizer(self, dummy_cmd: FrozenDict[str, Array]) -> Normalizer:
        return PassThrough()

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
            learning_rate=1e-5,
            save_every_n_steps=25,
            only_save_most_recent=False,
            reward_scaling_alpha=0.0,
            obs_norm_alpha=0.0,
            # ksim-legacy original setup was dt=0.003 and ctrl_dt=0.012 ~ 83.33 hz
            solver_iterations=6,
            solver_ls_iterations=6,
            scale_rewards=False,
            gamma=0.97,
            lam=0.95,
            normalize_advantage=True,
            normalize_advantage_in_minibatch=True,
            entropy_coef=0.001,
            clip_param=0.3,
            use_clipped_value_loss=False,
            max_grad_norm=1.0,
            max_action_latency=0.0,
            min_action_latency=0.0,
            eval_rollout_length=1000,
        ),
    )
