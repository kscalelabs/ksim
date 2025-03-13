"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

from ksim.actuators import MITPositionActuatorsBuilder
from ksim.commands import AngularVelocityCommand, LinearVelocityCommand
from ksim.env.mjx_env import MjxEnv, MjxEnvConfig
from ksim.model.base import ActorCriticAgent, KSimModule
from ksim.model.distributions import TanhGaussianDistribution
from ksim.model.types import ModelInput
from ksim.normalization import Normalizer, PassThrough, Standardize
from ksim.observation import (
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
from ksim.resets import (
    RandomizeJointPositions,
    RandomizeJointVelocities,
    XYPositionResetBuilder,
)
from ksim.rewards import (
    DHForwardReward,
    DHHealthyReward,
    SinusoidalFeetHeightRewardBuilder,
)
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.terminations import PitchTooGreatTermination, RollTooGreatTermination

NUM_OUTPUTS = 20


class KBot2Actor(eqx.Module, KSimModule):
    """Actor for the walking task."""

    mlp: eqx.nn.MLP
    min_std: float
    max_std: float
    var_scale: float

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=453,  # TODO: use similar pattern when dummy data gets passed in to populate
            out_size=NUM_OUTPUTS * 2,
            width_size=64,
            depth=5,
            key=key,
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
        jax.debug.breakpoint()

        return parametrization


class KBot2Critic(eqx.Module, KSimModule):
    """Critic for the walking task."""

    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=453,  # TODO: is there a nice way of inferring this?
            out_size=1,
            width_size=64,
            depth=5,
            key=key,
        )

    def forward(self, model_input: ModelInput) -> Array:
        obs_vec = jnp.concatenate([v for v in model_input.obs.values()], axis=-1)
        command_vec = jnp.concatenate([v for v in model_input.command.values()], axis=-1)
        x = jnp.concatenate([obs_vec, command_vec], axis=-1)
        return self.mlp(x)


@dataclass
class KBotV2WalkingConfig(PPOConfig, MjxEnvConfig):
    """Config for the KBotV2 walking task."""

    robot_model_path: str = xax.field(value="examples/kbot2/scene.mjcf")
    robot_metadata_path: str = xax.field(value="examples/kbot2/metadata.json")


class KBotV2WalkingTask(PPOTask[KBotV2WalkingConfig]):
    def get_environment(self) -> MjxEnv:
        return MjxEnv(
            self.config,
            robot_dir_path=self.config.robot_model_name,
            actuators=MITPositionActuatorsBuilder(),
            terminations=[
                RollTooGreatTermination(max_roll=0.3),
                PitchTooGreatTermination(max_pitch=0.3),
            ],
            resets=[
                XYPositionResetBuilder(),
                RandomizeJointVelocities(scale=0.01),
                RandomizeJointPositions(scale=0.01),
            ],
            rewards=[
                DHHealthyReward(
                    scale=0.5,
                ),
                DHForwardReward(scale=0.2),
                SinusoidalFeetHeightRewardBuilder(
                    left_foot_geom_name="KB_D_501L_L_LEG_FOOT_box_collision",
                    right_foot_geom_name="KB_D_501R_R_LEG_FOOT_box_collision",
                    amplitude=0.1,
                    period=0.4,
                    scale=2.0,
                    vertical_offset=-0.09,
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
        return ActorCriticAgent(
            critic_model=KBot2Critic(key),
            actor_model=KBot2Actor(key, min_std=0.01, max_std=1.0, var_scale=1.0),
            action_distribution=TanhGaussianDistribution(action_dim=NUM_OUTPUTS),
        )

    def get_obs_normalizer(self, dummy_obs: FrozenDict[str, Array]) -> Normalizer:
        return Standardize()

    def get_cmd_normalizer(self, dummy_cmd: FrozenDict[str, Array]) -> Normalizer:
        return PassThrough()

    def get_init_actor_carry(self) -> jnp.ndarray | None:
        return None

    def get_init_critic_carry(self) -> None:
        return None


if __name__ == "__main__":
    # python -m examples.kbot.walking action=env
    KBotV2WalkingTask.launch(
        KBotV2WalkingConfig(
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
