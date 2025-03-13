"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp
import xax
from jaxtyping import PRNGKeyArray

from ksim.commands import AngularVelocityCommand, LinearVelocityCommand
from ksim.env.mjx_env import MjxEnv, MjxEnvConfig
from ksim.model.base import ActorCriticAgent
from ksim.model.factory import mlp_actor_critic_agent
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
from ksim.rewards import DHForwardReward, DHHealthyReward
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.terminations import PitchTooGreatTermination, RollTooGreatTermination

######################
# Static Definitions #
######################

NUM_OUTPUTS = 20


@dataclass
class KBotV2WalkingConfig(PPOConfig, MjxEnvConfig):
    """Config for the KBotV2 walking task."""

    robot_model_name: str = xax.field(value="examples/kbot2/")


class KBotV2WalkingTask(PPOTask[KBotV2WalkingConfig]):
    def get_environment(self) -> MjxEnv:
        return MjxEnv(
            self.config,
            robot_dir_path=self.config.robot_model_name,
            actuators=Actuators(),
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
                LinearVelocityCommand(x_scale=1.0, y_scale=0.0, switch_prob=0.02, zero_prob=0.3),
                AngularVelocityCommand(scale=1.0, switch_prob=0.02, zero_prob=0.8),
            ],
        )

    def get_model(self, key: PRNGKeyArray) -> ActorCriticAgent:
        return mlp_actor_critic_agent(
            num_actions=NUM_OUTPUTS,
            prediction_type="mean_std",
            distribution_type="tanh_gaussian",
            actor_hidden_dims=(64,) * 5,
            critic_hidden_dims=(64,) * 5,
            kernel_initialization=nn.initializers.lecun_normal(),
            post_process_kwargs={"min_std": 0.01, "max_std": 1.0, "var_scale": 1.0},
        )

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
