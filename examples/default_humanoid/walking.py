"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import flax.linen as nn
import jax.numpy as jnp
import xax
from jaxtyping import PRNGKeyArray

from ksim.builders.commands import LinearVelocityCommand
from ksim.builders.observation import (
    ActuatorForceObservation,
    CenterOfMassInertiaObservation,
    CenterOfMassVelocityObservation,
    LegacyPositionObservation,
    LegacyVelocityObservation,
)
from ksim.builders.resets import RandomizeJointPositions, RandomizeJointVelocities
from ksim.builders.rewards import DHForwardReward, DHHealthyReward
from ksim.builders.terminations import UnhealthyTermination
from ksim.env.mjx.mjx_env import MjxEnv, MjxEnvConfig
from ksim.model.base import ActorCriticAgent
from ksim.model.factory import mlp_actor_critic_agent
from ksim.task.ppo import PPOConfig, PPOTask

######################
# Static Definitions #
######################

NUM_OUTPUTS = 21


@dataclass
class HumanoidWalkingConfig(PPOConfig, MjxEnvConfig):
    """Combining configs for the humanoid walking task and fixing params."""

    robot_model_name: str = "examples/default_humanoid/"


####################
# Task Definitions #
####################


class HumanoidWalkingTask(PPOTask[HumanoidWalkingConfig]):
    def get_environment(self) -> MjxEnv:
        return MjxEnv(
            self.config,
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
            save_every_n_steps=50,
            only_save_most_recent=False,
            reward_scaling_alpha=0.0,
            obs_norm_alpha=0.0,
            # ksim-legacy original setup was dt=0.003 and ctrl_dt=0.012 ~ 83.33 hz
            solver_iterations=6,
            solver_ls_iterations=6,
            actuator_type="mit",
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
        ),
    )
