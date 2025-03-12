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
    ActuatorForceObservation,
    BaseAngularVelocityObservation,
    BaseLinearVelocityObservation,
    BaseOrientationObservation,
    BasePositionObservation,
    CenterOfMassInertiaObservation,
    CenterOfMassVelocityObservation,
    JointPositionObservation,
    JointVelocityObservation,
    LegacyPositionObservation,
    LegacyVelocityObservation,
)
from ksim.builders.resets import RandomizeJointPositions, RandomizeJointVelocities
from ksim.builders.rewards import (
    DHControlPenalty,
    DHForwardReward,
    DHHealthyReward,
    DHTerminationPenalty,
)
from ksim.builders.terminations import UnhealthyTermination
from ksim.env.mjx.mjx_env import MjxEnv, MjxEnvConfig
from ksim.model.distributions import GaussianDistribution, TanhGaussianDistribution
from ksim.model.formulations import ActorCriticAgent, ActorModel
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask

######################
# Static Definitions #
######################

NUM_OUTPUTS = 21


@dataclass
class HumanoidWalkingConfig(PPOConfig, MjxEnvConfig):
    """Combining configs for the humanoid walking task and fixing params."""

    robot_model_name: str = "examples/default_humanoid/"
    actuator_type: str = "scaled_torque"


#####################
# Model Definitions #
#####################


class DefaultHumanoidActor(ActorModel):
    """Default humanoid actor."""

    underlying_actor: MLP
    std_range: tuple[float, float] = (0.1, 0.9)
    std_init: float = 0.3

    @nn.compact
    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        """Forward pass of the actor model."""
        # x = jnp.concatenate(list(obs.values()) + list(cmd.values()), axis=-1)
        x = jnp.concatenate(list(obs.values()), axis=-1)
        mean = self.underlying_actor(x)

        # using a single std for all actions for stability
        std = self.param(
            "std",
            nn.initializers.constant(self.std_init),
            (NUM_OUTPUTS,),
        )
        std = jnp.clip(std, self.std_range[0], self.std_range[1])
        std = jnp.tile(std, (*mean.shape[:-1], 1))

        # concatting because Gaussian-like distributions expect the parameters
        # to be mean concat std
        res = jnp.concatenate([mean, std], axis=-1)
        return res


class DefaultHumanoidLearnedStdActor(ActorModel):
    """Default humanoid actor with learned std."""

    underlying_actor: MLP

    @nn.compact
    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        """Forward pass of the actor model."""
        x = jnp.concatenate(list(obs.values()), axis=-1)
        return self.underlying_actor(x)


class DefaultHumanoidCritic(nn.Module):
    """Default humanoid critic."""

    underlying_critic: MLP

    @nn.compact
    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        """Forward pass of the critic model."""
        # x = jnp.concatenate(list(obs.values()) + list(cmd.values()), axis=-1)
        x = jnp.concatenate(list(obs.values()), axis=-1)
        res = self.underlying_critic(x)
        return res


class DefaultHumanoidAgent(ActorCriticAgent):
    """Default humanoid agent."""

    actor_module: DefaultHumanoidActor | DefaultHumanoidLearnedStdActor
    critic_module: DefaultHumanoidCritic

    def actor_obs(self, obs: FrozenDict[str, Array]) -> FrozenDict[str, Array]:
        """Sees all except base pos."""
        return obs

    def critic_obs(self, obs: FrozenDict[str, Array]) -> FrozenDict[str, Array]:
        """Full pass through."""
        return obs


##########################
# Experiment Definitions #
##########################


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
                DHForwardReward(scale=0.125),
                DHHealthyReward(
                    scale=0.5,
                ),
                # DHTerminationPenalty(
                #     scale=-2.0,
                #     healthy_z_lower=0.5,
                #     healthy_z_upper=1.5,
                # ),
                # DHControlPenalty(scale=-0.001),
            ],
            observations=[
                # JointPositionObservation(noise_type="gaussian", noise=0.01),
                # JointVelocityObservation(noise_type="gaussian", noise=0.01),
                # BaseOrientationObservation(noise_type="gaussian", noise=0.01),
                # BaseAngularVelocityObservation(noise_type="gaussian", noise=0.01),
                # BaseLinearVelocityObservation(noise_type="gaussian", noise=0.01),
                # BasePositionObservation(noise_type="gaussian", noise=0.01),
                # TODO: default humanoid doesn't have sensors, add them later
                # Legacy Ksim observation setup
                LegacyPositionObservation(exclude_xy=True),
                LegacyVelocityObservation(),
                CenterOfMassInertiaObservation(),
                CenterOfMassVelocityObservation(),
                ActuatorForceObservation(),
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
        return DefaultHumanoidAgent(
            actor_module=DefaultHumanoidActor(
                MLP(
                    out_dim=NUM_OUTPUTS,
                    hidden_dims=(64,) * 5,
                    activation=nn.relu,
                    bias_init=nn.initializers.zeros,
                ),
                std_range=(0.1, 0.9),
                std_init=0.3,
            ),
            # actor_module=DefaultHumanoidLearnedStdActor(
            #     MLP(
            #         out_dim=NUM_OUTPUTS * 2,
            #         hidden_dims=(64,) * 5,
            #         activation=nn.relu,
            #         bias_init=nn.initializers.zeros,
            #     ),
            # ),
            critic_module=DefaultHumanoidCritic(
                MLP(
                    out_dim=1,
                    hidden_dims=(64,) * 5,
                    activation=nn.relu,
                    bias_init=nn.initializers.zeros,
                ),
            ),
            distribution=GaussianDistribution(action_dim=NUM_OUTPUTS),
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
            learning_rate=0.00005,
            save_every_n_steps=50,
            only_save_most_recent=False,
            reward_scaling_alpha=0.0,
            obs_norm_alpha=0.0,
            # ksim-legacy original setup was dt=0.003 and ctrl_dt=0.012 ~ 83.33 hz
            solver_iterations=6,
            solver_ls_iterations=6,
            actuator_type="scaled_torque",
            scale_rewards=False,
            gamma=0.97,
            lam=0.95,
            normalize_advantage=True,
            normalize_advantage_in_minibatch=True,
            entropy_coef=0.001,
            clip_param=0.3,
            use_clipped_value_loss=False,
        ),
    )
