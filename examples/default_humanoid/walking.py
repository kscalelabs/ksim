"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import mujoco
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
)
from ksim.builders.resets import XYPositionResetBuilder
from ksim.builders.rewards import (
    HeightReward,
    TrackLinearVelocityXYReward,
)
from ksim.builders.terminations import (
    PitchTooGreatTermination,
    RollTooGreatTermination,
)
from ksim.env.mjx.mjx_env import MjxEnv, MjxEnvConfig
from ksim.model.formulations import ActionModel, ActorCriticAgent, GaussianActionModel
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask

NUM_OUTPUTS = 17


class HumanoidActorModel(GaussianActionModel):
    mlp: MLP

    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        x_n = jnp.concatenate([obs_array for obs_array in obs.values()], axis=-1)
        cmd_n = jnp.concatenate([cmd_array for cmd_array in cmd.values()], axis=-1)
        x_n = jnp.concatenate([x_n, cmd_n], axis=-1)
        actions_n = self.mlp(x_n)
        return actions_n


class HumanoidZeroActions(GaussianActionModel):
    mlp: MLP

    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        x_n = jnp.concatenate([obs_array for obs_array in obs.values()], axis=-1)
        cmd_n = jnp.concatenate([cmd_array for cmd_array in cmd.values()], axis=-1)
        x_n = jnp.concatenate([x_n, cmd_n], axis=-1)
        actions_n = self.mlp(x_n)
        return jnp.zeros_like(actions_n)

    def sample_and_log_prob(
        self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array], rng: PRNGKeyArray
    ) -> tuple[Array, Array]:
        mean = self(obs, cmd)
        return mean, mean


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
    actor_hidden_dims: int = xax.field(value=64)
    actor_num_layers: int = xax.field(value=4)
    critic_hidden_dims: int = xax.field(value=256)
    critic_num_layers: int = xax.field(value=5)

    # Termination conditions.
    max_episode_length: float = xax.field(value=10.0)
    max_pitch: float = xax.field(value=0.1)
    max_roll: float = xax.field(value=0.1)
    pretrained: str | None = xax.field(value=None)
    checkpoint_num: int | None = xax.field(value=None)


class HumanoidWalkingTask(PPOTask[HumanoidWalkingConfig]):
    def get_environment(self) -> MjxEnv:
        return MjxEnv(
            self.config,
            terminations=[
                RollTooGreatTermination(max_roll=1.04),
                PitchTooGreatTermination(max_pitch=1.04),
            ],
            resets=[
                XYPositionResetBuilder(),
            ],
            rewards=[
                TrackLinearVelocityXYReward(scale=0.5),
                HeightReward(
                    scale=0.5,
                    height_target=1.4,
                ),
            ],
            observations=[
                BaseOrientationObservation(noise_type="gaussian", noise=0.01),
                BaseLinearVelocityObservation(noise_type="gaussian", noise=0.01),
                BaseAngularVelocityObservation(noise_type="gaussian", noise=0.01),
                JointPositionObservation(noise_type="gaussian", noise=0.01),
                JointVelocityObservation(noise_type="gaussian", noise=0.01),
                # TODO: default humanoid doesn't have sensors, add them later
            ],
            commands=[
                LinearVelocityCommand(
                    x_scale=0.0,
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
                        actor = HumanoidActorModel(
                            mlp=mlp, init_log_std=-0.7, num_outputs=NUM_OUTPUTS
                        )
                    case "zero":
                        actor = HumanoidZeroActions(
                            mlp=mlp, init_log_std=-0.7, num_outputs=NUM_OUTPUTS
                        )
                    case _:
                        raise ValueError(
                            f"Invalid action: {self.config.viz_action}. "
                            f"Should be one of `policy` or `zero`."
                        )

                model = ActorCriticAgent(
                    actor_module=actor,
                    critic_module=HumanoidCriticModel(
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
    # python -m examples.default_humanoid.walking
    HumanoidWalkingTask.launch(
        HumanoidWalkingConfig(
            num_envs=1024,
            num_steps_per_trajectory=300,
            minibatch_size=640,
            num_learning_epochs=1,
            save_every_n_seconds=60 * 4,
            only_save_most_recent=False,
            # ksim-legacy original setup was dt=0.003 and ctrl_dt=0.012 ~ 83.33 hz
            ctrl_dt=0.01,
            dt=0.001,
            solver_type=mujoco.mjtSolver.mjSOL_NEWTON.value,
            solver_iterations=6,
            solver_ls_iterations=6,
            actuator_type="scaled_torque",
        ),
    )
