"""Defines simple task for training a walking policy for the default humanoid."""

from dataclasses import dataclass
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.base_env import EnvState
from ksim.env.builders.commands import AngularVelocityCommand, LinearVelocityCommand
from ksim.env.builders.observation import (
    BaseAngularVelocityObservation,
    BaseLinearVelocityObservation,
    BaseOrientationObservation,
    BasePositionObservation,
    JointPositionObservation,
    JointVelocityObservation,
    SensorObservationBuilder,
)
from ksim.env.builders.resets import XYPositionResetBuilder
from ksim.env.builders.rewards import (
    ActionSmoothnessPenalty,
    AngularVelocityXYPenalty,
    FootContactPenaltyBuilder,
    LinearVelocityZPenalty,
    TrackAngularVelocityZReward,
    TrackLinearVelocityXYReward,
)
from ksim.env.builders.terminations import IllegalContactTerminationBuilder, MinimumHeightTermination
from ksim.env.mjx.mjx_env import MjxEnv
from ksim.model.formulations import ActionModel, ActorCriticModel
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask

# Define constants for the humanoid
NUM_INPUTS = 47  # Based on the observation space dimensions
NUM_OUTPUTS = 17  # Based on the humanoid's controllable joints


class HumanoidActorModel(ActionModel):
    """Actor model for the humanoid walking task."""

    mlp: MLP

    def setup(self) -> None:
        self.log_std = self.param("log_std", nn.initializers.constant(-0.7), (NUM_OUTPUTS,))

    def input_vec_from_state(self, state: PyTree) -> jax.Array:
        lin_vel_cmd_2 = state["info"]["commands"]["linear_velocity_command"]
        ang_vel_cmd_1 = state["info"]["commands"]["angular_velocity_command"]
        joint_pos_j = state["obs"]["joint_position_observation"]
        joint_vel_j = state["obs"]["joint_velocity_observation"]

        base_lin_vel_3 = state["obs"]["base_linear_velocity_observation"]
        base_ang_vel_3 = state["obs"]["base_angular_velocity_observation"]

        x_n = jnp.concatenate(
            [
                lin_vel_cmd_2,
                ang_vel_cmd_1,
                base_lin_vel_3,
                base_ang_vel_3,
                joint_pos_j,
                joint_vel_j,
            ],
            axis=-1,
        )
        return x_n

    def out_from_input_vec(self, x_n: jax.Array) -> jax.Array:
        actions_n = self.mlp(x_n)
        return actions_n

    def __call__(self, state: PyTree) -> jax.Array:
        x_n = self.input_vec_from_state(state)
        actions_n = self.out_from_input_vec(x_n)
        return actions_n

    def calc_log_prob(self, prediction: jax.Array, action: jax.Array) -> jax.Array:
        mean = prediction
        std = jnp.exp(self.log_std)

        log_prob = -0.5 * jnp.square((action - mean) / std) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
        return jnp.sum(log_prob, axis=-1)

    def sample_and_log_prob(self, obs: Array, rng: PRNGKeyArray) -> Tuple[Array, Array]:
        mean = self(obs)
        std = jnp.exp(self.log_std)

        noise = jax.random.normal(rng, mean.shape)
        action = mean + noise * std
        log_prob = self.calc_log_prob(mean, action)

        return action, log_prob


class HumanoidCriticModel(nn.Module):
    """Critic model for the humanoid walking task."""

    mlp: MLP

    @nn.compact
    def __call__(self, state: PyTree) -> jax.Array:
        lin_vel_cmd_2 = state["info"]["commands"]["linear_velocity_command"]
        ang_vel_cmd_1 = state["info"]["commands"]["angular_velocity_command"]
        joint_pos_j = state["obs"]["joint_position_observation"]
        joint_vel_j = state["obs"]["joint_velocity_observation"]

        base_pos_3 = state["obs"]["base_position_observation"]
        base_ang_vel_3 = state["obs"]["base_angular_velocity_observation"]
        base_lin_vel_3 = state["obs"]["base_linear_velocity_observation"]
        base_quat_4 = state["obs"]["base_orientation_observation"]

        x_n = jnp.concatenate(
            [
                lin_vel_cmd_2,
                ang_vel_cmd_1,
                base_pos_3,
                base_ang_vel_3,
                base_lin_vel_3,
                base_quat_4,
                joint_pos_j,
                joint_vel_j,
            ],
            axis=-1,
        )
        value_estimate = self.mlp(x_n)

        return value_estimate


@dataclass
class HumanoidWalkingConfig(PPOConfig):
    """Configuration for the default humanoid walking task."""

    # Robot model name to use
    model_name: str = xax.field(value="humanoid")

    # ML model parameters
    actor_hidden_dims: int = xax.field(value=512)
    actor_num_layers: int = xax.field(value=2)
    critic_hidden_dims: int = xax.field(value=512)
    critic_num_layers: int = xax.field(value=4)
    init_noise_std: float = xax.field(value=1.0)

    # Termination conditions
    max_episode_length: float = xax.field(value=10.0)
    min_height: float = xax.field(value=0.7)  # Minimum height before termination


class HumanoidWalkingTask(PPOTask[HumanoidWalkingConfig]):
    """Task for training a default humanoid to walk."""

    def get_environment(self) -> MjxEnv:
        """Configure the environment with appropriate rewards, terminations, etc."""
        return MjxEnv(
            self.config,
            terminations=[
                IllegalContactTerminationBuilder(
                    body_names=["torso"],  # Terminate if torso touches ground
                ),
                MinimumHeightTermination(min_height=self.config.min_height),
            ],
            resets=[
                XYPositionResetBuilder(),  # Reset position when too far from origin
            ],
            rewards=[
                LinearVelocityZPenalty(scale=-0.1),  # Penalize vertical movement
                AngularVelocityXYPenalty(scale=-0.1),  # Penalize non-yaw rotation
                TrackLinearVelocityXYReward(scale=1.0),  # Reward following velocity commands
                TrackAngularVelocityZReward(scale=0.1),  # Reward following turning commands
                ActionSmoothnessPenalty(scale=-0.01),  # Penalize jerky movements
                # Foot contact penalties to encourage natural gait
                FootContactPenaltyBuilder(
                    scale=-0.05,
                    foot_name="left_foot",
                    allowed_contact_prct=0.6,
                    skip_if_zero_command=["linear_velocity_command", "angular_velocity_command"],
                ),
                FootContactPenaltyBuilder(
                    scale=-0.05,
                    foot_name="right_foot",
                    allowed_contact_prct=0.6,
                    skip_if_zero_command=["linear_velocity_command", "angular_velocity_command"],
                ),
            ],
            observations=[
                BasePositionObservation(noise_type="gaussian", noise=0.01),
                BaseOrientationObservation(noise_type="gaussian", noise=0.01),
                BaseLinearVelocityObservation(noise_type="gaussian", noise=0.01),
                BaseAngularVelocityObservation(noise_type="gaussian", noise=0.01),
                JointPositionObservation(noise_type="gaussian", noise=0.01),
                JointVelocityObservation(noise_type="gaussian", noise=0.01),
            ],
            commands=[
                LinearVelocityCommand(
                    x_scale=1.0,
                    y_scale=0.0,  # Only forward movement initially
                    switch_prob=0.02,
                    zero_prob=0.3,
                ),
                AngularVelocityCommand(
                    scale=0.5,
                    switch_prob=0.02,
                    zero_prob=0.8,
                ),
            ],
        )

    def get_model_obs_from_state(self, state: EnvState) -> PyTree:
        """Extract model observations from environment state."""
        return {
            "obs": state.obs,
            "info": state.info,
        }

    def get_model(self, key: PRNGKeyArray) -> ActorCriticModel:
        """Create the actor-critic model for the humanoid."""
        return ActorCriticModel(
            actor_module=HumanoidActorModel(
                mlp=MLP(
                    num_hidden_layers=self.config.actor_num_layers,
                    hidden_features=self.config.actor_hidden_dims,
                    out_features=NUM_OUTPUTS,
                ),
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
        """Get initial actor RNN state if using RNNs."""
        return None

    def get_init_critic_carry(self) -> None:
        """Get initial critic RNN state if using RNNs."""
        return None

    @property
    def observation_size(self) -> int:
        """Return the observation space size."""
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        """Return the action space size."""
        return self.env.action_size


if __name__ == "__main__":
    # python -m examples.default_humanoid.walking train
    # python -m examples.default_humanoid.walking viz
    HumanoidWalkingTask.launch(
        HumanoidWalkingConfig(
            num_envs=32,
            max_trajectory_seconds=10.0,
            robot_model_name="humanoid",
        ),
    )
