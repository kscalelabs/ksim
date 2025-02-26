"""Default Humanoid walking task using Gymnasium's Humanoid-v4 environment."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.base_env import EnvState
from ksim.env.toy.default_humanoid_env import DefaultHumanoidEnv
from ksim.model.formulations import ActionModel, ActorCriticModel
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask

logger = logging.getLogger(__name__)


# Humanoid-v5 has a 17-dimensional continuous action space
NUM_OUTPUTS = 17


@dataclass
class HumanoidWalkingConfig(PPOConfig):
    """Configuration for Humanoid walking training."""

    # ML model parameters
    actor_hidden_dims: int = xax.field(value=256, help="Hidden dimensions for the actor.")
    actor_num_layers: int = xax.field(value=2, help="Number of layers for the actor.")
    critic_hidden_dims: int = xax.field(value=256, help="Hidden dimensions for the critic.")
    critic_num_layers: int = xax.field(value=2, help="Number of layers for the critic.")

    # Environment parameters
    render_mode: str | None = xax.field(
        value=None, help="Render mode for the environment. Options: 'human', 'rgb_array', None"
    )

    # Defaults for Gymnasium's Humanoid-v4
    observation_size: int = 376
    action_size: int = 17

    # PPO hyperparameters tuned for Humanoid
    learning_rate: float = xax.field(value=3e-4, help="Learning rate for the optimizer.")
    clip_param: float = xax.field(value=0.2, help="PPO clip parameter.")
    entropy_coef: float = xax.field(value=0.01, help="Entropy coefficient.")
    max_grad_norm: float = xax.field(value=0.5, help="Maximum gradient norm.")


class HumanoidActorModel(ActionModel):
    """Gaussian policy model for Humanoid."""

    mlp: MLP

    def setup(self) -> None:
        # Initialize log standard deviations for the Gaussian policy
        self.log_std = self.param("log_std", nn.initializers.constant(-0.5), (NUM_OUTPUTS,))

    def __call__(self, state: PyTree) -> jax.Array:
        # Extract and flatten observations
        x = state["observations"]
        # Return mean of Gaussian policy
        return self.mlp(x)

    # TODO: Abstract into parent class
    def calc_log_prob(self, prediction: jax.Array, action: jax.Array) -> jax.Array:
        # Calculate log probability of actions under Gaussian distribution
        mean = prediction
        std = jnp.exp(self.log_std)

        log_prob = -0.5 * jnp.square((action - mean) / std) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
        return jnp.sum(log_prob, axis=-1)

    def sample_and_log_prob(self, obs: Array, rng: PRNGKeyArray) -> Tuple[Array, Array]:
        # Sample actions from Gaussian distribution and calculate their log probabilities
        mean = self(obs)
        std = jnp.exp(self.log_std)

        noise = jax.random.normal(rng, mean.shape)
        action = mean + noise * std
        log_prob = self.calc_log_prob(mean, action)

        return action, log_prob


class HumanoidCriticModel(nn.Module):
    """Value function model for Humanoid."""

    mlp: MLP

    def __call__(self, obs: PyTree) -> jax.Array:
        # Extract and flatten observations
        x = obs["observations"]
        # Return value estimate
        return self.mlp(x)


class HumanoidWalkingTask(PPOTask[HumanoidWalkingConfig]):
    """Task for training a default humanoid to walk."""

    def get_environment(self) -> DefaultHumanoidEnv:
        """Get the environment."""
        return DefaultHumanoidEnv(render_mode=self.config.render_mode)

    def get_model_obs_from_state(self, state: EnvState) -> PyTree:
        """Extract model observations from environment state."""
        return state.obs

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

    @property
    def observation_size(self) -> int:
        """Return the observation space size."""
        return self.config.observation_size

    @property
    def action_size(self) -> int:
        """Return the action space size."""
        return self.config.action_size

    # NOTE: Implemented in the HumanoidWalkingTask
    # because it's not the same pattern as the kscale API
    def run_visualization(self) -> None:
        """Run the environment in visualization mode."""
        # Get checkpoint path from CLI args if available
        checkpoint_path = None
        if hasattr(self.config, "checkpoint") and self.config.checkpoint is not None:
            checkpoint_path = Path(self.config.checkpoint)
            if not checkpoint_path.exists():
                logger.error("Checkpoint path %s does not exist", checkpoint_path)
                return
            logger.info("Loading checkpoint from %s", checkpoint_path)

        # Create environment with rendering enabled
        env = DefaultHumanoidEnv(render_mode="human")

        # Get model and initialize parameters
        rng = self.prng_key()
        model = self.get_model(rng)

        # Initialize parameters or load from checkpoint
        if checkpoint_path is not None:
            # TODO: implement proper checkpoint loading
            logger.error("Checkpoint loading not implemented yet")
            return
        else:
            params = self.get_init_params(rng)

        # Run episodes
        episode_count = 0

        try:
            while True:  # Keep running episodes until interrupted
                logger.info("Starting Episode %d", episode_count)
                total_reward = 0
                env_state = env.reset(rng)
                rng, _ = jax.random.split(rng)
                episode_length = 0

                while True:
                    # Get observations and use policy
                    obs = self.get_model_obs_from_state(env_state)
                    rng, action_rng = jax.random.split(rng)
                    action, _ = model.apply(params, obs, action_rng, method="actor_sample_and_log_prob")

                    # Take step
                    env_state = env.step(env_state, action)
                    reward = env_state.reward.item()
                    done = env_state.done.item()
                    total_reward += reward

                    episode_length += 1

                    if done:
                        logger.info(
                            "Episode %d finished after %d steps with total reward: %f",
                            episode_count,
                            episode_length,
                            total_reward,
                        )
                        episode_count += 1
                        time.sleep(1.0)  # Pause briefly between episodes
                        break

        except KeyboardInterrupt:
            logger.info("Stopping episodes - cleaning up...")
        finally:
            env.env.close()


if __name__ == "__main__":
    # python -m examples.default_humanoid.walking action=train
    HumanoidWalkingTask.launch(
        HumanoidWalkingConfig(
            num_envs=1,
            max_trajectory_seconds=10.0,
            valid_every_n_steps=5,
            learning_rate=1e-3,
        ),
    )
