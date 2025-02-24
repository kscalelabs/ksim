from dataclasses import dataclass
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import xax
from brax.envs.base import State as BraxState
from jaxtyping import Array, PRNGKeyArray

from ksim.env.toy.cartpole_env import CartPoleEnv
from ksim.model.formulations import ActionModel, ActorCriticModel
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask


class CartPoleActionModel(ActionModel):
    """Action model for CartPole."""

    mlp: MLP

    @nn.compact
    def __call__(self, state: Array) -> Array:
        return self.mlp(state)

    def calc_log_prob(self, prediction: Array, action: Array) -> Array:
        logits = prediction
        log_probs = jax.nn.log_softmax(logits)
        action_log_prob = log_probs[jnp.arange(log_probs.shape[0])[:, None], jnp.arange(log_probs.shape[1]), action]
        # NOTE: assumes two batching dimensions
        return action_log_prob

    def sample_and_log_prob(self, obs: Array, rng: PRNGKeyArray) -> Tuple[Array, Array]:
        logits = self(obs)
        log_probs = jax.nn.log_softmax(logits)
        sampled_actions = jax.random.categorical(rng, log_probs)
        action_log_prob = log_probs[jnp.arange(log_probs.shape[0]), sampled_actions]
        return sampled_actions, action_log_prob


class CartPoleCriticModel(nn.Module):
    """Critic model for CartPole."""

    mlp: MLP

    @nn.compact
    def __call__(self, state: Array) -> Array:
        return self.mlp(state)


@dataclass
class CartPoleConfig(PPOConfig):
    """Configuration for CartPole training."""

    # ML model parameters.
    actor_hidden_dims: int = xax.field(value=128, help="Hidden dimensions for the actor.")
    actor_num_layers: int = xax.field(value=2, help="Number of layers for the actor.")
    critic_hidden_dims: int = xax.field(value=128, help="Hidden dimensions for the critic.")
    critic_num_layers: int = xax.field(value=2, help="Number of layers for the critic.")

    observation_size: int = 4
    action_size: int = 1


class CartPoleTask(PPOTask[CartPoleConfig]):
    """Task for CartPole training."""

    def get_environment(self) -> CartPoleEnv:
        """Get the environment.

        Returns:
            The environment.
        """
        return CartPoleEnv()

    def get_model_obs_from_state(self, state: BraxState) -> Array:
        """Get the observation from the state."""
        return state.obs["observations"]

    def get_model(self, key: PRNGKeyArray) -> ActorCriticModel:
        """Get the model.
        Args:
            key: The random key.

        Returns:
            The model.
        """
        return ActorCriticModel(
            actor_module=CartPoleActionModel(
                mlp=MLP(
                    num_hidden_layers=self.config.actor_num_layers,
                    hidden_features=self.config.actor_hidden_dims,
                    out_features=2,  # two discrete actions for CartPole
                ),
            ),
            critic_module=CartPoleCriticModel(
                mlp=MLP(
                    num_hidden_layers=self.config.critic_num_layers,
                    hidden_features=self.config.critic_hidden_dims,
                    out_features=1,
                ),
            ),
        )


if __name__ == "__main__":
    # python -m examples.cartpole.cartpole_task train
    CartPoleTask.launch(
        CartPoleConfig(
            num_envs=1,
            max_trajectory_seconds=10.0,
            valid_every_n_steps=5,
            learning_rate=1e-3,
        ),
    )
