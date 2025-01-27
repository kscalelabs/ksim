"""Defines the actor-critic neural network model."""

from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
from distrax import Normal


class ActorCritic(eqx.Module):
    actor: eqx.nn.Sequential
    critic: eqx.nn.Sequential
    std: jnp.ndarray
    distribution: Normal = None

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: List[int] = [256, 256, 256],
        critic_hidden_dims: List[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        key: jax.random.PRNGKey = None,
    ) -> None:
        super().__init__()
        if key is None:
            key = jax.random.PRNGKey(0)
        actor_key, critic_key = jax.random.split(key)

        activation_fn = get_activation(activation)

        # Policy network
        actor_layers = []
        current_dim = num_actor_obs
        for hidden_dim in actor_hidden_dims:
            actor_layers.extend(
                [
                    eqx.nn.Linear(current_dim, hidden_dim, key=jax.random.fold_in(actor_key, len(actor_layers))),
                    activation_fn,
                ]
            )
            current_dim = hidden_dim
        actor_layers.append(
            eqx.nn.Linear(current_dim, num_actions, key=jax.random.fold_in(actor_key, len(actor_layers)))
        )
        self.actor = eqx.nn.Sequential(actor_layers)

        # Value function network
        critic_layers = []
        current_dim = num_critic_obs
        for hidden_dim in critic_hidden_dims:
            critic_layers.extend(
                [
                    eqx.nn.Linear(current_dim, hidden_dim, key=jax.random.fold_in(critic_key, len(critic_layers))),
                    activation_fn,
                ]
            )
            current_dim = hidden_dim
        critic_layers.append(eqx.nn.Linear(current_dim, 1, key=jax.random.fold_in(critic_key, len(critic_layers))))
        self.critic = eqx.nn.Sequential(critic_layers)

        # Action noise
        self.std = init_noise_std * jnp.ones(num_actions)

    def update_distribution(self, observations_bn: jnp.ndarray) -> None:
        mean_bn = self.actor(observations_bn)
        self.distribution = Normal(mean_bn, mean_bn * 0.0 + self.std)

    def act(self, observations_bn: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        mean_bn = self.actor(observations_bn)
        return Normal(mean_bn, self.std).sample(seed=key)

    def get_actions_log_prob(self, actions_bn: jnp.ndarray, observations_bn: jnp.ndarray) -> jnp.ndarray:
        mean_bn = self.actor(observations_bn)
        return Normal(mean_bn, self.std).log_prob(actions_bn).sum(axis=-1)

    def act_inference(self, observations_bn: jnp.ndarray) -> jnp.ndarray:
        return self.actor(observations_bn)

    def evaluate(self, critic_observations_bn: jnp.ndarray) -> jnp.ndarray:
        return self.critic(critic_observations_bn)


def get_activation(act_name: str) -> Callable:
    if act_name == "elu":
        return jax.nn.elu
    elif act_name == "selu":
        return jax.nn.selu
    elif act_name == "relu":
        return jax.nn.relu
    elif act_name == "crelu":
        return lambda x: jnp.concatenate([jax.nn.relu(x), jax.nn.relu(-x)], axis=-1)
    elif act_name == "lrelu":
        return jax.nn.leaky_relu
    elif act_name == "tanh":
        return jnp.tanh
    elif act_name == "sigmoid":
        return jax.nn.sigmoid
    else:
        raise ValueError(f"Invalid activation function: {act_name}")
