"""High Level Formulations of RL Models."""

import flax.linen as nn
import jax


class ActorCriticModel(nn.Module):
    """Actor-Critic model."""

    actor_module: nn.Module
    critic_module: nn.Module

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass of the model."""
        return self.actor_module(x), self.critic_module(x)

    def actor(self, x: jax.Array) -> jax.Array:
        """Actor forward pass."""
        return self.actor_module(x)

    def critic(self, x: jax.Array) -> jax.Array:
        """Critic forward pass."""
        return self.critic_module(x)
