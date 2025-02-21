from dataclasses import dataclass
from typing import Dict
import gymnasium as gym

import flax.linen as nn
import jax
import optax
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOBatch
import xax
from jaxtyping import PRNGKeyArray, Array, PyTree
import jax.numpy as jnp
from ksim.task.ppo import PPOConfig, PPOTask
from brax.base import System
from brax.envs.base import State as BraxState
import equinox as eqx


# Helper function: compute discounted cumulative sum for GAE.
def discount_cumsum_gae(delta: Array, mask: Array, discount: float, gae_lambda: float) -> Array:
    """Computes the discounted cumulative sums of deltas for Generalized Advantage Estimation."""

    def scan_fn(carry: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        d, m = x
        new_carry = d + discount * gae_lambda * m * carry
        return new_carry, new_carry

    # Reverse time axis, scan, then reverse back.
    _, out = jax.lax.scan(scan_fn, jnp.zeros_like(delta), (delta[::-1], mask[::-1]))
    return out[::-1]


class ActorCriticModel(nn.Module):
    """Actor-Critic model."""

    actor: nn.Module
    critic: nn.Module

    def __call__(self, obs: Array) -> tuple[Array, Array]:
        return self.actor(obs), self.critic(obs)


class CartPoleEnv(gym.Env):
    """CartPole environment wrapper to match the BraxState interface."""

    def __init__(self, env: gym.Env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def reset(self, rng: PRNGKeyArray) -> BraxState:
        obs, info = self.env.reset()
        return BraxState(
            pipeline_state=None,  # CartPole doesn't use pipeline state
            obs={"observations": jnp.array(obs)},
            reward=jnp.array(0.0),
            done=jnp.array(False),
            info={"rng": rng, **info},
        )

    def step(self, state: BraxState, action: Array) -> BraxState:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        return BraxState(
            pipeline_state=None,
            obs={"observations": jnp.array(obs)},
            reward=jnp.array(reward),
            done=jnp.array(done),
            info={"rng": state.info["rng"], **info},
        )


@dataclass
class CartPoleConfig(PPOConfig):
    """Configuration for CartPole training."""

    sutton_barto_reward: bool = xax.field(value=False, help="Use Sutton and Barto reward function.")
    actor_hidden_dims: int = xax.field(value=128, help="Hidden dimensions for the actor.")
    actor_num_layers: int = xax.field(value=2, help="Number of layers for the actor.")
    critic_hidden_dims: int = xax.field(value=128, help="Hidden dimensions for the critic.")
    critic_num_layers: int = xax.field(value=2, help="Number of layers for the critic.")
    # Additional hyperparameters for PPO:
    discount: float = xax.field(value=0.99, help="Discount factor.")
    gae_lambda: float = xax.field(value=0.95, help="GAE lambda.")
    clip_param: float = xax.field(value=0.2, help="Clipping parameter for PPO.")
    value_coef: float = xax.field(value=0.5, help="Coefficient for value loss.")
    # entropy_coef: float = xax.field(value=0.01, help="Coefficient for entropy bonus (not used here).")


class CartPoleTask(PPOTask[CartPoleConfig]):
    """Task for CartPole training."""

    def get_environment(self) -> CartPoleEnv:
        """Get the environment.
        Returns:
            The environment.
        """
        return CartPoleEnv(gym.make("CartPole-v1"))

    def get_model(self, key: PRNGKeyArray) -> ActorCriticModel:
        """Get the model.
        Args:
            key: The random key.
        Returns:
            The model.
        """
        return ActorCriticModel(
            actor=MLP(
                num_hidden_layers=self.config.actor_num_layers,
                hidden_features=self.config.actor_hidden_dims,
                out_features=2,  # two discrete actions for CartPole
            ),
            critic=MLP(
                num_hidden_layers=self.config.critic_num_layers,
                hidden_features=self.config.critic_hidden_dims,
                out_features=1,
            ),
        )

    def get_init_params(self, key: PRNGKeyArray) -> PyTree:
        """Get the initial parameters.
        Args:
            key: The random key.
        Returns:
            The initial parameters.
        """
        # Provide a dummy observation (of appropriate shape) for initialization.
        dummy_obs = jnp.array([0.0, 0.0])
        return self.get_model(key).init(key, obs=dummy_obs)

    def get_init_actor_carry(self) -> Array:
        raise NotImplementedError("Not a recurrent model.")

    @eqx.filter_jit
    def get_actor_output(
        self,
        model: ActorCriticModel,
        params: PyTree,
        sys: System,
        state: BraxState,
        rng: PRNGKeyArray,
        carry: Array | None,
    ) -> tuple[Array, None]:
        """Get the actor output.
        Args:
            model: The model definition.
            params: The parameters.
            sys: The system.
            state: The state.
            rng: The random key.
            carry: The carry state.
        Returns:
            The actor output and no carry state (keeping for consistency)
        """
        model_out = model.actor.apply(params, obs=state.obs["observations"])
        assert isinstance(model_out, Array)
        return model_out, None

    def get_init_critic_carry(self) -> Array:
        """Get the critic carry state."""
        raise NotImplementedError("Not a recurrent model.")

    @eqx.filter_jit
    def get_critic_output(
        self,
        model: ActorCriticModel,
        params: PyTree,
        sys: System,
        state: BraxState,
        rng: PRNGKeyArray,
        carry: Array | None,
    ) -> tuple[Array, None]:
        """Get the critic output.
        Args:
            model: The model.
            params: The parameters.
            sys: The system.
            state: The state.
            rng: The random key.
            carry: The carry state.
        Returns:
            The critic output and no carry state (keeping for consistency)
        """
        model_out = model.critic.apply(params, obs=state.obs["observations"])
        assert isinstance(model_out, Array)
        return model_out, None

    def _single_unroll(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env: CartPoleEnv,
        rng: PRNGKeyArray,
    ) -> PPOBatch:
        """Rollout the model for a given number of steps.
        Args:
            model: The model.
            params: The parameters (really a variable dictionary).
            env: The environment.
            rng: The random key.
        Returns:
            A PPOBatch containing trajectories.
        """
        observations = []
        actions = []
        rewards = []
        truncation = []
        termination = []
        action_log_probs = []

        state = env.reset(rng)
        for _ in range(self.max_trajectory_steps):
            logits = model.actor.apply(params, obs=state.obs["observations"])
            assert isinstance(logits, Array)
            log_probs = jax.nn.log_softmax(logits)

            sampled_action = jax.random.categorical(rng, logits)
            log_prob = log_probs[sampled_action]

            state = env.step(state, sampled_action)
            observations.append(state.obs["observations"])
            actions.append(sampled_action)
            rewards.append(state.reward)
            truncation.append(state.done)
            termination.append(state.done)
            action_log_probs.append(log_prob)

        observations = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *observations)
        next_observations = jax.tree_util.tree_map(
            lambda x: jnp.concatenate([x[1:], x[-1:][None, ...]], axis=0), observations
        )
        actions = jnp.stack(actions)
        rewards = jnp.stack(rewards)
        truncation = jnp.stack(truncation)
        termination = jnp.stack(termination)
        action_log_probs = jnp.stack(action_log_probs)

        return PPOBatch(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            truncation=truncation,
            termination=termination,
            action_log_probs=action_log_probs,
        )

    def _multiple_unroll(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env: CartPoleEnv,
        rng: PRNGKeyArray,
        num_envs: int,
    ) -> PPOBatch:
        """Rollout the model for a given number of steps.
        Args:
            model: The model.
            params: The parameters (really a variable dictionary).
            env: The environment.
            rng: The random key.
            num_envs: The number of environments.
        Returns:
            A PPOBatch containing trajectories.
        """
        rngs = jax.random.split(rng, num_envs)
        return jax.vmap(self._single_unroll, in_axes=(None, None, None, 0))(
            model, params, env, rngs
        )

    def compute_values(
        self,
        model: ActorCriticModel,
        params: PyTree,
        batch: PPOBatch,
    ) -> Array:
        """Computes the state-values using the critic model."""
        values = model.critic.apply(params, obs=batch.observations)
        assert isinstance(values, Array)
        return jnp.squeeze(values, axis=-1)  # TODO: see if this is needed

    def compute_advantages(
        self,
        values: Array,
        batch: PPOBatch,
    ) -> Array:
        """Computes the advantages using Generalized Advantage Estimation (GAE).
        Args:
            values: The value estimates for observations, (num_envs, num_steps, ...).
            batch: The batch containing rewards and termination signals, (num_envs, num_steps, ...).
        Returns:
            Advantages with the same shape as values.
        """
        next_values = jnp.concatenate([values[:, 1:], values[:, -1:]], axis=1)
        mask = jnp.where(batch.termination, 0.0, 1.0)

        # getting td residuals
        deltas = batch.rewards + self.config.discount * next_values * mask - values
        advantages = jax.vmap(
            lambda delta, m: discount_cumsum_gae(
                delta, m, self.config.discount, self.config.gae_lambda
            )
        )(deltas, mask)
        return advantages

    def compute_loss(
        self,
        model: ActorCriticModel,
        params: PyTree,
        mini_batch: PPOBatch,
    ) -> tuple[Array, Dict[str, Array]]:
        """Compute the PPO loss.
        Args:
            model: The model.
            params: The parameters.
            mini_batch: The mini-batch containing trajectories.
        Returns:
            A tuple of (loss, metrics).
        """
        # Compute current state-value estimates.
        values = self.compute_values(model, params, mini_batch)  # (num_envs, num_steps, ...)
        advantages = self.compute_advantages(values, mini_batch)  # (num_envs, num_steps, ...)
        # Compute returns (targets for the critic).
        returns = advantages + values

        # flattening
        flat_batch = jax.tree_util.tree_map(lambda x: x.reshape(-1, x.shape[-1]), mini_batch)
        flat_values = values.reshape(-1, values.shape[-1])
        flat_advantages = advantages.reshape(-1, advantages.shape[-1])
        flat_returns = returns.reshape(-1, returns.shape[-1])

        # Compute new logits and log probabilities from the actor.
        new_logits = model.actor.apply(params, obs=flat_batch.observations)
        new_log_probs = jax.nn.log_softmax(new_logits)
        new_log_prob = new_log_probs[jnp.arange(new_logits.shape[0]), flat_batch.actions]
        ratio = jnp.exp(new_log_prob - flat_batch.action_log_probs)

        # Clipped surrogate loss.
        unclipped_obj = ratio * flat_advantages
        clipped_ratio = jnp.clip(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param)
        clipped_obj = clipped_ratio * flat_advantages
        policy_loss = -jnp.mean(jnp.minimum(unclipped_obj, clipped_obj))

        # Critic (value) loss.
        value_loss = jnp.mean((flat_values - flat_returns) ** 2)

        # Total loss: sum of policy loss and value loss.
        total_loss = policy_loss + self.config.value_coef * value_loss

        # Optionally, add an entropy bonus here to encourage exploration.
        # For example:
        # entropy = -jnp.mean(jnp.sum(jax.nn.softmax(new_logits) * new_log_probs, axis=-1))
        # total_loss -= self.config.entropy_coef * entropy

        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "total_loss": total_loss,
        }
        return total_loss, metrics

    def model_update(
        self,
        model: ActorCriticModel,
        params: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        batch: PPOBatch,
    ) -> tuple[PyTree, optax.OptState]:
        """Performs a single optimization update.
        Args:
            model: The model.
            params: The current parameters.
            optimizer: The optimizer.
            opt_state: The current optimizer state.
            batch: The batch used for this update.
        Returns:
            Updated parameters and optimizer state.
        """
        # Compute gradients with respect to the total loss.
        loss_fn = lambda p: self.compute_loss(model, p, batch)[0]
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state
