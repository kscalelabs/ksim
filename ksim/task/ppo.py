"""Defines a standard task interface for training a policy."""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from brax.base import State as BraxState
from jaxtyping import PyTree

from ksim.task.rl import RLConfig, RLTask


@jax.tree_util.register_dataclass
@dataclass
class PPOConfig(RLConfig):
    clip_param: float = xax.field(value=0.2, help="Clipping parameter for PPO.")
    gamma: float = xax.field(value=0.998, help="Discount factor for PPO.")
    lam: float = xax.field(value=0.95, help="Lambda parameter for PPO.")
    value_loss_coef: float = xax.field(value=0.5, help="Value loss coefficient for PPO.")
    entropy_coef: float = xax.field(value=0.01, help="Entropy coefficient for PPO.")
    learning_rate: float = xax.field(value=1e-3, help="Learning rate for PPO.")
    max_grad_norm: float = xax.field(value=1.0, help="Maximum gradient norm for PPO.")
    use_clipped_value_loss: bool = xax.field(value=True, help="Whether to use clipped value loss for PPO.")
    schedule: str = xax.field(value="fixed", help="Schedule for PPO.")
    desired_kl: float = xax.field(value=0.01, help="Desired KL divergence for PPO.")
    normalize_advantage: bool = xax.field(value=True, help="Whether to normalize advantages.")


@jax.tree_util.register_dataclass
@dataclass
class PPOBatch:
    """A batch of PPO training data."""

    observations: jnp.ndarray
    next_observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    truncation: jnp.ndarray
    termination: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray


Config = TypeVar("Config", bound=PPOConfig)


class PPOTask(RLTask[Config], Generic[Config], ABC):
    """Base class for PPO tasks."""

    def compute_gae(
        self,
        batch: PPOBatch,
        bootstrap_value: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute generalized advantage estimation (GAE).

        Args:
            batch: The batch of training data.
            bootstrap_value: Value estimate for the final state.

        Returns:
            A tuple of (values, advantages).
        """
        # Compute value targets using GAE
        truncation_mask = 1 - batch.truncation
        values_t_plus_1 = jnp.concatenate([batch.values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)

        # Compute TD errors
        deltas = batch.rewards + self.config.gamma * (1 - batch.termination) * values_t_plus_1 - batch.values
        deltas *= truncation_mask

        # Initialize accumulator for GAE computation
        acc = jnp.zeros_like(bootstrap_value)

        def compute_vs_minus_v_xs(
            carry: tuple[float, jnp.ndarray], target_t: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        ) -> tuple[tuple[float, jnp.ndarray], jnp.ndarray]:
            lambda_, acc = carry
            truncation_mask, delta, termination = target_t
            acc = delta + self.config.gamma * (1 - termination) * truncation_mask * lambda_ * acc
            return (lambda_, acc), acc

        # Compute advantages using scan
        (_, _), advantages = jax.lax.scan(
            compute_vs_minus_v_xs,
            (self.config.lam, acc),
            (truncation_mask, deltas, batch.termination),
            length=int(truncation_mask.shape[0]),
            reverse=True,
        )

        # Compute value targets
        values = advantages + batch.values

        return (jax.lax.stop_gradient(values), jax.lax.stop_gradient(advantages))

    def compute_loss(
        self,
        model: eqx.Module,
        batch: PPOBatch,
        output: Any,
        state: xax.State,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute PPO losses.

        Args:
            model: The policy model.
            batch: Training data batch.
            output: Model outputs containing policy_logits and values.
            state: Current training state.

        Returns:
            A tuple of (total_loss, metrics).
        """
        # Get bootstrap value for final state
        bootstrap_value = output.values[-1]

        # Compute advantages and value targets
        values, advantages = self.compute_gae(batch, bootstrap_value)

        # Normalize advantages if configured
        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute probability ratio
        log_probs = output.action_log_probs
        ratio = jnp.exp(log_probs - batch.log_probs)

        # Compute surrogate losses
        surrogate1 = ratio * advantages
        surrogate2 = (
            jnp.clip(
                ratio,
                1 - self.config.clip_param,
                1 + self.config.clip_param,
            )
            * advantages
        )
        policy_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))

        # Compute value loss
        if self.config.use_clipped_value_loss:
            value_pred_clipped = batch.values + jnp.clip(
                output.values - batch.values,
                -self.config.clip_param,
                self.config.clip_param,
            )
            value_losses = jnp.square(output.values - values)
            value_losses_clipped = jnp.square(value_pred_clipped - values)
            value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))
        else:
            value_loss = 0.5 * jnp.mean(jnp.square(output.values - values))

        # Compute entropy loss
        entropy = -jnp.mean(
            jnp.sum(jax.nn.softmax(output.policy_logits) * jax.nn.log_softmax(output.policy_logits), axis=-1)
        )
        entropy_loss = -self.config.entropy_coef * entropy

        # Compute total loss
        total_loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss

        metrics = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean((ratio - 1) - jnp.log(ratio)),
        }

        return total_loss, metrics

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(1e-3),
        )

    @eqx.filter_jit
    def model_update(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectory: BraxState,
    ) -> tuple[PyTree, optax.OptState, jnp.ndarray]:
        # PPO hyperparameters
        gamma = self.config.gamma
        clip_eps = self.config.clip_param
        value_coef = self.config.value_loss_coef
        entropy_coef = self.config.entropy_coef

        def flatten(x: jnp.ndarray) -> jnp.ndarray:
            if isinstance(x, jnp.ndarray) and x.ndim >= 2:
                return x.reshape((-1,) + x.shape[2:])
            return x

        flat_obs = jax.tree_util.tree_map(flatten, trajectory.obs)
        flat_actions = flatten(trajectory.actions)
        flat_old_logp = flatten(trajectory.logp)
        flat_rewards = flatten(trajectory.reward)
        flat_done = flatten(trajectory.done)

        def discount_cumsum(rewards: jnp.ndarray, dones: jnp.ndarray, gamma: float) -> jnp.ndarray:
            def scan_fn(carry: jnp.ndarray, elem: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
                r, d = elem
                new_carry = r + gamma * carry * (1.0 - d)
                return new_carry, new_carry

            _, out = jax.lax.scan(scan_fn, 0.0, (rewards[::-1], dones[::-1]))
            return out[::-1]

        flat_returns = discount_cumsum(flat_rewards, flat_done, gamma)

        def gaussian_log_prob(mean: jnp.ndarray, log_std: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            var = jnp.exp(2 * log_std)
            logp = -0.5 * (((action - mean) ** 2) / (var + 1e-8) + 2 * log_std + jnp.log(2 * jnp.pi))
            return jnp.sum(logp, axis=-1)

        def gaussian_entropy(log_std: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)

        def loss_fn(model: PyTree) -> jnp.ndarray:
            policy_mean, policy_log_std, value = model(flat_obs)
            new_logp = gaussian_log_prob(policy_mean, policy_log_std, flat_actions)
            ratio = jnp.exp(new_logp - flat_old_logp)
            advantages = flat_returns - value
            advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
            surrogate1 = ratio * advantages
            surrogate2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            actor_loss = -jnp.mean(jnp.minimum(surrogate1, surrogate2))
            critic_loss = jnp.mean((flat_returns - value) ** 2)
            entropy_bonus = jnp.mean(gaussian_entropy(policy_log_std))
            total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_bonus
            return total_loss

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
