"""Defines a standard task interface for training a policy."""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import xax

from ksim.task.rl import RLConfig, RLTask


@jax.tree_util.register_dataclass
@dataclass
class PPOConfig(RLConfig):
    clip_param: float = xax.field(value=0.2, help="Clipping parameter for PPO.")
    gamma: float = xax.field(value=0.998, help="Discount factor for PPO.")
    lam: float = xax.field(value=0.95, help="Lambda parameter for PPO.")
    value_loss_coef: float = xax.field(value=1.0, help="Value loss coefficient for PPO.")
    entropy_coef: float = xax.field(value=0.0, help="Entropy coefficient for PPO.")
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
        values_t_plus_1 = jnp.concatenate(
            [batch.values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0
        )

        # Compute TD errors
        deltas = (
            batch.rewards
            + self.config.gamma * (1 - batch.termination) * values_t_plus_1
            - batch.values
        )
        deltas *= truncation_mask

        # Initialize accumulator for GAE computation
        acc = jnp.zeros_like(bootstrap_value)

        def compute_vs_minus_v_xs(carry: tuple[float, jnp.ndarray], target_t: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[tuple[float, jnp.ndarray], jnp.ndarray]:
            lambda_, acc = carry
            truncation_mask, delta, termination = target_t
            acc = (
                delta
                + self.config.gamma
                * (1 - termination)
                * truncation_mask
                * lambda_
                * acc
            )
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

        return (
            jax.lax.stop_gradient(values),
            jax.lax.stop_gradient(advantages)
        )

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
        entropy = -jnp.mean(jnp.sum(
            jax.nn.softmax(output.policy_logits) *
            jax.nn.log_softmax(output.policy_logits),
            axis=-1
        ))
        entropy_loss = -self.config.entropy_coef * entropy

        # Compute total loss
        total_loss = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            + entropy_loss
        )

        metrics = {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "entropy": entropy,
            "approx_kl": jnp.mean((ratio - 1) - jnp.log(ratio)),
        }

        return total_loss, metrics
