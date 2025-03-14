"""Defines a standard PPO loss policy.

We can move it to losses xax/nn/losses when we add more algorithms.
"""

import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.losses.loss import Loss
from ksim.model.base import ActorCriticAgent
from ksim.model.types import ModelInput
from ksim.normalization import Normalizer


class PPOLoss(Loss):
    """Computes the PPO loss."""

    def __init__(
        self,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.001,
        clip_value_loss: bool = True,
        normalize_advantage: bool = True,
        gamma: float = 0.99,
        lam: float = 0.95,
        eps: float = 1e-6,
    ) -> None:
        """Initialize the PPO loss."""
        super().__init__()
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.normalize_advantage = normalize_advantage
        self.gamma = gamma
        self.lam = lam
        self.clip_value_loss = clip_value_loss
        self.eps = eps

    def compute_loss(
        self,
        agent: ActorCriticAgent,
        command: Array,
        observation: Array,
        initial_action: Array,
        initial_action_log_probs: Array,
        initial_values: Array,
        value_targets: Array,
        advantages: Array,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
        rng: PRNGKeyArray,
    ) -> tuple[Array, dict[str, Array]]:
        """Compute the PPO loss."""
        # get the log probs of the current model
        model_input = ModelInput(
            obs=obs_normalizer(observation),
            command=cmd_normalizer(command),
            action_history=None,
            recurrent_state=None,
        )

        prediction = agent.actor_model.forward(model_input)
        log_probs = agent.action_distribution.log_prob(prediction, initial_action)
        log_prob_diff = log_probs - initial_action_log_probs
        ratio = jnp.exp(log_prob_diff)

        # Andrychowicz (2021) did not find any benefit in minibatch normalization.
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps)

        # Policy loss, with clipping
        clipped_ratio = jnp.clip(ratio, 1 - self.clip_param, 1 + self.clip_param)
        policy_objective = jnp.mean(
            jnp.minimum(
                ratio * advantages,
                clipped_ratio * advantages,
            )
        )

        # Value loss term
        values = agent.critic_model.forward(model_input).squeeze(axis=-1)
        # give axes
        values = values.squeeze(axis=-1)  # (time, env)
        value_mse = jax.lax.cond(
            self.clip_value_loss,
            lambda: 0.5
            * self._clipped_value_loss(
                target_values=initial_values,
                values=values,
                value_targets=value_targets,
            ),
            lambda: 0.5 * jnp.mean((value_targets - values) ** 2),
        )
        value_objective = self.config.value_loss_coef * value_mse

        # Entropy bonus term.
        entropies = agent.action_distribution.entropy(prediction, rng)
        entropy_objective = self.entropy_coef * jnp.mean(entropies)

        total_objective = policy_objective - value_objective + entropy_objective
        total_loss = -total_objective

        metrics_to_log: dict[str, Array] = {
            "policy_objective": policy_objective,
            "value_objective": value_objective,
            "entropy_objective": entropy_objective,
            "total_objective": total_objective,
            "ratio_mean": jnp.mean(ratio),
            "ratio_std": jnp.std(ratio),
            "ratio_max": jnp.max(ratio),
            "ratio_min": jnp.min(ratio),
            "log_prob_diff_mean": jnp.mean(log_prob_diff),
            "advantage_norm_mean": jnp.mean(advantages),
            "action_mean": jnp.mean(initial_action),
            "action_std": jnp.std(initial_action),
            "action_max": jnp.max(initial_action),
            "action_min": jnp.min(initial_action),
            "prediction_mean": jnp.mean(prediction),
            "prediction_std": jnp.std(prediction),
            "log_prob_mean": jnp.mean(log_probs),
            "log_prob_max": jnp.max(log_probs),
            "log_prob_min": jnp.min(log_probs),
            "values_std": jnp.std(values),
            "values_mean": jnp.mean(values),
            "obs_nans_ratio": xax.compute_nan_ratio(initial_action),
            "action_nans_ratio": xax.compute_nan_ratio(initial_action),
        }

        return total_loss, metrics_to_log

    def compute_advantages_and_value_targets(
        self,
        values: Array,
        rewards: Array,
        dones: Array,
    ) -> tuple[Array, Array]:
        """Computes the advantages using Generalized Advantage Estimation (GAE).

        Values, rewards, dones must have shape of (time, *batch_dims, ...).
        """
        assert values.ndim == rewards.ndim == dones.ndim == 2

        def _scan_fn(adv_t_plus_1: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
            """Scanning this computes the advantages in reverse order."""
            delta, mask = x
            adv_t = delta + self.gamma * self.lam * mask * adv_t_plus_1
            return adv_t, adv_t

        # We use the last value as the bootstrap value (although it is not fully correct)
        values_shifted = jnp.concatenate([values[1:], jnp.expand_dims(values[-1], 0)], axis=0)
        mask = 1 - dones

        # getting td residuals
        deltas = rewards + self.gamma * values_shifted * mask - values

        _, advantages = jax.lax.scan(_scan_fn, jnp.zeros_like(deltas[-1]), (deltas, mask), reverse=True)
        value_targets = jnp.add(advantages, values)

        return advantages, value_targets

    def _clipped_value_loss(
        self,
        target_values: Array,
        values: Array,
        value_targets: Array,
    ) -> Array:
        """Compute the clipped value loss."""
        value_clipped = target_values + (values - target_values).clip(-self.clip_param, self.clip_param)
        clipped_error = value_clipped - value_targets
        error = values - value_targets

        return jnp.maximum(error**2, clipped_error**2).mean()
