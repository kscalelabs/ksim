"""Defines loss functions for policy optimization."""

import jax.numpy as jnp


def compute_kl_divergence(
    old_mu: jnp.ndarray,
    old_sigma: jnp.ndarray,
    new_mu: jnp.ndarray,
    new_sigma: jnp.ndarray,
) -> jnp.ndarray:
    """Compute KL divergence between two Gaussian distributions.

    Args:
        old_mu: Mean of old policy distribution.
        old_sigma: Standard deviation of old policy distribution.
        new_mu: Mean of new policy distribution.
        new_sigma: Standard deviation of new policy distribution.

    Returns:
        KL divergence between distributions.
    """
    return jnp.sum(
        jnp.log(new_sigma / old_sigma + 1.0e-5)
        + (jnp.square(old_sigma) + jnp.square(old_mu - new_mu)) / (2.0 * jnp.square(new_sigma))
        - 0.5,
        axis=-1,
    )


def compute_surrogate_loss(
    advantages: jnp.ndarray,
    ratio: jnp.ndarray,
    clip_param: float,
) -> jnp.ndarray:
    """Compute PPO surrogate loss with clipping.

    Args:
        advantages: Advantage estimates.
        ratio: Ratio of new to old action probabilities.
        clip_param: Clipping parameter epsilon.

    Returns:
        Clipped surrogate loss.
    """
    surrogate = -advantages * ratio
    surrogate_clipped = -advantages * jnp.clip(ratio, 1.0 - clip_param, 1.0 + clip_param)
    return jnp.mean(jnp.maximum(surrogate, surrogate_clipped))


def compute_value_loss(
    values: jnp.ndarray,
    returns: jnp.ndarray,
    old_values: jnp.ndarray | None = None,
    clip_param: float | None = None,
) -> jnp.ndarray:
    """Compute value function loss, optionally with clipping.

    Args:
        values: Value function predictions.
        returns: Return estimates.
        old_values: Previous value predictions (for clipping).
        clip_param: Clipping parameter epsilon.

    Returns:
        Value function loss.
    """
    if old_values is not None and clip_param is not None:
        values_clipped = old_values + jnp.clip(values - old_values, -clip_param, clip_param)
        value_losses = jnp.square(values - returns)
        value_losses_clipped = jnp.square(values_clipped - returns)
        return jnp.mean(jnp.maximum(value_losses, value_losses_clipped))
    else:
        return jnp.mean(jnp.square(returns - values))


def compute_policy_loss(
    surrogate_loss: jnp.ndarray,
    value_loss: jnp.ndarray,
    entropy: jnp.ndarray,
    value_loss_coef: float,
    entropy_coef: float,
) -> jnp.ndarray:
    """Compute total policy loss combining surrogate, value, and entropy terms.

    Args:
        surrogate_loss: PPO surrogate loss.
        value_loss: Value function loss.
        entropy: Policy entropy.
        value_loss_coef: Value loss coefficient.
        entropy_coef: Entropy coefficient.

    Returns:
        Combined policy loss.
    """
    return surrogate_loss + value_loss_coef * value_loss - entropy_coef * jnp.mean(entropy)
