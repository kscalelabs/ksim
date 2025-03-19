"""Defines a standard task interface for training a policy."""

from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.data import Transition
from ksim.task.rl import RLConfig, RLTask


@jax.tree_util.register_dataclass
@dataclass
class PPOMetrics:
    initial_action_log_probs: Array
    initial_values: Array
    advantages: Array
    value_targets: Array
    returns: Array


def compute_returns(rewards: Array, dones: Array, gamma: float) -> Array:
    """Calculate returns from rewards and dones."""

    # Calculating returns separately using gamma.
    def scan_fn(returns_t_plus_1: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        """Scanning this computes the returns in reverse order."""
        reward, mask = x
        returns = reward + gamma * mask * returns_t_plus_1
        return returns, returns

    _, returns = jax.lax.scan(
        scan_fn,
        jnp.zeros_like(rewards[-1]),
        (rewards, dones),
        reverse=True,
    )

    return returns


def get_deltas(
    rewards: Array,
    values: Array,
    values_shifted: Array,
    termination_mask: Array,
    decay_gamma: float,
) -> Array:
    """Computes the TD residuals for a rollout."""
    deltas = rewards + decay_gamma * values_shifted * termination_mask - values
    return deltas


def compute_advantages_and_value_targets(
    values: Array,
    rewards: Array,
    dones: Array,
    decay_gamma: float,
    gae_lambda: float,
) -> tuple[Array, Array]:
    """Computes the advantages using Generalized Advantage Estimation (GAE).

    Note that some of this logic is NOT stock PPO, using Brax's PPO
    implementation as a reference.

    Values, rewards, dones must have shape of (time, *batch_dims, ...).
    """

    def scan_fn(adv_t_plus_1: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        """Scanning this computes the advantages in reverse order."""
        delta, mask = x
        adv_t = delta + decay_gamma * gae_lambda * mask * adv_t_plus_1
        return adv_t, adv_t

    # We use the last value as the bootstrap value (although it is not fully correct)
    values_shifted = jnp.concatenate([values[1:], jnp.expand_dims(values[-1], 0)], axis=0)
    mask = jnp.where(dones, 0.0, 1.0)

    deltas = get_deltas(rewards, values, values_shifted, mask, decay_gamma)

    _, gae = jax.lax.scan(scan_fn, jnp.zeros_like(deltas[-1]), (deltas, mask), reverse=True)
    value_targets = jnp.add(gae, values)

    # Following Brax and applying another TD step to get the value targets.
    # TODO: Experiment with original GAE & value targets
    value_targets_shifted = jnp.concatenate([value_targets[1:], value_targets[-1:]], axis=0)
    advantages = rewards + decay_gamma * value_targets_shifted * mask - values

    return advantages, value_targets


def clipped_value_loss(
    target_values: Array,
    values: Array,
    value_targets: Array,
    clip_param: float,
) -> Array:
    """Compute the clipped value loss."""
    value_clipped = target_values + (values - target_values).clip(-clip_param, clip_param)
    clipped_error = value_clipped - value_targets
    error = values - value_targets
    return jnp.maximum(error**2, clipped_error**2).mean()


def compute_ppo_loss(
    model: PyTree,
    transitions: Transition,
    ppo_metrics: PPOMetrics,
    rng: PRNGKeyArray,
    *,
    normalize_advantage: bool = True,
    normalize_advantage_in_minibatch: bool = False,
    eps: float = 1e-6,
    clip_param: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.008,
    use_clipped_value_loss: bool = True,
) -> tuple[Array, dict[str, Array]]:
    """Compute the PPO loss.

    Note: minibatches will be shape: (time, env, ...). Depending on the
    sampling function, these may be contiguous along the time dim.
    """
    # get the log probs of the current model
    prediction = jax.vmap(model.actor_model.batched_forward_across_time, in_axes=(0, 0))(
        transitions.obs, transitions.command
    )  # TODO: maybe assume it'll be BT, vmap this... this will break otherwise...
    values = jax.vmap(model.critic_model.batched_forward_across_time, in_axes=(0, 0))(
        transitions.obs, transitions.command
    )
    values = values.squeeze(axis=-1)
    log_probs_per_action = model.action_distribution.log_prob(prediction, transitions.action)
    log_probs = jnp.sum(log_probs_per_action, axis=-1)

    log_prob_diff = log_probs - ppo_metrics.initial_action_log_probs
    ratio = jnp.exp(log_prob_diff)

    # get the state-value estimates
    advantages = ppo_metrics.advantages
    value_targets = ppo_metrics.value_targets

    # Andrychowicz (2021) did not find any benefit in minibatch normalization.
    if normalize_advantage and normalize_advantage_in_minibatch:
        advantages = (advantages - advantages.mean()) / (advantages.std() + eps)

    # Policy loss, with clipping
    clipped_ratio = jnp.clip(ratio, 1 - clip_param, 1 + clip_param)
    policy_objective = jnp.mean(
        jnp.minimum(
            ratio * advantages,
            clipped_ratio * advantages,
        )
    )

    # value loss term
    value_mse = jax.lax.cond(
        use_clipped_value_loss,
        lambda: 0.5
        * clipped_value_loss(
            target_values=ppo_metrics.initial_values,
            values=values,
            value_targets=value_targets,
            clip_param=clip_param,
        ),
        lambda: 0.5 * jnp.mean((value_targets - values) ** 2),
    )
    value_objective = value_loss_coef * value_mse

    # Entropy bonus term.
    entropies = model.action_distribution.entropy(prediction, rng)
    entropy_objective = entropy_coef * jnp.mean(entropies)

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
        "advantage_norm_mean": jnp.mean(ppo_metrics.advantages),
        "action_mean": jnp.mean(transitions.action),
        "action_std": jnp.std(transitions.action),
        "action_max": jnp.max(transitions.action),
        "action_min": jnp.min(transitions.action),
        "prediction_mean": jnp.mean(prediction),
        "prediction_std": jnp.std(prediction),
        "log_prob_mean": jnp.mean(log_probs),
        "log_prob_max": jnp.max(log_probs),
        "log_prob_min": jnp.min(log_probs),
        "values_std": jnp.std(values),
        "values_mean": jnp.mean(values),
        "obs_nans_ratio": xax.compute_nan_ratio(transitions.obs),
        "action_nans_ratio": xax.compute_nan_ratio(transitions.action),
    }

    return total_loss, metrics_to_log


@jax.tree_util.register_dataclass
@dataclass
class PPOConfig(RLConfig):
    clip_param: float = xax.field(
        value=0.2,
        help="Clipping parameter for PPO, see Schulman et al. (2017)",
    )
    value_loss_coef: float = xax.field(
        value=0.5,
        help="Value loss coefficient for PPO.",
    )
    use_clipped_value_loss: bool = xax.field(
        value=True,
        help="Whether to use clipped value loss.",
    )
    entropy_coef: float = xax.field(
        value=0.008,
        help="Entropy coefficient for PPO.",
    )
    gamma: float = xax.field(
        value=0.99,
        help="Discount factor for PPO",
    )
    lam: float = xax.field(
        value=0.95,
        help="Lambda for GAE: high = more bias; low = more variance",
    )
    eps: float = xax.field(
        value=1e-6,
        help="Small epsilon value to avoid division by zero.",
    )
    learning_rate: float = xax.field(
        value=1e-4,
        help="Learning rate for PPO.",
    )
    max_grad_norm: float = xax.field(
        value=0.5,
        help="Maximum gradient norm for clipping.",
    )
    scale_rewards: bool = xax.field(
        value=False,
        help="Whether to scale rewards, see Engstrom, Ilyas, et al. (2020).",
    )
    normalize_advantage: bool = xax.field(
        value=True,
        help="Whether to normalize advantages.",
    )
    normalize_advantage_in_minibatch: bool = xax.field(
        value=False,
        help="Whether to normalize advantages at the minibatch level as per OpenAI baselines.",
    )
    reward_scaling_alpha: float = xax.field(
        value=0.0003,
        help="Rate at which to update reward scaling online as per Hessel, Soyer, et al. (2018).",
    )
    obs_norm_alpha: float = xax.field(
        value=0.0003,
        help="Rate at which to update observation norm stats, Andrychowicz (2021) and Duan (2016).",
    )
    pretrained: str | None = xax.field(
        value=None,
        help="The path to a pretrained model to load.",
    )
    checkpoint_num: int | None = xax.field(
        value=None,
        help="The checkpoint number to load. Otherwise the latest checkpoint is loaded.",
    )


Config = TypeVar("Config", bound=PPOConfig)


class PPOTask(RLTask[Config], Generic[Config], ABC):
    """Base class for PPO tasks."""

    def get_optimizer(self) -> optax.GradientTransformation:
        """Builds the optimizer.

        This provides a reasonable default optimizer for training PPO models,
        but can be overridden by subclasses who want to do something different.
        """
        return optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )

    @eqx.filter_jit
    def get_rollout_time_stats(self, transitions: Transition, model: PyTree) -> PPOMetrics:
        """Calculating advantages and returns for a rollout."""
        prediction = jax.vmap(model.actor_model.batched_forward_across_time, in_axes=(0, 0))(
            transitions.obs, transitions.command
        )
        initial_values = jax.vmap(model.critic_model.batched_forward_across_time, in_axes=(0, 0))(
            transitions.obs, transitions.command
        )
        initial_values = initial_values.squeeze(axis=-1)

        initial_action_log_probs_per_action = model.action_distribution.log_prob(
            parameters=prediction,
            actions=transitions.action,
        )
        initial_action_log_probs = jnp.sum(initial_action_log_probs_per_action, axis=-1)

        advantages, value_targets = compute_advantages_and_value_targets(
            values=initial_values,
            rewards=transitions.reward,
            dones=transitions.done,
            decay_gamma=self.config.gamma,
            gae_lambda=self.config.lam,
        )

        # we decouple the computation of returns from the value targets.
        returns = compute_returns(
            rewards=transitions.reward,
            dones=transitions.done,
            gamma=self.config.gamma,
        )

        # normalizing at the trajectory dataset level
        if self.config.normalize_advantage and not self.config.normalize_advantage_in_minibatch:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.config.eps)

        return PPOMetrics(
            initial_action_log_probs=jax.lax.stop_gradient(initial_action_log_probs),
            initial_values=jax.lax.stop_gradient(initial_values),
            advantages=jax.lax.stop_gradient(advantages),
            value_targets=jax.lax.stop_gradient(value_targets),
            returns=jax.lax.stop_gradient(returns),
        )

    def loss_metrics_grads(
        self,
        model: PyTree,
        transitions: Transition,
        rng: PRNGKeyArray,
    ) -> tuple[Array, dict[str, Array], PyTree]:
        """Value_and_grad computation with metrics."""

        def loss_fn(model: PyTree) -> tuple[Array, dict[str, Array]]:
            """Model is a PyTree and can be optimized via optax."""
            rollout_time_stats = minibatch.rollout_time_stats
            assert isinstance(rollout_time_stats, PPORolloutTimeStats)
            return compute_ppo_loss(
                model=model,
                transitions=minibatch.transitions,
                rollout_time_stats=rollout_time_stats,
                rng=rng,
                normalize_advantage=self.config.normalize_advantage,
                normalize_advantage_in_minibatch=self.config.normalize_advantage_in_minibatch,
                eps=self.config.eps,
                clip_param=self.config.clip_param,
                value_loss_coef=self.config.value_loss_coef,
                entropy_coef=self.config.entropy_coef,
                use_clipped_value_loss=self.config.use_clipped_value_loss,
            )

        # TODO: let's own our own debuggable filter value and grad
        (loss_val, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        return loss_val, metrics, grads

    def _single_step(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        transitions: Transition,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, Array, FrozenDict[str, Array]]:
        loss_val, metrics, grads = self.loss_metrics_grads(
            model=model,
            transitions=transitions,
            rng=rng,
        )

        # Apply the gradient updates to the model.
        updates, new_opt_state = optimizer.update(grads, opt_state, model)  # type: ignore[operator]
        new_model = eqx.apply_updates(model, updates)

        # Add global gradient norm and loss to the metrics, to monitor training.
        grad_norm = optax.global_norm(grads)
        assert isinstance(grad_norm, Array)
        metrics["loss"] = loss_val
        metrics["grad_norm"] = grad_norm

        return new_model, new_opt_state, loss_val, FrozenDict(metrics)

    def update_model(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        transitions: Transition,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, Array, FrozenDict[str, Array]]:
        """Returns the updated parameters, optimizer state, loss value, and metrics."""
        breakpoint()

        # return self._single_step(
        #     model=model,
        #     optimizer=optimizer,
        #     opt_state=opt_state,
        #     transitions=transitions,
        #     rng=rng,
        # )

        # TODO: Implement this.
        return model, opt_state, 0.0, FrozenDict({})
