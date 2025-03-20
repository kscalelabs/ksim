"""Defines a standard task interface for training a policy."""

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.data import Transition, generate_transition_batches
from ksim.task.rl import RLConfig, RLTask


def get_deltas(
    rewards: Array,
    values: Array,
    values_shifted: Array,
    termination_mask: Array,
    decay_gamma: float,
) -> Array:
    """Computes the TD residuals for a rollout."""
    return rewards + decay_gamma * values_shifted * termination_mask - values


def compute_advantages_and_value_targets(
    values_bt: Array,
    rewards_bt: Array,
    dones_bt: Array,
    decay_gamma: float,
    gae_lambda: float,
    normalize_advantages: bool = True,
) -> tuple[Array, Array]:
    """Computes the advantages using Generalized Advantage Estimation (GAE)."""

    def scan_fn(adv_t_plus_1: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        """Scanning this computes the advantages in reverse order."""
        delta, mask = x
        adv_t = delta + decay_gamma * gae_lambda * mask * adv_t_plus_1
        return adv_t, adv_t

    def compute_for_sample(values_t: Array, rewards_t: Array, dones_t: Array) -> tuple[Array, Array]:
        # If the episode terminated at the last step, use 0 for bootstrapping.
        bootstrap_value = jnp.where(dones_t[-1], 0.0, values_t[-1])
        values_shifted = jnp.concatenate([values_t[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
        mask = jnp.where(dones_t, 0.0, 1.0)
        deltas = get_deltas(rewards_t, values_t, values_shifted, mask, decay_gamma)
        _, gae = jax.lax.scan(scan_fn, jnp.zeros_like(deltas[-1]), (deltas, mask), reverse=True)
        value_targets = gae + values_t
        return gae, value_targets

    # Compute the advantages and value targets for each sample in the batch.
    par_compute = jax.vmap(compute_for_sample, in_axes=0)
    advantages_bt, value_targets_bt = par_compute(values_bt, rewards_bt, dones_bt)

    if normalize_advantages:
        advantages_bt = advantages_bt / (advantages_bt.std(axis=-1, keepdims=True) + 1e-6)

    return advantages_bt, value_targets_bt


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
    return jnp.maximum(error**2, clipped_error**2)


def compute_ppo_loss(
    log_probs_btn: Array,
    values_bt: Array,
    on_policy_log_probs_btn: Array,
    on_policy_values_bt: Array,
    advantages_bt: Array,
    value_targets_bt: Array,
    dones_bt: Array,
    *,
    entropy_btn: Array | None = None,
    clip_param: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.008,
    use_clipped_value_loss: bool = True,
) -> Array:
    """Compute PPO loss.

    Args:
        log_probs_btn: The log probabilities of the actions, with shape (B, T, *A).
        values_bt: The state-value estimates, with shape (B, T).
        on_policy_log_probs_btn: The original policy's log probabilities of the
            actions, with shape (B, T, *A).
        on_policy_values_bt: The original policy's values of the actions, with
            shape (B, T).
        advantages_bt: The advantages, with shape (B, T).
        value_targets_bt: The value targets, with shape (B, T).
        dones_bt: The termination mask, with shape (B, T).
        entropy_btn: The entropy of the action distribution, with shape (B, T, *A).
        clip_param: The clip parameter for PPO.
        value_loss_coef: The value loss coefficient for PPO.
        entropy_coef: The entropy coefficient for PPO.
        use_clipped_value_loss: Whether to use clipped value loss.

    Returns:
        The PPO loss, with shape (B, T).
    """
    chex.assert_equal_shape_prefix(
        [
            log_probs_btn,
            values_bt,
            on_policy_log_probs_btn,
            on_policy_values_bt,
            advantages_bt,
            value_targets_bt,
            dones_bt,
        ],
        prefix_len=2,
    )

    # Only compute gradient to the current policy.
    on_policy_log_probs_btn = jax.lax.stop_gradient(on_policy_log_probs_btn)
    on_policy_values_bt = jax.lax.stop_gradient(on_policy_values_bt)
    advantages_bt = jax.lax.stop_gradient(advantages_bt)
    value_targets_bt = jax.lax.stop_gradient(value_targets_bt)

    def compute_for_sample(
        log_probs_n: Array,
        values: Array,
        on_policy_log_probs_n: Array,
        on_policy_values: Array,
        advantages: Array,
        value_targets: Array,
        dones: Array,
        entropy_n: Array | None,
    ) -> Array:
        # Preventing underflow / overflow in calculating the ratio.
        ratio_n = jnp.exp(jnp.clip(log_probs_n - on_policy_log_probs_n, -10, 10))

        # Computes clipped policy objective.
        clipped_ratio_n = jnp.clip(ratio_n, 1 - clip_param, 1 + clip_param)
        policy_objective_n = jnp.minimum(ratio_n * advantages[..., None], clipped_ratio_n * advantages[..., None])
        policy_objective = policy_objective_n.mean(axis=-1)

        # Computes the value loss, with or without clipping.
        if use_clipped_value_loss:
            value_mse = 0.5 * clipped_value_loss(
                target_values=on_policy_values,
                values=values,
                value_targets=value_targets,
                clip_param=clip_param,
            )
        else:
            value_mse = 0.5 * (value_targets - values) ** 2

        value_objective = value_loss_coef * value_mse
        total_objective = policy_objective - value_objective

        # Adds the entropy bonus term, if provided.
        if entropy_n is not None:
            total_objective = total_objective + entropy_coef * entropy_n.mean(axis=-1)

        # Maximize the objective.
        total_loss = -total_objective

        # Zero out the loss for terminated trajectories.
        total_loss = jnp.where(dones, 0.0, total_loss)

        return total_loss

    par_time = jax.vmap(compute_for_sample, in_axes=0)
    par_batch = jax.vmap(par_time, in_axes=0)

    # Computes the vectorized loss.
    total_loss_bt = par_batch(
        log_probs_btn,
        values_bt,
        on_policy_log_probs_btn,
        on_policy_values_bt,
        advantages_bt,
        value_targets_bt,
        dones_bt,
        entropy_btn,
    )

    return total_loss_bt


@jax.tree_util.register_dataclass
@dataclass
class PPOConfig(RLConfig):
    # Batching parameters.
    group_batches_by_length: bool = xax.field(
        value=True,
        help="Whether to group transitions by length, otherwise, transitions are grouped randomly.",
    )
    num_passes: int = xax.field(
        value=1,
        help="The number of update passes over the set of trajectories",
    )
    include_last_batch: bool = xax.field(
        value=True,
        help="Whether to include the last batch if it's not full.",
    )
    min_batch_size: int = xax.field(
        value=2,
        help="The minimum number of transitions to include in a batch.",
    )
    min_trajectory_length: int = xax.field(
        value=3,
        help="The minimum number of transitions in a trajectory.",
    )

    # PPO parameters.
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
    normalize_advantages: bool = xax.field(
        value=True,
        help="Whether to normalize the advantages.",
    )


Config = TypeVar("Config", bound=PPOConfig)


class PPOTask(RLTask[Config], Generic[Config], ABC):
    """Base class for PPO tasks."""

    @abstractmethod
    def get_on_policy_log_probs(self, model: PyTree, transitions: Transition, rng: PRNGKeyArray) -> Array:
        """Gets the initial log probabilities of the given transitions.

        This function returns the log probabilities of the sampled actions,
        according to the original policy that was used to sample the actions.
        One way to implement this is to compute the log probabilities when
        sampling the actions and store them in the `aux_outputs` field.

        Args:
            model: The user-provided model.
            transitions: The batch of transitions to get probabilities for.
            rng: A random seed.

        Returns:
            The log probabilities of the given actions, with shape (B, T, *A).
        """

    @abstractmethod
    def get_on_policy_values(self, model: PyTree, transitions: Transition, rng: PRNGKeyArray) -> Array:
        """Gets the initial values of the given transitions.

        This function returns the values of the sampled actions, according to
        the original policy that was used to sample the actions.

        Args:
            model: The user-provided model.
            transitions: The batch of transitions to get probabilities for.
            rng: A random seed.

        Returns:
            The values of the given actions, with shape (B, T).
        """

    @abstractmethod
    def get_log_probs(self, model: PyTree, transitions: Transition, rng: PRNGKeyArray) -> tuple[Array, Array | None]:
        """Gets the log probabilities of the given transitions.

        This function operates on the entire batch of actions, observations,
        and commands, so users who implement it should take care to vectorize
        over the relevant dimensions.

        We can also pass an additional entropy term, which is used to add an
        entropy bonus term to the loss function to encourage exploration.

        Args:
            model: The user-provided model.
            transitions: The batch of transitions to get probabilities for.
            rng: A random seed.

        Returns:
            The log probabilites of the given actions, with shape (B, T, *A),
            and the entropy of the action distribution, with shape (B, T, *A),
            or None if we do not want to use the entropy bonus term.
        """

    @abstractmethod
    def get_values(self, model: PyTree, transitions: Transition, rng: PRNGKeyArray) -> Array:
        """Gets the state-value estimates for the given transitions.

        This is usually provided by a critic model.

        This function operates on the entire batch of actions, observations,
        and commands, so users who implement it should take care to vectorize
        over the relevant dimensions.

        Args:
            model: The user-provided model.
            transitions: The batch of transitions estimates for.
            rng: A random seed.

        Returns:
            The state-value estimates for the given transitions, with shape (B, T).
        """

    def get_ppo_metrics(
        self,
        transitions: Transition,
        loss_bt: Array,
        on_policy_log_probs_btn: Array,
        log_probs_btn: Array,
        entropy_btn: Array,
        values_bt: Array,
        value_targets_bt: Array,
        advantages_bt: Array,
    ) -> dict[str, Array | tuple[Array, Array]]:
        """Gets the metrics to be logged.

        If the metric is a scalar, it will be logged as a scalar. If the
        metric is a tuple, it is assumed to be a distribution in (mean, std)
        format and will be logged as a distribution.

        Args:
            transitions: The batch of transitions to get metrics for.
            loss_bt: The PPO loss value.
            on_policy_log_probs_btn: The log probabilities of the actions, with shape (B, T, *A).
            log_probs_btn: The log probabilities of the actions, with shape (B, T, *A).
            entropy_btn: The entropy of the action distribution, with shape (B, T, *A).
            values_bt: The state-value estimates, with shape (B, T).
            value_targets_bt: The value targets, with shape (B, T).
            advantages_bt: The advantages, with shape (B, T).

        Returns:
            A dictionary of metrics to be logged.
        """
        return {
            "loss": (loss_bt.mean(), loss_bt.std()),
            "log_probs": (log_probs_btn.mean(), log_probs_btn.std()),
            "entropy": (entropy_btn.mean(), entropy_btn.std()),
            "value": (values_bt.mean(), values_bt.std()),
            "value_targets": (value_targets_bt.mean(), value_targets_bt.std()),
            "advantages": (advantages_bt.mean(), advantages_bt.std()),
        }

    def get_grad_metrics(self, grads: PyTree) -> dict[str, Array | tuple[Array, Array]]:
        """Gets the metrics to be logged for the gradients.

        If the metric is a scalar, it will be logged as a scalar. If the
        metric is a tuple, it is assumed to be a distribution in (mean, std)
        format and will be logged as a distribution.

        Args:
            grads: The gradients of the model.

        Returns:
            A dictionary of metrics to be logged.
        """
        return {
            "grad_norm": optax.global_norm(grads),
        }

    def get_loss_and_metrics(
        self,
        model: PyTree,
        transitions: Transition,
        rng: PRNGKeyArray,
    ) -> tuple[Array, FrozenDict[str, Array]]:
        """Computes the PPO loss and additional metrics.

        Args:
            model: The model to optimize.
            transitions: The batch of transitions to compute the loss and metrics for.
            rng: A random seed.

        Returns:
            A tuple containing the loss value as a scalar, and a dictionary of
            metrics to log.
        """
        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
        on_policy_log_probs_btn = self.get_on_policy_log_probs(model, transitions, rng1)
        on_policy_values_bt = self.get_on_policy_values(model, transitions, rng2)
        log_probs_btn, entropy_btn = self.get_log_probs(model, transitions, rng3)
        values_bt = self.get_values(model, transitions, rng4)

        advantages_bt, value_targets_bt = compute_advantages_and_value_targets(
            values_bt=values_bt,
            rewards_bt=transitions.reward,
            dones_bt=transitions.done,
            decay_gamma=self.config.gamma,
            gae_lambda=self.config.lam,
            normalize_advantages=self.config.normalize_advantages,
        )

        loss_bt = compute_ppo_loss(
            log_probs_btn=log_probs_btn,
            values_bt=values_bt,
            on_policy_log_probs_btn=on_policy_log_probs_btn,
            on_policy_values_bt=on_policy_values_bt,
            advantages_bt=advantages_bt,
            value_targets_bt=value_targets_bt,
            dones_bt=transitions.done,
            entropy_btn=entropy_btn,
            clip_param=self.config.clip_param,
            value_loss_coef=self.config.value_loss_coef,
            entropy_coef=self.config.entropy_coef,
            use_clipped_value_loss=self.config.use_clipped_value_loss,
        )

        metrics = self.get_ppo_metrics(
            transitions=transitions,
            loss_bt=loss_bt,
            on_policy_log_probs_btn=on_policy_log_probs_btn,
            log_probs_btn=log_probs_btn,
            entropy_btn=entropy_btn,
            values_bt=values_bt,
            value_targets_bt=value_targets_bt,
            advantages_bt=advantages_bt,
        )

        # Mean over all non-masked transitions.
        num_valid = jnp.sum(~transitions.done)
        loss = loss_bt.sum() / (num_valid + 1e-6)

        return loss, metrics

    def get_loss_metrics_and_grads(
        self,
        model: PyTree,
        transitions: Transition,
        rng: PRNGKeyArray,
    ) -> tuple[Array, dict[str, Array | tuple[Array, Array]], PyTree]:
        loss_fn = functools.partial(self.get_loss_and_metrics, transitions=transitions, rng=rng)
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        return loss, metrics, grads

    def _single_step(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        transitions: Transition,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, FrozenDict[str, Array | tuple[Array, Array]]]:
        _, ppo_metrics, grads = self.get_loss_metrics_and_grads(
            model=model,
            transitions=transitions,
            rng=rng,
        )

        # Apply the gradient updates to the model.
        updates, new_opt_state = optimizer.update(grads, opt_state, model)  # type: ignore[operator]
        new_model = eqx.apply_updates(model, updates)

        # Monitor global gradient norm.
        grad_metrics = self.get_grad_metrics(grads)

        return new_model, new_opt_state, FrozenDict(ppo_metrics | grad_metrics)

    @eqx.filter_jit
    def _update_model_on_batches(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        transition_batches: list[Transition],
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, FrozenDict[str, Array]]:
        # JAX requires that we partition the model into mutable and static
        # parts in order to use lax.scan, so that `arr` can be a PyTree.`
        arr, static = eqx.partition(model, eqx.is_inexact_array)

        # Loops over the transition batches and applies gradient updates.
        def scan_fn(
            carry: tuple[PyTree, optax.OptState, PRNGKeyArray],
            xt: Transition,
        ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], FrozenDict[str, Array]]:
            arr, opt_state, rng = carry
            model = eqx.combine(arr, static)
            rng, batch_rng = jax.random.split(rng)
            model, opt_state, metrics = self._single_step(model, optimizer, opt_state, xt, batch_rng)
            arr, _ = eqx.partition(model, eqx.is_inexact_array)
            return (arr, opt_state, rng), metrics

        # Applines N steps of gradient updates.
        def batch_scan_fn(
            carry: tuple[PyTree, optax.OptState, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], FrozenDict[str, Array]]:
            all_metrics = []

            # Looping over the transition batches since it is a list of batches
            # with different lengths, rather than a vectorizable PyTree.
            for transition_batch in transition_batches:
                carry, metrics = scan_fn(carry, transition_batch)
                all_metrics.append(metrics)

            metrics_concat = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *all_metrics)
            return carry, metrics_concat

        # Applies gradient updates.
        (arr, opt_state, _), metrics = jax.lax.scan(
            batch_scan_fn,
            (arr, opt_state, rng),
            length=self.config.num_passes,
        )

        # Get mean metric values over all batches.
        metrics = jax.tree.map(lambda x: x.mean(), metrics)

        # Recombines the mutable and static parts of the model.
        model = eqx.combine(arr, static)

        return model, opt_state, metrics

    def update_model(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        transitions: Transition,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, FrozenDict[str, Array]]:
        """Returns the updated parameters, optimizer state, loss value, and metrics."""
        transition_batches = generate_transition_batches(
            transitions,
            batch_size=self.config.batch_size,
            min_batch_size=self.config.min_batch_size,
            min_trajectory_length=self.config.min_trajectory_length,
            group_by_length=self.config.group_batches_by_length,
            include_last_batch=self.config.include_last_batch,
        )

        model, opt_state, batch_metrics = self._update_model_on_batches(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            transition_batches=transition_batches,
            rng=rng,
        )

        return model, opt_state, batch_metrics
