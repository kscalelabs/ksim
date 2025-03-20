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

from ksim.env.data import Rewards, Trajectory
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
    use_two_step_td_target: bool = False,
) -> tuple[Array, Array]:
    """Computes the advantages using Generalized Advantage Estimation (GAE)."""

    def scan_fn(adv_t_plus_1: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        """Scanning this computes the advantages in reverse order."""
        delta, mask = x
        adv_t = delta + decay_gamma * gae_lambda * mask * adv_t_plus_1
        return adv_t, adv_t

    def compute_gae_and_targets_for_sample(values_t: Array, rewards_t: Array, dones_t: Array) -> tuple[Array, Array]:
        # Use the last value as the bootstrap value.
        values_shifted_t = jnp.concatenate([values_t[1:], jnp.expand_dims(values_t[-1], 0)], axis=0)
        mask_t = jnp.where(dones_t, 0.0, 1.0)
        deltas_t = get_deltas(rewards_t, values_t, values_shifted_t, mask_t, decay_gamma)

        # Compute the GAE.
        _, gae_t = jax.lax.scan(scan_fn, jnp.zeros_like(deltas_t[-1]), (deltas_t, mask_t), reverse=True)
        value_targets_t = gae_t + values_t

        if not use_two_step_td_target:
            return gae_t, value_targets_t

        # Apply another TD step to get the value targets.
        value_targets_shifted_t = jnp.concatenate([value_targets_t[1:], value_targets_t[-1:]], axis=0)
        advantages_t = rewards_t + decay_gamma * value_targets_shifted_t * mask_t - values_t

        return advantages_t, value_targets_t

    # Compute the advantages and value targets for each sample in the batch.
    par_compute = jax.vmap(compute_gae_and_targets_for_sample, in_axes=0)
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
    log_clip_value: float = 10.0,
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
        log_clip_value: The log clip value for PPO, for numerical stability.
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

    def compute_loss_for_sample(
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
        log_ratio = jnp.sum(log_probs_n - on_policy_log_probs_n, axis=-1)
        ratio = jnp.exp(jnp.clip(log_ratio, -log_clip_value, log_clip_value))
        clipped_ratio = jnp.clip(ratio, 1 - clip_param, 1 + clip_param)
        surrogate_1 = ratio * advantages
        surrogate_2 = clipped_ratio * advantages
        policy_objective = jnp.minimum(surrogate_1, surrogate_2)

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

    par_time = jax.vmap(compute_loss_for_sample, in_axes=0)
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
    num_passes: int = xax.field(
        value=1,
        help="The number of update passes over the set of trajectories",
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
    log_clip_value: float = xax.field(
        value=10.0,
        help="The log clip value for PPO, for numerical stability.",
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
    use_two_step_td_target: bool = xax.field(
        value=False,
        help="Whether to use two-step TD targets.",
    )


Config = TypeVar("Config", bound=PPOConfig)


class PPOTask(RLTask[Config], Generic[Config], ABC):
    """Base class for PPO tasks."""

    @abstractmethod
    def get_on_policy_log_probs(self, model: PyTree, trajectories: Trajectory, rng: PRNGKeyArray) -> Array:
        """Gets the initial log probabilities of the given trajectories.

        This function returns the log probabilities of the sampled actions,
        according to the original policy that was used to sample the actions.
        One way to implement this is to compute the log probabilities when
        sampling the actions and store them in the `aux_outputs` field.

        Args:
            model: The user-provided model.
            trajectories: The batch of trajectories to get probabilities for.
            rng: A random seed.

        Returns:
            The log probabilities of the given actions, with shape (B, T, *A).
        """

    @abstractmethod
    def get_on_policy_values(self, model: PyTree, trajectories: Trajectory, rng: PRNGKeyArray) -> Array:
        """Gets the initial values of the given trajectories.

        This function returns the values of the sampled actions, according to
        the original policy that was used to sample the actions.

        Args:
            model: The user-provided model.
            trajectories: The batch of trajectories to get probabilities for.
            rng: A random seed.

        Returns:
            The values of the given actions, with shape (B, T).
        """

    @abstractmethod
    def get_log_probs(self, model: PyTree, trajectories: Trajectory, rng: PRNGKeyArray) -> tuple[Array, Array | None]:
        """Gets the log probabilities of the given trajectories.

        This function operates on the entire batch of actions, observations,
        and commands, so users who implement it should take care to vectorize
        over the relevant dimensions.

        We can also pass an additional entropy term, which is used to add an
        entropy bonus term to the loss function to encourage exploration.

        Args:
            model: The user-provided model.
            trajectories: The batch of trajectories to get probabilities for.
            rng: A random seed.

        Returns:
            The log probabilites of the given actions, with shape (B, T, *A),
            and the entropy of the action distribution, with shape (B, T, *A),
            or None if we do not want to use the entropy bonus term.
        """

    @abstractmethod
    def get_values(self, model: PyTree, trajectories: Trajectory, rng: PRNGKeyArray) -> Array:
        """Gets the state-value estimates for the given trajectories.

        This is usually provided by a critic model.

        This function operates on the entire batch of actions, observations,
        and commands, so users who implement it should take care to vectorize
        over the relevant dimensions.

        Args:
            model: The user-provided model.
            trajectories: The batch of trajectories estimates for.
            rng: A random seed.

        Returns:
            The state-value estimates for the given trajectories, with shape (B, T).
        """

    def get_ppo_metrics(
        self,
        trajectories: Trajectory,
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
            trajectories: The batch of trajectories to get metrics for.
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
        len_b = (~trajectories.done).sum(-1, dtype=loss_bt.dtype) + 1e-6
        batch_size = loss_bt.shape[0]
        return {
            "loss": loss_bt.sum(1) / len_b,
            "traj_len_seconds": len_b * self.config.ctrl_dt,
            "log_probs": log_probs_btn.sum(1).reshape(batch_size, -1) / len_b[:, None],
            "entropy": entropy_btn.sum(1).reshape(batch_size, -1) / len_b[:, None],
            "value": values_bt.sum(1) / len_b,
            "value_targets": value_targets_bt.sum(1) / len_b,
            "advantages": advantages_bt.sum(1) / len_b,
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
        trajectories: Trajectory,
        rewards: Array,
        rng: PRNGKeyArray,
    ) -> tuple[Array, FrozenDict[str, Array]]:
        """Computes the PPO loss and additional metrics.

        Args:
            model: The model to optimize.
            trajectories: The batch of trajectories to compute the loss and metrics for.
            rewards: The rewards for the trajectories.
            rng: A random seed.

        Returns:
            A tuple containing the loss value as a scalar, and a dictionary of
            metrics to log.
        """
        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)
        on_policy_log_probs_btn = self.get_on_policy_log_probs(model, trajectories, rng1)
        on_policy_values_bt = self.get_on_policy_values(model, trajectories, rng2)
        log_probs_btn, entropy_btn = self.get_log_probs(model, trajectories, rng3)
        values_bt = self.get_values(model, trajectories, rng4)

        advantages_bt, value_targets_bt = compute_advantages_and_value_targets(
            values_bt=values_bt,
            rewards_bt=rewards,
            dones_bt=trajectories.done,
            decay_gamma=self.config.gamma,
            gae_lambda=self.config.lam,
            normalize_advantages=self.config.normalize_advantages,
            use_two_step_td_target=self.config.use_two_step_td_target,
        )

        loss_bt = compute_ppo_loss(
            log_probs_btn=log_probs_btn,
            values_bt=values_bt,
            on_policy_log_probs_btn=on_policy_log_probs_btn,
            on_policy_values_bt=on_policy_values_bt,
            advantages_bt=advantages_bt,
            value_targets_bt=value_targets_bt,
            dones_bt=trajectories.done,
            entropy_btn=entropy_btn,
            clip_param=self.config.clip_param,
            value_loss_coef=self.config.value_loss_coef,
            entropy_coef=self.config.entropy_coef,
            log_clip_value=self.config.log_clip_value,
            use_clipped_value_loss=self.config.use_clipped_value_loss,
        )

        metrics = self.get_ppo_metrics(
            trajectories=trajectories,
            loss_bt=loss_bt,
            on_policy_log_probs_btn=on_policy_log_probs_btn,
            log_probs_btn=log_probs_btn,
            entropy_btn=entropy_btn,
            values_bt=values_bt,
            value_targets_bt=value_targets_bt,
            advantages_bt=advantages_bt,
        )

        # Mean over all non-masked trajectories.
        num_valid = jnp.sum(~trajectories.done)
        loss = loss_bt.sum() / (num_valid + 1e-6)

        return loss, metrics

    def get_loss_metrics_and_grads(
        self,
        model: PyTree,
        trajectories: Trajectory,
        rewards: Array,
        rng: PRNGKeyArray,
    ) -> tuple[Array, dict[str, Array | tuple[Array, Array]], PyTree]:
        loss_fn = functools.partial(self.get_loss_and_metrics, trajectories=trajectories, rewards=rewards, rng=rng)
        (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        return loss, metrics, grads

    def _single_step(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectories: Trajectory,
        rewards: Array,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, FrozenDict[str, Array | tuple[Array, Array]]]:
        _, ppo_metrics, grads = self.get_loss_metrics_and_grads(
            model=model,
            trajectories=trajectories,
            rewards=rewards,
            rng=rng,
        )

        # Apply the gradient updates to the model.
        updates, new_opt_state = optimizer.update(grads, opt_state, model)  # type: ignore[operator]
        new_model = eqx.apply_updates(model, updates)

        # Monitor global gradient norm.
        grad_metrics = self.get_grad_metrics(grads)

        return new_model, new_opt_state, FrozenDict(ppo_metrics | grad_metrics)

    @eqx.filter_jit
    def update_model(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectories: Trajectory,
        rewards: Rewards,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, FrozenDict[str, Array]]:
        """Runs PPO updates on a given set of trajectory batches.

        Args:
            model: The model to update.
            optimizer: The optimizer to use.
            opt_state: The optimizer state.
            trajectories: The trajectories to update the model on.
            rewards: The rewards for the trajectories.
            rng: A random seed.

        Returns:
            A tuple containing the updated parameters, optimizer state, and metrics.
        """
        # JAX requires that we partition the model into mutable and static
        # parts in order to use lax.scan, so that `arr` can be a PyTree.`
        arr, static = eqx.partition(model, eqx.is_inexact_array)

        # Loops over the trajectory batches and applies gradient updates.
        def scan_fn(
            carry: tuple[PyTree, optax.OptState, PRNGKeyArray],
            xt: Array,
        ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], FrozenDict[str, Array]]:
            arr, opt_state, rng = carry
            model = eqx.combine(arr, static)
            rng, batch_rng = jax.random.split(rng)

            # Gets the current batch of trajectories and rewards.
            trajectory_batch = jax.tree.map(lambda x: x[xt], trajectories)
            reward_batch = rewards.total[xt]

            model, opt_state, metrics = self._single_step(
                model=model,
                optimizer=optimizer,
                opt_state=opt_state,
                trajectories=trajectory_batch,
                rewards=reward_batch,
                rng=batch_rng,
            )
            arr, _ = eqx.partition(model, eqx.is_inexact_array)
            return (arr, opt_state, rng), metrics

        # Applines N steps of gradient updates.
        def batch_scan_fn(
            carry: tuple[PyTree, optax.OptState, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], FrozenDict[str, Array]]:
            arr, opt_state, rng = carry
            rng, indices_rng = jax.random.split(rng)
            indices = jax.random.permutation(indices_rng, trajectories.done.shape[0])
            indices = indices.reshape(self.config.num_batches, self.batch_size)
            carry = (arr, opt_state, rng)
            carry, new_metrics = jax.lax.scan(scan_fn, carry, indices)
            new_metrics = jax.tree.map(lambda x: x.reshape(x.shape[0] * x.shape[1], -1), new_metrics)
            return carry, metrics

        # Applies gradient updates.
        (arr, opt_state, _), metrics = jax.lax.scan(
            batch_scan_fn,
            (arr, opt_state, rng),
            length=self.config.num_passes,
        )

        # Recombines the mutable and static parts of the model.
        model = eqx.combine(arr, static)

        return model, opt_state, metrics
