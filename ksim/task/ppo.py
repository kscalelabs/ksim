"""Defines a standard task interface for training a policy."""

__all__ = [
    "PPOConfig",
    "PPOTask",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.task.rl import RLConfig, RLTask
from ksim.types import Rewards, Trajectory


def get_deltas(
    rewards: Array,
    values: Array,
    values_shifted: Array,
    termination_mask: Array,
    decay_gamma: float,
) -> Array:
    """Computes the TD residuals for a rollout."""
    return rewards + decay_gamma * values_shifted * termination_mask - values


@xax.jit(static_argnames=["decay_gamma", "gae_lambda", "normalize_advantages", "use_two_step_td_target"])
def compute_advantages_and_value_targets(
    values_t: Array,
    rewards_t: Array,
    dones_t: Array,
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
    advantages_t, value_targets_t = compute_gae_and_targets_for_sample(values_t, rewards_t, dones_t)

    if normalize_advantages:
        advantages_t = advantages_t / (advantages_t.std(axis=-1, keepdims=True) + 1e-6)

    return advantages_t, value_targets_t


@xax.jit(static_argnames=["clip_param"])
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


@xax.jit(static_argnames=["clip_param", "value_loss_coef", "entropy_coef", "log_clip_value", "use_clipped_value_loss"])
def compute_ppo_loss(
    log_probs_tn: Array,
    values_t: Array,
    on_policy_log_probs_tn: Array,
    on_policy_values_t: Array,
    advantages_t: Array,
    value_targets_t: Array,
    dones_t: Array,
    *,
    entropy_tn: Array | None = None,
    clip_param: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.008,
    log_clip_value: float = 10.0,
    use_clipped_value_loss: bool = True,
) -> Array:
    """Compute PPO loss.

    Args:
        log_probs_tn: The log probabilities of the actions, with shape (T, *A).
        values_t: The state-value estimates, with shape (T,).
        on_policy_log_probs_tn: The original policy's log probabilities of the
            actions, with shape (T, *A).
        on_policy_values_t: The original policy's values of the actions, with
            shape (T,).
        advantages_t: The advantages, with shape (T,).
        value_targets_t: The value targets, with shape (T,).
        dones_t: The termination mask, with shape (T,).
        entropy_tn: The entropy of the action distribution, with shape (T, *A).
        clip_param: The clip parameter for PPO.
        value_loss_coef: The value loss coefficient for PPO.
        entropy_coef: The entropy coefficient for PPO.
        log_clip_value: The log clip value for PPO, for numerical stability.
        use_clipped_value_loss: Whether to use clipped value loss.

    Returns:
        The PPO loss, with shape (T,).
    """
    chex.assert_equal_shape_prefix(
        [
            log_probs_tn,
            values_t,
            on_policy_log_probs_tn,
            on_policy_values_t,
            advantages_t,
            value_targets_t,
            dones_t,
        ],
        prefix_len=1,
    )

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

    par_fn = jax.vmap(compute_loss_for_sample, in_axes=0)

    # Computes the vectorized loss.
    total_loss_t = par_fn(
        log_probs_tn,
        values_t,
        on_policy_log_probs_tn,
        on_policy_values_t,
        advantages_t,
        value_targets_t,
        dones_t,
        entropy_tn,
    )

    return total_loss_t


@jax.tree_util.register_dataclass
@dataclass
class PPOConfig(RLConfig):
    # Batching parameters.
    num_passes: int = xax.field(
        value=1,
        help="The number of update passes over the set of trajectories",
    )

    # Gradient parameters.
    global_grad_clip: float = xax.field(
        value=10.0,
        help="The maximum gradient norm to clip to.",
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
        rewards: Rewards,
        loss_t: Array,
        on_policy_log_probs_tn: Array,
        log_probs_tn: Array,
        entropy_tn: Array,
        values_t: Array,
        value_targets_t: Array,
        advantages_t: Array,
    ) -> dict[str, Array]:
        """Gets the metrics to be logged.

        If the metric is a scalar, it will be logged as a scalar. If the
        metric is a tuple, it is assumed to be a distribution in (mean, std)
        format and will be logged as a distribution.

        Args:
            trajectories: The batch of trajectories to get metrics for.
            rewards: The rewards for the trajectories.
            loss_t: The PPO loss value.
            on_policy_log_probs_tn: The log probabilities of the actions, with shape (T, *A).
            log_probs_tn: The log probabilities of the actions, with shape (T, *A).
            entropy_tn: The entropy of the action distribution, with shape (T, *A).
            values_t: The state-value estimates, with shape (T,).
            value_targets_t: The value targets, with shape (T,).
            advantages_t: The advantages, with shape (T,).

        Returns:
            A dictionary of metrics to be logged.
        """
        return {
            "loss": loss_t.mean(),
            "log_probs": log_probs_tn.mean(0).flatten(),
            "entropy": entropy_tn.mean(0).flatten(),
            "value": values_t.mean(),
            "value_targets": value_targets_t.mean(),
            "advantages": advantages_t.mean(),
        }

    @xax.jit(static_argnames=["self", "model_static"])
    def get_loss_and_metrics(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        trajectories: Trajectory,
        rewards: Rewards,
        rng: PRNGKeyArray,
    ) -> tuple[Array, xax.FrozenDict[str, Array]]:
        """Computes the PPO loss and additional metrics.

        Args:
            model_arr: The mutable part of the model to optimize.
            model_static: The static part of the model to optimize.
            trajectories: The batch of trajectories to compute the loss and metrics for.
            rewards: The rewards for the trajectories.
            rng: A random seed.

        Returns:
            A tuple containing the loss value as a scalar, and a dictionary of
            metrics to log.
        """
        model = eqx.combine(model_arr, model_static)

        def loss_and_metrics_fn(
            model: PyTree,
            trajectories: Trajectory,
            rewards: Rewards,
            rng: PRNGKeyArray,
        ) -> tuple[Array, xax.FrozenDict[str, Array]]:
            rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

            on_policy_log_probs_tn = jax.lax.stop_gradient(self.get_on_policy_log_probs(model, trajectories, rng1))
            on_policy_values_t = jax.lax.stop_gradient(self.get_on_policy_values(model, trajectories, rng2))
            log_probs_tn, entropy_tn = self.get_log_probs(model, trajectories, rng3)
            values_t = self.get_values(model, trajectories, rng4)

            advantages_t, value_targets_t = compute_advantages_and_value_targets(
                values_t=jax.lax.stop_gradient(values_t),
                rewards_t=rewards.total,
                dones_t=trajectories.done,
                decay_gamma=self.config.gamma,
                gae_lambda=self.config.lam,
                normalize_advantages=self.config.normalize_advantages,
                use_two_step_td_target=self.config.use_two_step_td_target,
            )

            loss_t = compute_ppo_loss(
                log_probs_tn=log_probs_tn,
                values_t=values_t,
                on_policy_log_probs_tn=on_policy_log_probs_tn,
                on_policy_values_t=on_policy_values_t,
                advantages_t=advantages_t,
                value_targets_t=value_targets_t,
                dones_t=trajectories.done,
                entropy_tn=entropy_tn,
                clip_param=self.config.clip_param,
                value_loss_coef=self.config.value_loss_coef,
                entropy_coef=self.config.entropy_coef,
                log_clip_value=self.config.log_clip_value,
                use_clipped_value_loss=self.config.use_clipped_value_loss,
            )

            if entropy_tn is None:
                raise ValueError("Entropy is required for PPO metrics")

            metrics = self.get_ppo_metrics(
                trajectories=trajectories,
                rewards=rewards,
                loss_t=loss_t,
                on_policy_log_probs_tn=on_policy_log_probs_tn,
                log_probs_tn=log_probs_tn,
                entropy_tn=entropy_tn,
                values_t=values_t,
                value_targets_t=value_targets_t,
                advantages_t=advantages_t,
            )

            # Mean over all non-masked trajectories.
            num_valid = jnp.sum(~trajectories.done)
            loss = loss_t.sum() / (num_valid + 1e-6)

            return loss, xax.FrozenDict(metrics)

        # Gets the loss and metrics for each trajectory in the batch.
        rngs = jax.random.split(rng, rewards.total.shape[0])
        par_fn = jax.vmap(loss_and_metrics_fn, in_axes=(None, 0, 0, 0))
        loss, metrics = par_fn(model, trajectories, rewards, rngs)

        return loss.mean(), metrics

    @xax.jit(static_argnames=["self", "model_static"])
    def _get_loss_metrics_and_grads(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        trajectories: Trajectory,
        rewards: Rewards,
        rng: PRNGKeyArray,
    ) -> tuple[dict[str, Array], PyTree]:
        loss_fn = jax.grad(self.get_loss_and_metrics, argnums=0, has_aux=True)
        loss_fn = xax.jit(static_argnums=[1])(loss_fn)
        grads, metrics = loss_fn(model_arr, model_static, trajectories, rewards, rng)
        return metrics, grads

    @xax.jit(static_argnames=["self", "optimizer"])
    def _update_model(
        self,
        model_arr: PyTree,
        grads: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
    ) -> tuple[PyTree, optax.OptState, dict[str, Array]]:
        grad_norm = optax.global_norm(grads)
        grad_metrics = {"grad_norm": grad_norm}

        def apply(grads: PyTree, grad_norm: Array) -> tuple[PyTree, optax.OptState]:
            # Clip the global gradient norm to some desired range.
            grad_factor = self.config.global_grad_clip / jnp.maximum(grad_norm, 1e-6)
            grads = jax.tree.map(lambda x: x * grad_factor, grads)

            # Apply the gradient updates.
            updates, new_opt_state = optimizer.update(grads, opt_state, model_arr)
            new_model_arr = eqx.apply_updates(model_arr, updates)
            return new_model_arr, new_opt_state

        # Don't apply updates if the gradient is NaN or Inf.
        new_model_arr, new_opt_state = jax.lax.cond(
            jnp.isnan(grad_norm) | jnp.isinf(grad_norm),
            lambda *_: (model_arr, opt_state),
            apply,
            grads,
            grad_norm,
        )

        return new_model_arr, new_opt_state, grad_metrics

    @xax.jit(static_argnames=["self", "model_static", "optimizer"])
    def _single_step(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectories: Trajectory,
        rewards: Rewards,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, xax.FrozenDict[str, Array]]:
        ppo_metrics, grads = self._get_loss_metrics_and_grads(
            model_arr=model_arr,
            model_static=model_static,
            trajectories=trajectories,
            rewards=rewards,
            rng=rng,
        )

        new_model_arr, new_opt_state, grad_metrics = self._update_model(
            model_arr=model_arr,
            grads=grads,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        return new_model_arr, new_opt_state, xax.FrozenDict(dict(ppo_metrics) | dict(grad_metrics))

    @xax.jit(static_argnames=["self", "model_static", "optimizer"])
    def update_model(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectories: Trajectory,
        rewards: Rewards,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, xax.FrozenDict[str, Array]]:
        """Runs PPO updates on a given set of trajectory batches.

        Args:
            model_arr: The mutable part of the model to update.
            model_static: The static part of the model to update.
            optimizer: The optimizer to use.
            opt_state: The optimizer state.
            trajectories: The trajectories to update the model on.
            rewards: The rewards for the trajectories.
            rng: A random seed.

        Returns:
            A tuple containing the updated parameters, optimizer state, and metrics.
        """

        # Loops over the trajectory batches and applies gradient updates.
        def scan_fn(
            carry: tuple[PyTree, optax.OptState, PRNGKeyArray],
            xt: Array,
        ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], xax.FrozenDict[str, Array]]:
            model_arr, opt_state, rng = carry
            rng, batch_rng = jax.random.split(rng)

            # Gets the current batch of trajectories and rewards.
            trajectory_batch = jax.tree.map(lambda x: x[xt], trajectories)
            reward_batch = jax.tree.map(lambda x: x[xt], rewards)

            model_arr, opt_state, metrics = self._single_step(
                model_arr=model_arr,
                model_static=model_static,
                optimizer=optimizer,
                opt_state=opt_state,
                trajectories=trajectory_batch,
                rewards=reward_batch,
                rng=batch_rng,
            )

            return (model_arr, opt_state, rng), metrics

        # Applines N steps of gradient updates.
        def batch_scan_fn(
            carry: tuple[PyTree, optax.OptState, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], xax.FrozenDict[str, Array]]:
            arr, opt_state, rng = carry

            # Shuffling causes a strange kernel caching issue on GPUs.
            # rng, indices_rng = jax.random.split(rng)
            # indices = jax.random.permutation(indices_rng, trajectories.done.shape[0])
            indices = jnp.arange(trajectories.done.shape[0])

            indices = indices.reshape(self.num_batches, self.batch_size)
            carry = (arr, opt_state, rng)

            carry, metrics = jax.lax.scan(scan_fn, carry, indices)

            # Manual version, instead of using scan.
            # metrics = []
            # for i in indices:
            #     carry, metric = scan_fn(carry, i)
            #     metrics.append(metric)
            # metrics = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *metrics)

            return carry, metrics

        carry = (model_arr, opt_state, rng)

        # Applies gradient updates.
        carry, metrics = jax.lax.scan(batch_scan_fn, carry, length=self.config.num_passes)

        # Manual version, instead of using scan.
        # metrics = []
        # for _ in range(self.config.num_passes):
        #     carry, metric = batch_scan_fn(carry, None)
        #     metrics.append(metric)
        # metrics = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *metrics)

        model_arr, opt_state, _ = carry
        return model_arr, opt_state, metrics
