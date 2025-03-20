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


@eqx.filter_jit
def compute_returns(rewards_bt: Array, dones_bt: Array, gamma: float) -> Array:
    """Calculate returns from rewards and dones."""

    def scan_fn(returns_t_plus_1: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        """Scanning this computes the returns in reverse order."""
        reward, mask = x
        returns = reward + gamma * mask * returns_t_plus_1
        return returns, returns

    def compute_return_for_sample(rewards_t: Array, dones_t: Array, gamma: float) -> Array:
        _, returns = jax.lax.scan(scan_fn, jnp.zeros_like(rewards_t[-1]), (rewards_t, dones_t), reverse=True)
        return returns

    # Compute returns for each sample in the batch.
    returns_bt = jax.vmap(compute_return_for_sample, in_axes=(0, 0, None))(rewards_bt, dones_bt, gamma)

    return returns_bt


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
) -> tuple[Array, Array]:
    """Computes the advantages using Generalized Advantage Estimation (GAE)."""

    def scan_fn(adv_t_plus_1: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        """Scanning this computes the advantages in reverse order."""
        delta, mask = x
        adv_t = delta + decay_gamma * gae_lambda * mask * adv_t_plus_1
        return adv_t, adv_t

    def compute_for_sample(
        values_t: Array,
        rewards_t: Array,
        dones_t: Array,
        decay_gamma: float,
        gae_lambda: float,
    ) -> tuple[Array, Array]:
        # We use the last value as the bootstrap value (although it is not fully correct)
        values_shifted = jnp.concatenate([values_t[1:], jnp.expand_dims(values_t[-1], 0)], axis=0)
        mask = jnp.where(dones_t, 0.0, 1.0)

        deltas = get_deltas(rewards_t, values_t, values_shifted, mask, decay_gamma)

        _, gae = jax.lax.scan(scan_fn, jnp.zeros_like(deltas[-1]), (deltas, mask), reverse=True)
        value_targets = jnp.add(gae, values_t)

        # Following Brax and applying another TD step to get the value targets.
        # TODO: Experiment with original GAE & value targets
        value_targets_shifted = jnp.concatenate([value_targets[1:], value_targets[-1:]], axis=0)
        advantages = rewards_t + decay_gamma * value_targets_shifted * mask - values_t

        return advantages, value_targets

    # Compute the advantages and value targets for each sample in the batch.
    par_compute = jax.vmap(compute_for_sample, in_axes=(0, 0, 0, None, None))
    advantages_bt, value_targets_bt = par_compute(values_bt, rewards_bt, dones_bt, decay_gamma, gae_lambda)

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
            advantages_bt,
            value_targets_bt,
            dones_bt,
        ],
        prefix_len=2,
    )

    # Only compute gradient to the current policy.
    on_policy_log_probs_btn = jax.lax.stop_gradient(on_policy_log_probs_btn)
    advantages_bt = jax.lax.stop_gradient(advantages_bt)
    value_targets_bt = jax.lax.stop_gradient(value_targets_bt)

    def compute_for_sample(
        log_probs_n: Array,
        values: Array,
        on_policy_log_probs_n: Array,
        advantages: Array,
        value_targets: Array,
        dones: Array,
        entropy_n: Array | None,
    ) -> Array:
        ratio_n = jnp.exp(log_probs_n - on_policy_log_probs_n)

        # Computes clipped policy objective.
        clipped_ratio_n = jnp.clip(ratio_n, 1 - clip_param, 1 + clip_param)
        policy_objective_n = jnp.minimum(ratio_n * advantages[..., None], clipped_ratio_n * advantages[..., None])
        policy_objective = policy_objective_n.mean(axis=-1)

        # Computes the value loss, with or without clipping.
        value_mse = jax.lax.cond(
            use_clipped_value_loss,
            lambda: 0.5
            * clipped_value_loss(
                target_values=values,
                values=values,
                value_targets=value_targets,
                clip_param=clip_param,
            ),
            lambda: 0.5 * (value_targets - values) ** 2,
        )
        value_objective = value_loss_coef * value_mse
        total_objective = policy_objective - value_objective

        # Adds the entropy bonus term, if provided.
        total_objective = jax.lax.cond(
            entropy_n is not None,
            lambda: total_objective + entropy_coef * entropy_n.mean(axis=-1),
            lambda: total_objective,
        )

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
        log_probs_btn: Array,
        entropy_btn: Array,
        values_bt: Array,
        value_targets_bt: Array,
        advantages_bt: Array,
    ) -> dict[str, Array]:
        """Gets the metrics to be logged.

        Args:
            transitions: The batch of transitions to get metrics for.
            loss_bt: The PPO loss value.
            log_probs_btn: The log probabilities of the actions, with shape (B, T, *A).
            entropy_btn: The entropy of the action distribution, with shape (B, T, *A).
            values_bt: The state-value estimates, with shape (B, T).
            value_targets_bt: The value targets, with shape (B, T).
            advantages_bt: The advantages, with shape (B, T).

        Returns:
            A dictionary of metrics to be logged.
        """
        return {
            "loss_mean": loss_bt.mean(),
            "loss_std": loss_bt.std(),
            "entropy_mean": entropy_btn.mean(),
            "entropy_std": entropy_btn.std(),
            "value_mean": values_bt.mean(),
            "value_std": values_bt.std(),
            "value_targets_mean": value_targets_bt.mean(),
            "value_targets_std": value_targets_bt.std(),
            "advantages_mean": advantages_bt.mean(),
            "advantages_std": advantages_bt.std(),
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
        rng, log_probs_rng, values_rng = jax.random.split(rng, 3)
        on_policy_log_probs_btn = self.get_on_policy_log_probs(model, transitions, log_probs_rng)
        log_probs_btn, entropy_btn = self.get_log_probs(model, transitions, log_probs_rng)
        values_bt = self.get_values(model, transitions, values_rng)

        advantages_bt, value_targets_bt = compute_advantages_and_value_targets(
            values_bt=values_bt,
            rewards_bt=transitions.reward,
            dones_bt=transitions.done,
            decay_gamma=self.config.gamma,
            gae_lambda=self.config.lam,
        )

        # TODO: These do not look correct...
        # returns_bt = compute_returns(transitions.reward, transitions.done, self.config.gamma)

        loss_bt = compute_ppo_loss(
            log_probs_btn=log_probs_btn,
            values_bt=values_bt,
            on_policy_log_probs_btn=on_policy_log_probs_btn,
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
            log_probs_btn=log_probs_btn,
            entropy_btn=entropy_btn,
            values_bt=values_bt,
            value_targets_bt=value_targets_bt,
            advantages_bt=advantages_bt,
        )

        loss = loss_bt.mean()

        return loss, metrics

    def get_loss_metrics_and_grads(
        self,
        model: PyTree,
        transitions: Transition,
        rng: PRNGKeyArray,
    ) -> tuple[Array, dict[str, Array], PyTree]:
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
    ) -> tuple[PyTree, optax.OptState, FrozenDict[str, Array]]:
        _, metrics, grads = self.get_loss_metrics_and_grads(
            model=model,
            transitions=transitions,
            rng=rng,
        )

        # Apply the gradient updates to the model.
        updates, new_opt_state = optimizer.update(grads, opt_state, model)  # type: ignore[operator]
        new_model = eqx.apply_updates(model, updates)

        # Monitor global gradient norm.
        metrics["grad_norm"] = optax.global_norm(grads)

        return new_model, new_opt_state, FrozenDict(metrics)

    def _update_model_on_batches(
        self,
        model: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        transition_batches: list[Transition],
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, FrozenDict[str, Array]]:

        def scan_fn(
            carry: tuple[PyTree, optax.OptState],
            xt: tuple[Transition, PRNGKeyArray],
        ) -> tuple[tuple[PyTree, optax.OptState], FrozenDict[str, Array]]:
            model, opt_state = carry
            batch, batch_rng = xt
            model, opt_state, metrics = self._single_step(model, optimizer, opt_state, batch, batch_rng)
            return (model, opt_state), metrics

        def batch_scan_fn(
            carry: tuple[PyTree, optax.OptState],
            _: None,
            transition_batches: list[Transition],
        ) -> tuple[tuple[PyTree, optax.OptState], FrozenDict[str, Array]]:
            xs = (transition_batches, jax.random.split(rng, len(transition_batches)))
            (model, opt_state), metrics = jax.lax.scan(scan_fn, carry, xs)
            return (model, opt_state), metrics

        # Applies gradient updates.
        (model, opt_state), metrics = jax.lax.scan(
            functools.partial(batch_scan_fn, transition_batches=transition_batches),
            (model, opt_state),
            length=self.config.num_passes,
        )

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

        return self._update_model_on_batches(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            transition_batches=transition_batches,
            rng=rng,
        )
