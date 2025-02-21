"""Defines a standard task interface for training a policy."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, NamedTuple, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from brax.base import State as BraxState, System
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.task.rl import RLConfig, RLTask


@jax.tree_util.register_dataclass
@dataclass
class PPOConfig(RLConfig):
    # For the CLIP term (see Schulman et al. 2017)
    clip_param: float = xax.field(value=0.2, help="Clipping parameter for PPO.")
    normalize_advantage: bool = xax.field(value=True, help="Whether to normalize advantages.")

    # For the Value Function (VF) term
    value_loss_coef: float = xax.field(value=1.0, help="Value loss coefficient for PPO.")
    use_clipped_value_loss: bool = xax.field(value=True, help="Whether to use clipped value loss.")

    # For the entropy bonus term
    entropy_coef: float = xax.field(value=0.008, help="Entropy coefficient for PPO.")

    # For the GAE computation
    gamma: float = xax.field(value=0.99, help="Discount factor for PPO")
    lam: float = xax.field(value=0.95, help="Lambda for GAE: high = more bias; low = more variance")

    # General training parameters
    # TODO: none of these except `max_grad_norm` are actually used in the training script
    num_learning_epochs: int = xax.field(value=5, help="Number of learning epochs per PPO update.")
    num_mini_batches: int = xax.field(value=4, help="Number of mini-batches per PPO epoch.")
    learning_rate: float = xax.field(value=1e-3, help="Learning rate for PPO.")
    schedule: str = xax.field(
        value="adaptive", help="Learning rate schedule for PPO ('fixed' or 'adaptive')."
    )
    desired_kl: float = xax.field(
        value=0.01, help="Desired KL divergence for adaptive learning rate."
    )
    max_grad_norm: float = xax.field(value=1.0, help="Maximum gradient norm for clipping.")


class PPOBatch(NamedTuple):
    """A batch of PPO training data."""

    observations: PyTree
    next_observations: PyTree
    actions: Array
    rewards: Array
    done: Array
    action_log_probs: Array


@dataclass
class PPOOutput(NamedTuple):
    """Output from PPO model forward pass."""

    values: Array
    action_log_probs: Array


Config = TypeVar("Config", bound=PPOConfig)


class PPOTask(RLTask[Config], Generic[Config], ABC):
    """Base class for PPO tasks.

    Attributes:
        config: The PPO configuration.
        model: The PPO model.
        optimizer: The PPO optimizer.
        state: The PPO state.
        dataset: The PPO dataset.
        max_trajectory_steps: The maximum number of steps in a trajectory.
    """

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(1e-3),
        )

    @abstractmethod
    def get_init_critic_carry(self) -> jnp.ndarray | None: ...

    @abstractmethod
    def get_critic_output(
        self,
        model: PyTree,
        sys: System,
        state: BraxState,
        rng: PRNGKeyArray,
        carry: jnp.ndarray | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]: ...

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
            def scan_fn(
                carry: jnp.ndarray, elem: tuple[jnp.ndarray, jnp.ndarray]
            ) -> tuple[jnp.ndarray, jnp.ndarray]:
                r, d = elem
                new_carry = r + gamma * carry * (1.0 - d)
                return new_carry, new_carry

            _, out = jax.lax.scan(scan_fn, 0.0, (rewards[::-1], dones[::-1]))
            return out[::-1]

        flat_returns = discount_cumsum(flat_rewards, flat_done, gamma)

        def gaussian_log_prob(
            mean: jnp.ndarray, log_std: jnp.ndarray, action: jnp.ndarray
        ) -> jnp.ndarray:
            var = jnp.exp(2 * log_std)
            logp = -0.5 * (
                ((action - mean) ** 2) / (var + 1e-8) + 2 * log_std + jnp.log(2 * jnp.pi)
            )
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
