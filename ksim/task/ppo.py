"""Defines a standard task interface for training a policy."""

from abc import ABC
from dataclasses import dataclass
from typing import Dict, Generic, NamedTuple, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from brax.envs.base import State as BraxState
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.base_env import BaseEnv
from ksim.model.formulations import ActorCriticModel
from ksim.task.rl import RLConfig, RLTask
from ksim.types import ModelObs, ModelOut


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
    schedule: str = xax.field(value="adaptive", help="Learning rate schedule for PPO ('fixed' or 'adaptive').")
    desired_kl: float = xax.field(value=0.01, help="Desired KL divergence for adaptive learning rate.")
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
        """Get the optimizer."""
        return optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )

    def get_init_actor_carry(self) -> Array:
        """Get the actor carry state."""
        raise NotImplementedError("Not implemented at the base PPO class.")

    def get_init_critic_carry(self) -> Array:
        """Get the critic carry state."""
        raise NotImplementedError("Not implemented at the base PPO class.")

    @staticmethod
    @eqx.filter_jit
    def apply_actor(model: ActorCriticModel, params: PyTree, obs: ModelObs) -> ModelOut:
        """Apply the actor model to inputs.

        Args:
            model: The linen-based neural network model.
            params: The variable dictionary of the model {"params": {...}, "other_vars": {...}}
            obs: The input to the model (max_steps, num_envs, observation_size)

        Returns:
            The output of the actor model.
        """
        res = model.apply(params, method="actor", obs=obs)
        return res

    @staticmethod
    @eqx.filter_jit
    def apply_critic(model: ActorCriticModel, params: PyTree, obs: ModelObs) -> ModelOut:
        """Apply the critic model to inputs.

        Args:
            model: The linen-based neural network model.
            params: The variable dictionary of the model {"params": {...}, "other_vars": {...}}
            obs: The input to the model (max_steps, num_envs, observation_size)

        Returns:
            The output of the critic model.
        """
        res = model.apply(params, method="critic", obs=obs)
        return res

    def get_trajectory_batch(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env: BaseEnv,
        rng: PRNGKeyArray,
    ) -> NamedTuple:
        """Rollout the model for a given number of steps.

        Args:
            model: The model (see `apply_actor`)
            params: The variable dictionary of the model {"params": {...}, "other_vars": {...}}
            env: The environment (see `unroll_trajectories`)
            rng: The random key.

        Returns:
            A PPOBatch containing trajectories with shape (num_steps, ...).
        """

        @jax.jit
        def action_log_prob_fn(state: BraxState, rng: PRNGKeyArray) -> Tuple[Array, Array]:
            obs = self.get_model_obs_from_state(state)
            actions, log_probs = model.apply(params, obs, rng, method="actor_sample_and_log_prob")
            return actions, log_probs

        trajectory = env.unroll_trajectories(
            action_log_prob_fn=action_log_prob_fn,
            rng=rng,
            num_steps=self.max_trajectory_steps,
            num_envs=self.config.num_envs,
        )
        observations = self.get_model_obs_from_state(trajectory)
        next_observations = jax.tree_util.tree_map(lambda x: jnp.roll(x, shift=-1, axis=0), trajectory.obs)
        actions = trajectory.info["actions"]
        rewards = trajectory.reward
        done = trajectory.done
        action_log_probs = trajectory.info["action_log_probs"]

        return PPOBatch(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            done=done,
            action_log_probs=action_log_probs,
        )

    @eqx.filter_jit
    def compute_advantages(
        self,
        values: Array,
        batch: PPOBatch,
    ) -> Array:
        """Computes the advantages using Generalized Advantage Estimation (GAE).

        Args:
            values: The value estimates for observations, (num_envs, num_steps, ...).
            batch: The batch containing rewards and termination signals, (num_envs, num_steps, ...).

        Returns:
            Advantages with the same shape as values.
        """

        def scan_fn(carry: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
            d, m = x
            new_carry = d + self.config.gamma * self.config.lam * m * carry
            return new_carry, new_carry

        next_values = jnp.roll(values, shift=-1, axis=0)
        mask = jnp.where(batch.done, 0.0, 1.0)

        # getting td residuals
        deltas = batch.rewards + self.config.gamma * next_values * mask - values

        _, advantages = jax.lax.scan(scan_fn, jnp.zeros_like(deltas[-1]), (deltas[::-1], mask[::-1]))
        return advantages[::-1]

    @eqx.filter_jit
    def compute_loss(
        self,
        model: ActorCriticModel,
        params: PyTree,
        batch: PPOBatch,
    ) -> tuple[Array, Dict[str, Array]]:
        """Compute the PPO loss.

        Args:
            model: The model.
            params: The parameters.
            batch: The mini-batch containing trajectories.

        Returns:
            A tuple of (loss, metrics).
        """
        # get the log probs of the current model
        predictions = self.apply_actor(model, params, batch.observations)
        assert isinstance(predictions, Array)
        log_probs = model.apply(params, predictions, method="actor_calc_log_prob", action=batch.actions)
        ratio = jnp.exp(log_probs - batch.action_log_probs)

        # get the state-value estimates
        values = self.apply_critic(model, params, batch.observations)
        assert isinstance(values, Array)
        values = values.squeeze(axis=-1)  # values is (time, env)
        advantages = self.compute_advantages(values, batch)  # (time, env)

        returns = advantages + values
        # normalizing advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # policy loss with clipping
        policy_loss = -jnp.mean(
            jnp.minimum(
                ratio * advantages,
                jnp.clip(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param) * advantages,
            )
        )

        # value loss term
        # TODO: add clipping
        value_pred = self.apply_critic(model, params, batch.observations)
        value_pred = value_pred.squeeze(axis=-1)  # (time, env)
        value_loss = 0.5 * jnp.mean((returns - value_pred) ** 2)

        # entropy bonus term
        probs = jax.nn.softmax(predictions)  # TODO: make this live in the model
        entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))
        entropy_loss = -self.config.entropy_coef * entropy

        total_loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss

        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
        }
        return total_loss, metrics

    @eqx.filter_jit
    def _jitted_value_and_grad(
        self,
        model: ActorCriticModel,
        params: PyTree,
        batch: PPOBatch,
    ) -> tuple[Array, PyTree]:
        """Jitted version of value_and_grad computation."""
        loss_fn = lambda p: self.compute_loss(model, p, batch)[0]
        return jax.value_and_grad(loss_fn)(params)

    @eqx.filter_jit
    def model_update(
        self,
        model: ActorCriticModel,
        params: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        batch: PPOBatch,
    ) -> tuple[PyTree, optax.OptState]:
        """Update the model parameters."""
        loss_val, grads = self._jitted_value_and_grad(model, params, batch)
        print(f"Loss: {loss_val}")
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state
