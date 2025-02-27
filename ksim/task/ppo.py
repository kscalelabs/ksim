"""Defines a standard task interface for training a policy."""

from abc import ABC
from dataclasses import dataclass
from typing import Dict, Generic, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.base_env import BaseEnv, EnvState
from ksim.model.formulations import ActorCriticModel
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
    schedule: str = xax.field(value="adaptive", help="Learning rate schedule 'fixed' | 'adaptive'")
    desired_kl: float = xax.field(value=0.01, help="Desired KL divergence for adaptive LR.")
    max_grad_norm: float = xax.field(value=1.0, help="Maximum gradient norm for clipping.")


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

    ########################
    # Implementing RL Task #
    ########################

    # TODO from ML eventually we should create
    def get_init_actor_carry(self) -> Array:
        """Get the actor carry state."""
        raise NotImplementedError("Not implemented at the base PPO class.")

    def get_init_critic_carry(self) -> Array:
        """Get the critic carry state."""
        raise NotImplementedError("Not implemented at the base PPO class.")

    def get_trajectory_batch(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env: BaseEnv,
        rng: PRNGKeyArray,
    ) -> EnvState:
        """Rollout the model for a given number of steps, dims (num_steps, num_envs, ...)"""

        @jax.jit
        def action_log_prob_fn(
            obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array], rng: PRNGKeyArray
        ) -> Tuple[Array, Array]:
            actions, log_probs = model.apply(
                params, obs, cmd, rng, method="actor_sample_and_log_prob"
            )
            assert isinstance(actions, Array)
            assert isinstance(log_probs, Array)
            return actions, log_probs

        env_state_batch = env.unroll_trajectories(
            action_log_prob_fn=action_log_prob_fn,
            rng=rng,
            num_steps=self.max_trajectory_steps,
            num_envs=self.config.num_envs,
        )

        return env_state_batch

    @eqx.filter_jit
    def model_update(
        self,
        model: ActorCriticModel,
        params: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        env_state_batch: EnvState,
    ) -> tuple[PyTree, optax.OptState]:
        """Update the model parameters."""
        loss_val, grads = self._jitted_value_and_grad(model, params, env_state_batch)
        print(f"Loss: {loss_val}")
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    def get_optimizer(self) -> optax.GradientTransformation:
        """Get the optimizer: handled by XAX."""
        return optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )

    # Pass-through abstract methods:
    # `get_environment`, `viz_environment`

    ######################
    # Training Utilities #
    ######################

    @eqx.filter_jit
    def apply_critic(
        self,
        model: ActorCriticModel,
        params: PyTree,
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
    ) -> Array:
        """Apply the critic model to inputs. Used by all actor-critic tasks.

        TODO: it might be worth creating another Task abstraction that requires `apply_critic`
        """
        res = model.apply(params, obs=obs, cmd=cmd, method="critic")
        assert isinstance(res, Array)
        return res

    @eqx.filter_jit
    def compute_loss(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env_state_batch: EnvState,
    ) -> tuple[Array, Dict[str, Array]]:
        """Compute the PPO loss (required by XAX).

        Args:
            model: The model.
            params: The parameters.
            env_state_batch: The environment state batch containing trajectories.

        Returns:
            A tuple of (loss, metrics).
        """
        # get the log probs of the current model
        prediction = self.apply_actor(model, params, env_state_batch.obs, env_state_batch.commands)
        assert isinstance(prediction, Array)
        log_probs = model.apply(
            variables=params,
            prediction=prediction,
            action=env_state_batch.action_at_prev_step,
            method="actor_calc_log_prob",
        )
        ratio = jnp.exp(log_probs - env_state_batch.action_log_prob_at_prev_step)

        # get the state-value estimates
        values = self.apply_critic(model, params, env_state_batch.obs, env_state_batch.commands)
        assert isinstance(values, Array)
        values = values.squeeze(axis=-1)  # values is (time, env)
        advantages = self._compute_advantages(values, env_state_batch)  # (time, env)

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
        value_pred = self.apply_critic(model, params, env_state_batch.obs, env_state_batch.commands)
        value_pred = value_pred.squeeze(axis=-1)  # (time, env)
        value_loss = 0.5 * jnp.mean((returns - value_pred) ** 2)

        # entropy bonus term
        probs = jax.nn.softmax(prediction)  # TODO: make this live in the model
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
    def _compute_advantages(
        self,
        values: Array,
        env_state_batch: EnvState,
    ) -> Array:
        """Computes the advantages using Generalized Advantage Estimation (GAE)."""
        done = env_state_batch.done
        rewards = env_state_batch.reward

        def scan_fn(carry: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
            d, m = x
            new_carry = d + self.config.gamma * self.config.lam * m * carry
            return new_carry, new_carry

        next_values = jnp.roll(values, shift=-1, axis=0)
        mask = jnp.where(done, 0.0, 1.0)

        # getting td residuals
        deltas = rewards + self.config.gamma * next_values * mask - values

        _, advantages = jax.lax.scan(scan_fn, jnp.zeros_like(deltas[-1]), (deltas[::-1], mask[::-1]))
        return advantages[::-1]

    @eqx.filter_jit
    def _jitted_value_and_grad(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env_state_batch: EnvState,
    ) -> tuple[Array, PyTree]:
        """Jitted version of value_and_grad computation."""

        def loss_fn(p: PyTree) -> Array:
            return self.compute_loss(model, p, env_state_batch)[0]

        return jax.value_and_grad(loss_fn)(params)
