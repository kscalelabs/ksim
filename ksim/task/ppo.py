"""Defines a standard task interface for training a policy."""

import os
from abc import ABC
from dataclasses import dataclass
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
import optax
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PyTree

from ksim.env.types import EnvState
from ksim.model.formulations import ActorCriticModel
from ksim.task.rl import RLConfig, RLTask
from ksim.task.types import PPORolloutTimeLossComponents, RolloutTimeLossComponents
from ksim.utils.jit import legit_jit


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
    learning_rate: float = xax.field(value=1e-3, help="Learning rate for PPO.")
    schedule: str = xax.field(value="adaptive", help="Learning rate schedule 'fixed' | 'adaptive'")
    desired_kl: float = xax.field(value=0.01, help="Desired KL divergence for adaptive LR.")
    max_grad_norm: float = xax.field(value=1.0, help="Maximum gradient norm for clipping.")


Config = TypeVar("Config", bound=PPOConfig)


class PPOTask(RLTask[Config], Generic[Config], ABC):
    """Base class for PPO tasks."""

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

    @legit_jit(static_argnames=["self", "model"])
    def get_rollout_time_loss_components(
        self,
        model: ActorCriticModel,
        params: PyTree,
        trajectory_dataset: EnvState,
    ) -> RolloutTimeLossComponents:
        """Calculating advantages and returns for a rollout."""
        prediction = self.apply_actor(
            model, params, trajectory_dataset.obs, trajectory_dataset.command
        )  # I'm fine with recomputing once to ensure separation of rollout and training logic
        initial_action_log_probs = model.apply(
            variables=params,
            prediction=prediction,
            action=trajectory_dataset.action,
            method="actor_calc_log_prob",
        )
        assert isinstance(initial_action_log_probs, Array)
        initial_values = self.apply_critic(
            model, params, trajectory_dataset.obs, trajectory_dataset.command
        ).squeeze(axis=-1)
        # We squeeze because last dimension is a singleton, advantages expects (batch_dims,)

        advantages = self._compute_advantages(initial_values, trajectory_dataset)
        returns = advantages + initial_values

        if self.config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return PPORolloutTimeLossComponents(
            initial_action_log_probs=jax.lax.stop_gradient(initial_action_log_probs),
            initial_values=jax.lax.stop_gradient(initial_values),
            advantages=jax.lax.stop_gradient(advantages),
            returns=jax.lax.stop_gradient(returns),
        )

    @legit_jit(static_argnames=["self", "model", "optimizer"])
    def model_update(
        self,
        model: ActorCriticModel,
        params: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        env_state_batch: EnvState,
        rollout_time_loss_components: RolloutTimeLossComponents,
    ) -> tuple[PyTree, optax.OptState, Array, FrozenDict[str, Array]]:
        """Update the model parameters."""
        loss_val, metrics, grads = self._jitted_value_and_grad(
            model, params, env_state_batch, rollout_time_loss_components
        )
        print(f"Loss: {loss_val}")
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val, metrics

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

    @legit_jit(static_argnames=["self", "model"])
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

    @legit_jit(static_argnames=["self", "model"])
    def compute_loss(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env_state_batch: EnvState,
        rollout_time_loss_components: RolloutTimeLossComponents,
    ) -> tuple[Array, FrozenDict[str, Array]]:
        """Compute the PPO loss (required by XAX)."""
        # get the log probs of the current model
        prediction = self.apply_actor(model, params, env_state_batch.obs, env_state_batch.command)
        log_probs = model.apply(
            variables=params,
            prediction=prediction,
            action=env_state_batch.action,
            method="actor_calc_log_prob",
        )

        assert isinstance(prediction, Array)
        assert isinstance(log_probs, Array)
        assert isinstance(rollout_time_loss_components, PPORolloutTimeLossComponents)

        log_prob_diff = log_probs - rollout_time_loss_components.initial_action_log_probs
        ratio = jnp.exp(log_prob_diff)

        # get the state-value estimates
        values = self.apply_critic(model, params, env_state_batch.obs, env_state_batch.command)
        assert isinstance(values, Array)
        values = values.squeeze(axis=-1)  # values is (time, env)

        # policy loss with clipping
        policy_loss = -jnp.mean(
            jnp.minimum(
                ratio * rollout_time_loss_components.advantages,
                jnp.clip(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param)
                * rollout_time_loss_components.advantages,
            )
        )

        # value loss term
        # TODO: add clipping
        value_pred = self.apply_critic(model, params, env_state_batch.obs, env_state_batch.command)
        value_pred = value_pred.squeeze(axis=-1)  # (time, env)
        value_loss = 0.5 * jnp.mean((rollout_time_loss_components.returns - value_pred) ** 2)

        # entropy bonus term
        probs = jax.nn.softmax(prediction)  # TODO: make this live in the model
        entropy = -jnp.mean(jnp.sum(jax.scipy.special.entr(probs), axis=-1))
        entropy_loss = -self.config.entropy_coef * entropy

        total_loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss

        metrics_to_log = FrozenDict(
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "total_loss": total_loss,
                "average_ratio": jnp.mean(ratio),
                "average_log_prob_diff": jnp.mean(log_prob_diff),
            }
        )

        use_debug = os.environ.get("DEBUG", "0") == "1"
        if use_debug and jnp.isnan(total_loss):  # should skip compilation
            breakpoint()

        return total_loss, metrics_to_log

    @legit_jit(static_argnames=["self"])
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

        next_values = jnp.roll(values, shift=-1, axis=0)  # TODO: concat not roll...
        mask = jnp.where(done, 0.0, 1.0)

        # getting td residuals
        deltas = rewards + self.config.gamma * next_values * mask - values

        _, advantages = jax.lax.scan(scan_fn, jnp.zeros_like(deltas[-1]), (deltas[::-1], mask[::-1]))
        return advantages[::-1]

    @legit_jit(static_argnames=["self", "model"])
    def _jitted_value_and_grad(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env_state_batch: EnvState,
        rollout_time_loss_components: RolloutTimeLossComponents,
    ) -> tuple[Array, FrozenDict[str, Array], PyTree]:
        """Jitted version of value_and_grad computation."""

        def loss_fn(p: PyTree) -> tuple[Array, FrozenDict[str, Array]]:
            return self.compute_loss(model, p, env_state_batch, rollout_time_loss_components)

        (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return loss_val, metrics, grads
