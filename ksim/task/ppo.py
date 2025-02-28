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
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.base_env import BaseEnv, EnvState
from ksim.env.types import Minibatch
from ksim.model.formulations import ActorCriticModel
from ksim.task.rl import RLConfig, RLTask
from ksim.utils.jit import legit_jit


_EPS = 1e-8


@jax.tree_util.register_dataclass
@dataclass
class PPOConfig(RLConfig):
    # For the CLIP term (see Schulman et al. 2017)
    clip_param: float = xax.field(value=0.2, help="Clipping parameter for PPO.")
    normalize_returns: bool = xax.field(value=True, help="Whether to normalize returns.")

    # For the Value Function (VF) term
    value_loss_coef: float = xax.field(value=0.5, help="Value loss coefficient for PPO.")
    use_clipped_value_loss: bool = xax.field(value=True, help="Whether to use clipped value loss.")

    # For the entropy bonus term
    entropy_coef: float = xax.field(value=0.08, help="Entropy coefficient for PPO.")

    # For the GAE computation
    gamma: float = xax.field(value=0.99, help="Discount factor for PPO")
    lam: float = xax.field(value=0.95, help="Lambda for GAE: high = more bias; low = more variance")

    # General training parameters
    minibatch_size: int = xax.field(value=50, help="Equals to the number of updates per trajectory")
    num_mini_batches: int = xax.field(value=10, help="Number of mini-batches per PPO epoch.")
    learning_rate: float = xax.field(value=1e-4, help="Learning rate for PPO.")
    max_grad_norm: float = xax.field(value=0.5, help="Maximum gradient norm for clipping.")


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

    # @legit_jit(static_argnames=["self", "model", "env"], compile_timeout=10)
    def get_trajectory_batch(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env: BaseEnv,
        rng: PRNGKeyArray,
    ) -> EnvState:
        """Rollout the model for a given number of steps, dims (num_steps, num_envs, ...)"""

        env_state_batch = env.unroll_trajectories(
            model=model,
            params=params,
            rng=rng,
            num_steps=self.max_trajectory_steps,
            num_envs=self.config.num_envs,
        )

        return env_state_batch

    @legit_jit(static_argnames=["self", "model", "optimizer"])
    def model_update(
        self,
        model: ActorCriticModel,
        params: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        env_state_batch: EnvState,
    ) -> tuple[PyTree, optax.OptState, Array, dict[str, Array]]:
        """Update the model parameters."""
        loss_val, metrics, grads = self._jitted_value_and_grad(model, params, env_state_batch)
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

    # @legit_jit(static_argnames=["self", "model"])
    def update_minibatch(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env_state_batch: EnvState,
    ) -> Minibatch:
        """Update the model parameters."""
        # get the state-value estimates
        values = self.apply_critic(model, params, env_state_batch.obs, env_state_batch.command)
        assert isinstance(values, Array)
        values = values.squeeze(axis=-1)  # values is (time, env)
        advantages = self._compute_advantages(values, env_state_batch)  # (time, env)

        returns = advantages + values

        return Minibatch(
            obs=env_state_batch.obs,
            command=env_state_batch.command,
            action=env_state_batch.action,
            reward=env_state_batch.reward,
            done=env_state_batch.done,
            timestep=env_state_batch.timestep,
            rng=env_state_batch.rng,
            action_log_prob=env_state_batch.action_log_prob,
            advantages=advantages,
            returns=returns,
            values=values,
        )

    # @legit_jit(static_argnames=["self", "model"])
    def get_minibatches(
        self,
        trajectories: EnvState,
        model: ActorCriticModel,
        params: PyTree,
        rng: PRNGKeyArray,
    ) -> list[Minibatch]:
        """Get the minibatches for the current environment."""
        batch = self.update_minibatch(model, params, trajectories)

        rng, _rng = jax.random.split(rng)
        batch_size = self.config.num_mini_batches * self.config.minibatch_size
        assert batch_size == self.config.num_envs * self.max_trajectory_steps

        permutation = jax.random.permutation(_rng, batch_size)
        # skipp flattening if work on one env
        # batch = jax.lax.cond(
        #     len(batch.action.shape) == 2,
        #     lambda b: b,
        #     lambda b: jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), b),
        #     operand=batch,
        # )
        # permute
        permutted_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
        # divide
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, [self.config.num_mini_batches, -1] + list(x.shape[1:])),
            permutted_batch,
        )

        minibatch_list = [
            Minibatch(
                obs=FrozenDict({k: minibatches.obs[k][i] for k in minibatches.obs}),
                reward=minibatches.reward[i],
                done=minibatches.done[i],
                command=FrozenDict({k: minibatches.command[k][i] for k in minibatches.command}),
                timestep=minibatches.timestep[i],
                rng=minibatches.rng[i],
                action=minibatches.action[i],
                action_log_prob=minibatches.action_log_prob[i],
                advantages=minibatches.advantages[i],
                returns=minibatches.returns[i],
                values=minibatches.values[i],
            )
            for i in range(self.config.num_mini_batches)
        ]

        return minibatch_list

    @legit_jit(static_argnames=["self", "model"])
    def compute_loss(
        self,
        model: ActorCriticModel,
        params: PyTree,
        minibatch: EnvState,
    ) -> tuple[Array, dict[str, Array]]:
        """Compute the PPO loss (required by XAX)."""
        # get the log probs of the current model
        prediction = self.apply_actor(model, params, minibatch.obs, minibatch.command)
        assert isinstance(prediction, Array)
        log_probs = model.apply(
            variables=params,
            prediction=prediction,
            action=minibatch.action,
            method="actor_calc_log_prob",
        )
        log_prob_diff = log_probs - minibatch.action_log_prob
        ratio = jnp.exp(log_prob_diff)
        advantages = (minibatch.advantages - minibatch.advantages.mean()) / (
            minibatch.advantages.std() + _EPS
        )

        # normalize returns
        returns = jax.lax.cond(
            self.config.normalize_returns,
            lambda: (minibatch.returns - minibatch.returns.mean())/ (minibatch.returns.std() + _EPS),
            lambda: minibatch.returns,
        )

        # policy loss with clipping
        policy_loss = -jnp.mean(
            jnp.minimum(
                ratio * advantages,
                jnp.clip(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param)
                * minibatch.advantages,
            )
        )

        # value loss with clipping
        value_pred = self.apply_critic(model, params, minibatch.obs, minibatch.command)
        value_pred = value_pred.squeeze(axis=-1)  # (time, env)

        value_loss = jax.lax.cond(
            self.config.use_clipped_value_loss,
            lambda: self._clipped_value_loss(
                target_values=minibatch.values,
                values=value_pred,
                returns=returns,
            ),
            lambda: 0.5 * jnp.mean((returns - value_pred) ** 2),
        )

        # entropy bonus term
        probs = jax.nn.softmax(prediction)
        entropy = -jnp.mean(jnp.sum(jax.scipy.special.entr(probs), axis=-1))
        entropy_loss = -self.config.entropy_coef * entropy

        total_loss = policy_loss + self.config.value_loss_coef * value_loss + entropy_loss

        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "total_loss": total_loss,
            "average_ratio": jnp.mean(ratio),
            "average_log_prob_diff": jnp.mean(log_prob_diff),
        }

        use_debug = os.environ.get("DEBUG", "0") == "1"
        if use_debug and jnp.isnan(total_loss):
            breakpoint()

        return total_loss, metrics

    @legit_jit(static_argnames=["self"])
    def _clipped_value_loss(
        self,
        target_values: Array,
        values: Array,
        returns: Array,
    ) -> Array:
        """Compute the clipped value loss.
        Since we do one right now update per batch,
        the target values are the same as the values.
        """
        value_clipped = target_values + jnp.clip(
            values - target_values, -self.config.clip_param, self.config.clip_param
        )
        clipped_error = value_clipped - returns
        error = values - returns
        return 0.5 * jnp.mean(jnp.maximum(error**2, clipped_error**2))

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

        _, advantages = jax.lax.scan(
            scan_fn, jnp.zeros_like(deltas[-1]), (deltas[::-1], mask[::-1])
        )
        return advantages[::-1]

    @legit_jit(static_argnames=["self", "model"])
    def _jitted_value_and_grad(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env_state_batch: EnvState,
    ) -> tuple[Array, dict[str, Array], PyTree]:
        """Jitted version of value_and_grad computation."""

        def loss_fn(p: PyTree) -> tuple[Array, dict[str, Array]]:
            return self.compute_loss(model, p, env_state_batch)

        (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        return loss_val, metrics, grads
