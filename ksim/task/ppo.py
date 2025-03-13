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
from xax.task.mixins.train import Batch, Output

from ksim.env.types import EnvState
from ksim.model.distributions import GaussianDistribution
from ksim.model.formulations import ActorCriticAgent, update_actor_critic_normalization
from ksim.task.loss_helpers import compute_returns
from ksim.task.rl import RLConfig, RLTask
from ksim.task.types import PPORolloutTimeLossComponents, RolloutTimeLossComponents
from ksim.utils.constants import EPSILON
from ksim.utils.jit import legit_jit


@jax.tree_util.register_dataclass
@dataclass
class PPOConfig(RLConfig):
    # For the CLIP term (see Schulman et al. 2017)
    clip_param: float = xax.field(value=0.2, help="Clipping parameter for PPO.")

    # For the Value Function (VF) term
    value_loss_coef: float = xax.field(value=0.5, help="Value loss coefficient for PPO.")
    use_clipped_value_loss: bool = xax.field(value=True, help="Whether to use clipped value loss.")

    # For the entropy bonus term
    entropy_coef: float = xax.field(value=0.008, help="Entropy coefficient for PPO.")

    # For the GAE computation
    gamma: float = xax.field(value=0.99, help="Discount factor for PPO")
    lam: float = xax.field(value=0.95, help="Lambda for GAE: high = more bias; low = more variance")

    # General training parameters
    learning_rate: float = xax.field(value=1e-4, help="Learning rate for PPO.")
    max_grad_norm: float = xax.field(value=0.5, help="Maximum gradient norm for clipping.")

    # Normalization parameters
    scale_rewards: bool = xax.field(
        value=False,
        help="Whether to scale rewards, see Engstrom, Ilyas, et al., (2020).",
    )
    normalize_advantage: bool = xax.field(value=True, help="Whether to normalize advantages.")
    normalize_advantage_in_minibatch: bool = xax.field(
        value=False,
        help="Whether to normalize advantages at the minibatch level as per OpenAI baselines.",
    )
    # NOTE: if scale_rewards is True, advantages will still get its own normalization
    reward_scaling_alpha: float = xax.field(
        value=0.0003,
        help="Rate at which to update reward scaling online as per Hessel, Soyer, et al., (2018).",
    )
    obs_norm_alpha: float = xax.field(
        value=0.0003,
        help="Rate at which to update observation norm stats, Andrychowicz (2021) and Duan (2016).",
    )
    # NOTE: as per recommendations, going to enforce observation normalization.
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

    ########################
    # Implementing RL Task #
    ########################

    # TODO from ML eventually we should create
    def get_init_actor_carry(self) -> Array | None:
        """Get the actor carry state."""
        raise NotImplementedError("Not implemented at the base PPO class.")

    def get_init_critic_carry(self) -> Array | None:
        """Get the critic carry state."""
        raise NotImplementedError("Not implemented at the base PPO class.")

    @legit_jit(static_argnames=["self", "model"])
    def get_rollout_time_loss_components(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
        trajectory_dataset: EnvState,
    ) -> RolloutTimeLossComponents:
        """Calculating advantages and returns for a rollout."""

        # we recompute here because we update the normalization stats before the
        # minibatch training loop
        prediction = self.apply_actor(
            model,
            variables,
            trajectory_dataset.obs,
            trajectory_dataset.command,
        )

        initial_action_log_probs = model.apply(
            variables=variables,
            prediction=prediction,
            action=trajectory_dataset.action,
            method="actor_calc_log_prob",
        )
        assert isinstance(initial_action_log_probs, Array)

        initial_values = self.apply_critic(
            model,
            variables,
            trajectory_dataset.obs,
            trajectory_dataset.command,
        ).squeeze(axis=-1)

        advantages, value_targets = self._compute_advantages_and_value_targets(
            variables, initial_values, trajectory_dataset
        )

        # we decouple the computation of returns from the value targets
        returns = compute_returns(
            rewards=trajectory_dataset.reward,
            dones=trajectory_dataset.done,
            gamma=self.config.gamma,
        )

        # normalizing at the trajectory dataset level
        if self.config.normalize_advantage and not self.config.normalize_advantage_in_minibatch:
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPSILON)

        return PPORolloutTimeLossComponents(
            initial_action_log_probs=jax.lax.stop_gradient(initial_action_log_probs),
            initial_values=jax.lax.stop_gradient(initial_values),
            advantages=jax.lax.stop_gradient(advantages),
            value_targets=jax.lax.stop_gradient(value_targets),
            returns=jax.lax.stop_gradient(returns),
        )

    def _clipped_value_loss(
        self,
        target_values: Array,
        values: Array,
        value_targets: Array,
    ) -> Array:
        """Compute the clipped value loss."""
        value_clipped = target_values + (values - target_values).clip(
            -self.config.clip_param, self.config.clip_param
        )
        clipped_error = value_clipped - value_targets
        error = values - value_targets
        return jnp.maximum(error**2, clipped_error**2).mean()

    def model_update(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        env_state_batch: EnvState,
        rollout_time_loss_components: RolloutTimeLossComponents,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, Array, FrozenDict[str, Array]]:
        """Returns the updated parameters, optimizer state, loss value, and metrics."""
        loss_val, metrics, grads = self.loss_metrics_grads(
            model, variables, env_state_batch, rollout_time_loss_components, rng
        )

        # while other variables might be present in comp graph, only update params...
        params = variables["params"]
        grads = grads["params"]

        grad_norm = optax.global_norm(grads)
        assert isinstance(grad_norm, Array)
        metrics["loss"] = loss_val
        metrics["grad_norm"] = grad_norm
        frozen_metrics: FrozenDict[str, Array] = FrozenDict(metrics)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss_val, frozen_metrics

    def get_optimizer(self) -> optax.GradientTransformation:
        """Get the optimizer: handled by XAX."""
        return optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )

    def update_input_normalization_stats(
        self,
        variables: PyTree,
        trajectories_dataset: EnvState,
        initial_step: bool,
    ) -> PyTree:
        """Update the input normalization parameters."""
        obs_norm_alpha = self.config.obs_norm_alpha if initial_step else 0.0
        returns_norm_alpha = self.config.reward_scaling_alpha if initial_step else 0.0
        return update_actor_critic_normalization(
            variables=variables,
            returns_norm_alpha=returns_norm_alpha,
            obs_norm_alpha=obs_norm_alpha,
            trajectories_dataset=trajectories_dataset,
            gamma=self.config.gamma,
        )

    # Pass-through abstract methods:
    # `get_environment`

    ######################
    # Training Utilities #
    ######################

    def apply_critic(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
    ) -> Array:
        """Apply the critic model to inputs. Used by all actor-critic tasks.

        TODO: it might be worth creating another Task abstraction that requires `apply_critic`
        """
        res = model.apply(variables, obs=obs, cmd=cmd, method="critic")
        assert isinstance(res, Array)
        return res

    def compute_ppo_loss(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
        env_state_batch: EnvState,
        rollout_time_loss_components: RolloutTimeLossComponents,
        rng: PRNGKeyArray,
    ) -> tuple[Array, dict[str, Array]]:
        """Compute the PPO loss."""
        # get the log probs of the current model
        prediction = self.apply_actor(
            model, variables, env_state_batch.obs, env_state_batch.command
        )
        log_probs = model.apply(
            variables=variables,
            prediction=prediction,
            action=env_state_batch.action,
            method="actor_calc_log_prob",
        )

        assert isinstance(prediction, Array)
        assert isinstance(log_probs, Array)
        assert isinstance(rollout_time_loss_components, PPORolloutTimeLossComponents)

        log_prob_diff = log_probs - rollout_time_loss_components.initial_action_log_probs
        # Add numerical stability clipping
        # log_prob_diff = jnp.clip(log_prob_diff, -20.0, 20.0)  # prevents exp() from exploding
        ratio = jnp.exp(log_prob_diff)
        # ratio = jnp.clip(ratio, 0.0, 10.0)  # prevents extreme ratios

        # get the state-value estimates
        values = self.apply_critic(model, variables, env_state_batch.obs, env_state_batch.command)
        assert isinstance(values, Array)
        values = values.squeeze(axis=-1)  # values is (time, env)

        advantages = rollout_time_loss_components.advantages

        # worth noting that Andrychowicz (2021) found little advantage to
        # minibatch normalization.
        if self.config.normalize_advantage and self.config.normalize_advantage_in_minibatch:
            advantages = (advantages - advantages.mean()) / (advantages.std() + EPSILON)

        # policy loss with clipping
        clipped_ratio = jnp.clip(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param)
        policy_objective = jnp.mean(
            jnp.minimum(
                ratio * advantages,
                clipped_ratio * advantages,
            )
        )
        value_targets = rollout_time_loss_components.value_targets

        # value loss term
        value_pred = self.apply_critic(
            model, variables, env_state_batch.obs, env_state_batch.command
        )
        value_pred = value_pred.squeeze(axis=-1)  # (time, env)
        value_mse = jax.lax.cond(
            self.config.use_clipped_value_loss,
            lambda: 0.5
            * self._clipped_value_loss(
                target_values=rollout_time_loss_components.initial_values,
                values=value_pred,
                value_targets=value_targets,
            ),
            lambda: 0.5 * jnp.mean((value_targets - value_pred) ** 2),
        )
        value_objective = self.config.value_loss_coef * value_mse

        # entropy bonus term
        entropies = model.distribution.entropy(prediction, rng)
        entropy_objective = self.config.entropy_coef * jnp.mean(entropies)

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
            "advantage_norm_mean": jnp.mean(rollout_time_loss_components.advantages),
            "action_mean": jnp.mean(env_state_batch.action),
            "action_std": jnp.std(env_state_batch.action),
            "action_max": jnp.max(env_state_batch.action),
            "action_min": jnp.min(env_state_batch.action),
            "prediction_mean": jnp.mean(prediction),
            "prediction_std": jnp.std(prediction),
            "log_prob_mean": jnp.mean(log_probs),
            "log_prob_max": jnp.max(log_probs),
            "log_prob_min": jnp.min(log_probs),
            "values_std": jnp.std(values),
            "values_mean": jnp.mean(values),
        }

        if isinstance(model.distribution, GaussianDistribution):
            mu, sigma = model.distribution.get_mean_std(prediction)
            metrics_to_log["prediction_mu_mean"] = jnp.mean(mu)
            metrics_to_log["prediction_sigma_mean"] = jnp.mean(sigma)
            metrics_to_log["prediction_sigma_min"] = jnp.min(sigma)
            metrics_to_log["prediction_sigma_max"] = jnp.max(sigma)

        use_debug = os.environ.get("DEBUG", "0") == "1"
        if use_debug and jnp.isnan(total_loss):  # should skip compilation
            breakpoint()

        return total_loss, metrics_to_log

    def compute_loss(self, model: PyTree, batch: Batch, output: Output) -> Array:
        """Implementation to satisfy the TrainMixin interface.

        In reinforcement learning context, this method is not expected to be called
        directly by the XAX framework, but we provide an implementation to fulfill
        the interface requirements.
        """
        # For RL tasks, this shouldn't be called by the XAX framework
        # But we need to provide an implementation with the correct signature
        raise NotImplementedError(
            "Direct compute_loss from TrainMixin is not expected to be called in RL tasks. "
            "PPO tasks use model_update and loss_metrics_grads instead."
        )

    def _compute_advantages_and_value_targets(
        self,
        variables: PyTree,
        values: Array,
        env_state_batch: EnvState,
    ) -> tuple[Array, Array]:
        """Computes the advantages using Generalized Advantage Estimation (GAE).

        Note that some of this logic is NOT stock PPO, using Brax's
        implementation of PPO as a reference.
        """
        done = env_state_batch.done
        rewards = env_state_batch.reward

        if self.config.scale_rewards:  # as per Engstrom, Ilyas, et al., (2020)
            returns_std = variables["normalization"]["returns_std"]
            rewards = rewards / (returns_std + EPSILON)

        def scan_fn(adv_t_plus_1: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
            """Scanning this computes the advantages in reverse order."""
            delta, mask = x
            adv_t = delta + self.config.gamma * self.config.lam * mask * adv_t_plus_1
            return adv_t, adv_t

        values_shifted = jnp.concatenate([values[1:], values[-1:]], axis=0)
        # just repeating the last value for the last time step (should zero it out mathematically)
        mask = jnp.where(done, 0.0, 1.0)

        # getting td residuals
        deltas = rewards + self.config.gamma * values_shifted * mask - values

        _, gae = jax.lax.scan(scan_fn, jnp.zeros_like(deltas[-1]), (deltas, mask), reverse=True)
        value_targets = jnp.add(gae, values)
        # gae is the result from stock GAE...

        # Following Brax and applying another TD step to get the value targets
        # TODO: experiment with original GAE & value targets
        value_targets_shifted = jnp.concatenate([value_targets[1:], value_targets[-1:]], axis=0)
        advantages = rewards + self.config.gamma * value_targets_shifted * mask - values

        return advantages, value_targets

    def loss_metrics_grads(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
        env_state_batch: EnvState,
        rollout_time_loss_components: RolloutTimeLossComponents,
        rng: PRNGKeyArray,
    ) -> tuple[Array, dict[str, Array], PyTree]:
        """Jitted version of value_and_grad computation."""

        def loss_fn(p: PyTree) -> tuple[Array, dict[str, Array]]:
            return self.compute_ppo_loss(
                model, p, env_state_batch, rollout_time_loss_components, rng
            )

        (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables)
        return loss_val, metrics, grads
