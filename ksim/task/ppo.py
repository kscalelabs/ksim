"""Defines a standard task interface for training a policy."""

__all__ = [
    "PPOConfig",
    "PPOTask",
    "PPOInputs",
    "PPOVariables",
]

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Mapping, TypeVar

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.task.rl import RLConfig, RLTask, RolloutConstants, RolloutControlVariables, RolloutEnvironmentVariables
from ksim.types import Rewards, SingleTrajectory, Trajectory


@jax.tree_util.register_dataclass
@dataclass
class PPOInputs:
    advantages_t: Array
    value_targets_t: Array
    gae_t: Array
    returns_t: Array


@jax.tree_util.register_dataclass
@dataclass
class PPOVariables:
    log_probs: Array
    values: Array
    entropy: Array | None = None
    aux_losses: Mapping[str, Array] | None = None


@xax.jit(
    static_argnames=[
        "decay_gamma",
        "gae_lambda",
        "normalize_advantages",
        "use_two_step_td_target",
        "monte_carlo_returns",
    ]
)
def compute_ppo_inputs(
    values_t: Array,
    rewards_t: Array,
    dones_t: Array,
    decay_gamma: float,
    gae_lambda: float,
    normalize_advantages: bool = False,
    use_two_step_td_target: bool = False,
    monte_carlo_returns: bool = False,
) -> PPOInputs:
    """Computes the advantages using Generalized Advantage Estimation (GAE)."""

    def returns_scan_fn(returns_t_plus_1: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        """Scanning this computes the returns in reverse order."""
        reward, mask = x
        return_t = reward + decay_gamma * mask * returns_t_plus_1
        return return_t, return_t

    def gae_scan_fn(adv_t_plus_1: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        """Scanning this computes the advantages in reverse order."""
        delta, mask = x
        adv_t = delta + decay_gamma * gae_lambda * mask * adv_t_plus_1
        return adv_t, adv_t

    def compute_gae_and_targets_for_sample(values_t: Array, rewards_t: Array, dones_t: Array) -> PPOInputs:
        # Use the last value as the bootstrap value.
        values_shifted_t = jnp.concatenate([values_t[1:], jnp.expand_dims(values_t[-1], 0)], axis=0)
        mask_t = jnp.where(dones_t, 0.0, 1.0)

        # Compute returns.
        _, returns_t = jax.lax.scan(returns_scan_fn, jnp.zeros_like(rewards_t[-1]), (rewards_t, mask_t), reverse=True)

        # Compute the GAE.
        deltas_t = rewards_t + decay_gamma * values_shifted_t * mask_t - values_t
        _, gae_t = jax.lax.scan(gae_scan_fn, jnp.zeros_like(deltas_t[-1]), (deltas_t, mask_t), reverse=True)

        # Get the value targets.
        value_targets_t = returns_t if monte_carlo_returns else gae_t + values_t

        if not use_two_step_td_target:
            return PPOInputs(
                advantages_t=gae_t,
                value_targets_t=value_targets_t,
                gae_t=gae_t,
                returns_t=returns_t,
            )

        # Apply another TD step to get the value targets.
        value_targets_shifted_t = jnp.concatenate([value_targets_t[1:], value_targets_t[-1:]], axis=0)
        advantages_t = rewards_t + decay_gamma * value_targets_shifted_t * mask_t - values_t

        return PPOInputs(
            advantages_t=advantages_t,
            value_targets_t=value_targets_t,
            gae_t=gae_t,
            returns_t=returns_t,
        )

    # Compute the advantages and value targets for each sample in the batch.
    inputs = compute_gae_and_targets_for_sample(values_t, rewards_t, dones_t)

    if normalize_advantages:
        inputs.advantages_t = inputs.advantages_t / (inputs.advantages_t.std(axis=-1, keepdims=True) + 1e-6)

    return inputs


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
    ppo_inputs: PPOInputs,
    on_policy_variables: PPOVariables,
    off_policy_variables: PPOVariables,
    dones_t: Array,
    *,
    clip_param: float = 0.2,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.008,
    log_clip_value: float = 10.0,
    use_clipped_value_loss: bool = True,
) -> Array:
    """Compute PPO loss.

    Args:
        ppo_inputs: The pre-computed PPO inputs.
        on_policy_variables: The variables for the original policy.
        off_policy_variables: The variables for the new policy.
        dones_t: The termination mask, with shape (T,).
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
            on_policy_variables.log_probs,
            on_policy_variables.values,
            off_policy_variables.log_probs,
            off_policy_variables.values,
            ppo_inputs.advantages_t,
            ppo_inputs.value_targets_t,
            dones_t,
        ]
        + ([] if off_policy_variables.aux_losses is None else list(off_policy_variables.aux_losses.values())),
        prefix_len=1,
    )

    def compute_loss_for_sample(
        on_policy_variables: PPOVariables,
        off_policy_variables: PPOVariables,
        ppo_inputs: PPOInputs,
        dones: Array,
    ) -> Array:
        # Preventing underflow / overflow in calculating the ratio.
        log_ratio = jnp.sum(off_policy_variables.log_probs - on_policy_variables.log_probs, axis=-1)
        ratio = jnp.exp(jnp.clip(log_ratio, -log_clip_value, log_clip_value))
        clipped_ratio = jnp.clip(ratio, 1 - clip_param, 1 + clip_param)
        surrogate_1 = ratio * ppo_inputs.advantages_t
        surrogate_2 = clipped_ratio * ppo_inputs.advantages_t
        policy_objective = jnp.minimum(surrogate_1, surrogate_2)

        # Computes the value loss, with or without clipping.
        if use_clipped_value_loss:
            value_mse = 0.5 * clipped_value_loss(
                target_values=on_policy_variables.values,
                values=off_policy_variables.values,
                value_targets=ppo_inputs.value_targets_t,
                clip_param=clip_param,
            )
        else:
            value_mse = 0.5 * (ppo_inputs.value_targets_t - off_policy_variables.values) ** 2
        value_objective = value_loss_coef * value_mse

        total_objective = policy_objective - value_objective

        # Adds the entropy bonus term, if provided.
        if off_policy_variables.entropy is not None:
            total_objective = total_objective + entropy_coef * off_policy_variables.entropy.mean(axis=-1)

        # Adds any additional auxiliary losses.
        if off_policy_variables.aux_losses is not None:
            for aux_loss_value in off_policy_variables.aux_losses.values():
                total_objective = total_objective + jnp.mean(aux_loss_value)

        # Maximize the objective.
        total_loss = -total_objective

        # Zero out the loss for terminated trajectories.
        total_loss = jnp.where(dones, 0.0, total_loss)

        return total_loss

    par_fn = jax.vmap(compute_loss_for_sample, in_axes=0)

    # Computes the vectorized loss.
    total_loss_t = par_fn(on_policy_variables, off_policy_variables, ppo_inputs, dones_t)

    return total_loss_t


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
        help="Entropy coefficient for PPO: high = more exploration.",
    )
    log_clip_value: float = xax.field(
        value=20.0,
        help="The log clip value for PPO, for numerical stability. For FP16, this should be 10 instead.",
    )
    gamma: float = xax.field(
        value=0.99,
        help="Discount factor for PPO. Higher values mean more weight on future rewards.",
    )
    lam: float = xax.field(
        value=0.95,
        help="Lambda for GAE: high = more bias; low = more variance",
    )
    normalize_advantages: bool = xax.field(
        value=False,
        help="Whether to normalize the advantages.",
    )
    use_two_step_td_target: bool = xax.field(
        value=False,
        help="Whether to use two-step TD targets.",
    )
    monte_carlo_returns: bool = xax.field(
        value=False,
        help="Whether to use Monte Carlo returns.",
    )


Config = TypeVar("Config", bound=PPOConfig)


class PPOTask(RLTask[Config], Generic[Config], ABC):
    """Base class for PPO tasks."""

    @abstractmethod
    def get_ppo_variables(
        self,
        model: PyTree,
        trajectories: Trajectory,
        carry: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[PPOVariables, PyTree]:
        """Gets the variables required for computing PPO loss.

        Args:
            model: The user-provided model.
            trajectories: The batch of trajectories to get probabilities for.
            carry: The carry for the model
            rng: A random seed.

        Returns:
            The PPO variables and the next carry for the model.
        """

    def get_ppo_metrics(
        self,
        trajectories: Trajectory,
        rewards: Rewards,
        loss_t: Array,
        ppo_inputs: PPOInputs,
        on_policy_variables: PPOVariables,
        off_policy_variables: PPOVariables,
    ) -> dict[str, Array]:
        """Gets the metrics to be logged.

        If the metric is a scalar, it will be logged as a scalar. If the
        metric is a tuple, it is assumed to be a distribution in (mean, std)
        format and will be logged as a distribution.

        Args:
            trajectories: The batch of trajectories to get metrics for.
            rewards: The rewards for the trajectories.
            loss_t: The PPO loss value.
            ppo_inputs: The PPO inputs.
            on_policy_variables: The variables for the original policy.
            off_policy_variables: The variables for the new policy.

        Returns:
            A dictionary of metrics to be logged.
        """
        metrics = {
            "loss": loss_t.mean(),
            "log_probs": off_policy_variables.log_probs.mean(0).flatten(),
            "on_policy_log_probs": on_policy_variables.log_probs.mean(0).flatten(),
            "value": off_policy_variables.values.mean(),
            "on_policy_value": on_policy_variables.values.mean(),
            "value_targets": ppo_inputs.value_targets_t.mean(),
            "advantages": ppo_inputs.advantages_t.mean(),
        }
        if off_policy_variables.entropy is not None:
            metrics["entropy"] = off_policy_variables.entropy.mean(0).flatten()
        if off_policy_variables.aux_losses is not None:
            for aux_loss_name, aux_loss_value in off_policy_variables.aux_losses.items():
                metrics[aux_loss_name] = aux_loss_value.mean()
        return metrics

    def get_single_trajectory_metrics(
        self,
        trajectories: Trajectory,
        rewards: Rewards,
        ppo_inputs: PPOInputs,
        loss_t: Array,
        on_policy_variables: PPOVariables,
        off_policy_variables: PPOVariables,
    ) -> dict[str, Array]:
        """Gets the metrics to log for a single trajectory.

        If the metric is a scalar, it will be logged as a scalar. If the
        metric is a tuple, it is assumed to be a distribution in (mean, std)
        format and will be logged as a distribution.

        Args:
            trajectories: The batch of trajectories to get metrics for.
            rewards: The rewards for the trajectories.
            ppo_inputs: The PPO inputs.
            loss_t: The PPO loss value.
            on_policy_variables: The variables for the original policy.
            off_policy_variables: The variables for the new policy.

        Returns:
            A dictionary of metrics to be logged. Each metric should be a tensor
            with shape (T, *).
        """
        metrics = {
            "values": off_policy_variables.values,
            "value_targets": ppo_inputs.value_targets_t,
            "advantages": ppo_inputs.advantages_t,
            "loss": loss_t,
            "gae": ppo_inputs.gae_t,
            "returns": ppo_inputs.returns_t,
        }
        if off_policy_variables.entropy is not None:
            metrics["entropy"] = off_policy_variables.entropy
        if off_policy_variables.aux_losses is not None:
            for aux_loss_name, aux_loss_value in off_policy_variables.aux_losses.items():
                metrics[aux_loss_name] = aux_loss_value
        return metrics

    @xax.jit(static_argnames=["self", "model_static"])
    def get_loss_and_metrics(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        trajectories: Trajectory,
        rewards: Rewards,
        init_carry: PyTree,
        on_policy_variables: PPOVariables,
        rng: PRNGKeyArray,
    ) -> tuple[Array, tuple[xax.FrozenDict[str, Array], SingleTrajectory]]:
        """Computes the PPO loss and additional metrics.

        Args:
            model_arr: The mutable part of the model to optimize.
            model_static: The static part of the model to optimize.
            trajectories: The batch of trajectories to compute the loss and metrics for.
            rewards: The rewards for the trajectories.
            init_carry: The initial carry for the model.
            on_policy_variables: The PPO variables from the on-policy rollout.
            rng: A random seed.

        Returns:
            A tuple containing the loss value as a scalar, a dictionary of
            metrics to log, and the single trajectory to log.
        """
        model = eqx.combine(model_arr, model_static)

        def loss_and_metrics_fn(
            trajectories: Trajectory,
            rewards: Rewards,
            init_carry: PyTree,
            on_policy_variables: PPOVariables,
            rng: PRNGKeyArray,
        ) -> tuple[Array, xax.FrozenDict[str, Array], SingleTrajectory]:
            rng, rng2 = jax.random.split(rng)

            off_policy_scan_fn = functools.partial(self._policy_scan_fn, model=model)
            _, off_policy_variables = jax.lax.scan(off_policy_scan_fn, (rng2, init_carry), trajectories)

            ppo_inputs = compute_ppo_inputs(
                values_t=jax.lax.stop_gradient(off_policy_variables.values),
                rewards_t=rewards.total,
                dones_t=trajectories.done,
                decay_gamma=self.config.gamma,
                gae_lambda=self.config.lam,
                normalize_advantages=self.config.normalize_advantages,
                use_two_step_td_target=self.config.use_two_step_td_target,
                monte_carlo_returns=self.config.monte_carlo_returns,
            )

            loss_t = compute_ppo_loss(
                ppo_inputs=ppo_inputs,
                on_policy_variables=on_policy_variables,
                off_policy_variables=off_policy_variables,
                dones_t=trajectories.done,
                clip_param=self.config.clip_param,
                value_loss_coef=self.config.value_loss_coef,
                entropy_coef=self.config.entropy_coef,
                log_clip_value=self.config.log_clip_value,
                use_clipped_value_loss=self.config.use_clipped_value_loss,
            )

            metrics = self.get_ppo_metrics(
                trajectories=trajectories,
                rewards=rewards,
                loss_t=loss_t,
                ppo_inputs=ppo_inputs,
                on_policy_variables=on_policy_variables,
                off_policy_variables=off_policy_variables,
            )

            single_traj_metrics = self.get_single_trajectory_metrics(
                trajectories=trajectories,
                rewards=rewards,
                loss_t=loss_t,
                ppo_inputs=ppo_inputs,
                on_policy_variables=on_policy_variables,
                off_policy_variables=off_policy_variables,
            )

            single_traj = SingleTrajectory(
                trajectory=trajectories,
                rewards=rewards,
                metrics=xax.FrozenDict(single_traj_metrics),
            )

            # Mean over all non-masked trajectories.
            num_valid = jnp.sum(~trajectories.done)
            loss = loss_t.sum() / (num_valid + 1e-6)

            return loss, xax.FrozenDict(metrics), single_traj

        # Gets the loss and metrics for each trajectory in the batch.
        rngs = jax.random.split(rng, rewards.total.shape[0])
        par_fn = jax.vmap(loss_and_metrics_fn, in_axes=0)
        loss, metrics, single_traj = par_fn(trajectories, rewards, init_carry, on_policy_variables, rngs)

        # Only take the last trajectory.
        single_traj = jax.tree.map(lambda x: x[-1], single_traj)

        return loss.mean(), (metrics, single_traj)

    @xax.jit(static_argnames=["self", "model_static"])
    def _get_loss_metrics_and_grads(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        trajectories: Trajectory,
        rewards: Rewards,
        init_carry: PyTree,
        on_policy_variables: PPOVariables,
        rng: PRNGKeyArray,
    ) -> tuple[dict[str, Array], SingleTrajectory, PyTree]:
        loss_fn = jax.grad(self.get_loss_and_metrics, argnums=0, has_aux=True)
        loss_fn = xax.jit(static_argnums=[1])(loss_fn)
        grads, (metrics, single_traj) = loss_fn(
            model_arr,
            model_static,
            trajectories,
            rewards,
            init_carry,
            on_policy_variables,
            rng,
        )
        return metrics, single_traj, grads

    @xax.jit(static_argnames=["self", "model_static", "optimizer"])
    def _single_step(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectories: Trajectory,
        rewards: Rewards,
        init_carry: PyTree,
        on_policy_variables: PPOVariables,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, xax.FrozenDict[str, Array], SingleTrajectory]:
        ppo_metrics, single_traj, grads = self._get_loss_metrics_and_grads(
            model_arr=model_arr,
            model_static=model_static,
            trajectories=trajectories,
            rewards=rewards,
            init_carry=init_carry,
            on_policy_variables=on_policy_variables,
            rng=rng,
        )

        new_model_arr, new_opt_state, grad_metrics = self.apply_gradients_with_clipping(
            model_arr=model_arr,
            grads=grads,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        return new_model_arr, new_opt_state, xax.FrozenDict(dict(ppo_metrics) | dict(grad_metrics)), single_traj

    def _policy_scan_fn(
        self,
        carry: tuple[PRNGKeyArray, PyTree],
        xt: Trajectory,
        model: PyTree,
    ) -> tuple[tuple[PRNGKeyArray, PyTree], PPOVariables]:
        rng, model_carry = carry
        rng, ppo_rng, carry_rng = jax.random.split(rng, 3)
        ppo_vars, model_carry = self.get_ppo_variables(model, xt, model_carry, ppo_rng)

        # Conditionally reset the model carry on termination.
        model_carry = jax.lax.cond(
            xt.done,
            lambda: self.get_initial_carry(carry_rng),
            lambda: model_carry,
        )

        return (rng, model_carry), ppo_vars

    def update_model(
        self,
        optimizer: optax.GradientTransformation,
        opt_state: optax.OptState,
        trajectories: Trajectory,
        rewards: Rewards,
        rollout_constants: RolloutConstants,
        rollout_env_vars: RolloutEnvironmentVariables,
        rollout_ctrl_vars: RolloutControlVariables,
        rng: PRNGKeyArray,
    ) -> tuple[PyTree, optax.OptState, xax.FrozenDict[str, Array], SingleTrajectory]:
        """Runs PPO updates on a given set of trajectory batches.

        Args:
            optimizer: The optimizer to use.
            opt_state: The optimizer state.
            trajectories: The trajectories to update the model on.
            rewards: The rewards for the trajectories.
            rollout_constants: The constant inputs into the rollout.
            rollout_env_vars: The environment variables inputs into the rollout.
            rollout_ctrl_vars: The control variables inputs into the rollout.
            rng: A random seed.

        Returns:
            A tuple containing the updated parameters, optimizer state, metrics,
            and the single trajectory to log.
        """
        # Shuffling causes a strange kernel caching issue on GPUs.
        # rng, indices_rng = jax.random.split(rng)
        # indices = jax.random.permutation(indices_rng, trajectories.done.shape[0])
        indices = jnp.arange(trajectories.done.shape[0])
        indices = indices.reshape(self.num_batches, self.batch_size)

        model = eqx.combine(rollout_ctrl_vars.model_arr, rollout_constants.model_static)

        def on_policy_scan_fn(rng: PRNGKeyArray, trajectory: Trajectory, model_carry: PyTree) -> PPOVariables:
            rng, policy_vars_rng = jax.random.split(rng)
            on_policy_scan_fn = functools.partial(self._policy_scan_fn, model=model)
            (rng, _), on_policy_variables = jax.lax.scan(on_policy_scan_fn, (policy_vars_rng, model_carry), trajectory)
            return on_policy_variables

        def on_policy_batch_scan_fn(carry: PRNGKeyArray, xt: Array) -> tuple[PRNGKeyArray, PPOVariables]:
            rng, prng = jax.random.split(carry)
            policy_vars_rng = jax.random.split(prng, self.batch_size)
            trajectory_batch = jax.tree.map(lambda x: x[xt], trajectories)
            carry_batch = jax.tree.map(lambda x: x[xt], rollout_env_vars.carry)
            scan_fn = jax.vmap(on_policy_scan_fn, in_axes=0)
            on_policy_variables = scan_fn(policy_vars_rng, trajectory_batch, carry_batch)
            return rng, on_policy_variables

        rng, on_policy_variables = jax.lax.scan(on_policy_batch_scan_fn, rng, indices)

        def flatten_fn(x: Array) -> Array:
            return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])

        on_policy_variables = jax.tree.map(flatten_fn, on_policy_variables)

        # Loops over the trajectory batches and applies gradient updates.
        def scan_fn(
            carry: tuple[PyTree, optax.OptState, PRNGKeyArray],
            xt: Array,
        ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], tuple[xax.FrozenDict[str, Array], SingleTrajectory]]:
            model_arr, opt_state, rng = carry
            rng, batch_rng = jax.random.split(rng)

            # Gets the current batch of trajectories and rewards.
            trajectory_batch = jax.tree.map(lambda x: x[xt], trajectories)
            reward_batch = jax.tree.map(lambda x: x[xt], rewards)
            carry_batch = jax.tree.map(lambda x: x[xt], rollout_env_vars.carry)
            on_policy_variables_batch = jax.tree.map(lambda x: x[xt], on_policy_variables)

            model_arr, opt_state, metrics, single_traj = self._single_step(
                model_arr=model_arr,
                model_static=rollout_constants.model_static,
                optimizer=optimizer,
                opt_state=opt_state,
                trajectories=trajectory_batch,
                rewards=reward_batch,
                init_carry=carry_batch,
                on_policy_variables=on_policy_variables_batch,
                rng=batch_rng,
            )

            return (model_arr, opt_state, rng), (metrics, single_traj)

        # Applines N steps of gradient updates.
        def batch_scan_fn(
            carry: tuple[PyTree, optax.OptState, PRNGKeyArray],
            _: None,
        ) -> tuple[tuple[PyTree, optax.OptState, PRNGKeyArray], tuple[xax.FrozenDict[str, Array], SingleTrajectory]]:
            arr, opt_state, rng = carry
            carry = (arr, opt_state, rng)
            carry, (metrics, single_traj) = jax.lax.scan(scan_fn, carry, indices)

            # Get the last trajectory.
            single_traj = jax.tree.map(lambda x: x[-1], single_traj)

            return carry, (metrics, single_traj)

        carry = (rollout_ctrl_vars.model_arr, opt_state, rng)

        # Applies gradient updates.
        carry, (metrics, single_traj) = jax.lax.scan(batch_scan_fn, carry, length=self.config.num_passes)

        # Get the last trajectory.
        single_traj = jax.tree.map(lambda x: x[-1], single_traj)

        model_arr, opt_state, _ = carry
        return model_arr, opt_state, metrics, single_traj
