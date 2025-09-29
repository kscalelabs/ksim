"""Defines a task for training a policy using PPO."""

__all__ = [
    "PPOConfig",
    "PPOTask",
    "PPOInputs",
    "PPOVariables",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Generic, Mapping, TypeVar

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.debugging import JitLevel
from ksim.task.rl import RLConfig, RLLoopCarry, RLLoopConstants, RLTask
from ksim.types import RewardState, Trajectory


@jax.tree_util.register_dataclass
@dataclass
class PPOInputs:
    advantages_bt: Array
    value_targets_bt: Array
    gae_bt: Array
    returns_bt: Array


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
        "monte_carlo_returns",
    ],
    jit_level=JitLevel.RL_CORE,
)
def compute_ppo_inputs(
    values_bt: Array,
    rewards_bt: Array,
    dones_bt: Array,
    successes_bt: Array,
    decay_gamma: float,
    gae_lambda: float,
    normalize_advantages: bool = False,
    monte_carlo_returns: bool = False,
) -> PPOInputs:
    """Computes the advantages using Generalized Advantage Estimation (GAE)."""

    def returns_and_gae_scan_fn(
        x_t_plus_1: tuple[Array, Array],
        x: tuple[Array, Array, Array],
    ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        """Scanning this computes the returns in reverse order."""
        reward, delta, mask = x
        returns_t_plus_1, adv_t_plus_1 = x_t_plus_1
        return_t = reward + decay_gamma * mask * returns_t_plus_1
        adv_t = delta + decay_gamma * gae_lambda * mask * adv_t_plus_1
        return (return_t, adv_t), (return_t, adv_t)

    # values_shifted_t is V(s_{t+1}) for t < T_rollout, and V(s_T) for t = T_rollout
    # Uses the last value of the trajectory as the bootstrap value.
    values_shifted_t = jnp.concatenate([values_bt[..., 1:], values_bt[..., -1:]], axis=-1)

    # 1-step bootstrap on successful terminations.
    trunc_mask_bt = jnp.where(successes_bt, 1.0, 0.0)
    bootstrapped_rewards_bt = (1 - decay_gamma) * rewards_bt + decay_gamma * values_bt * trunc_mask_bt

    mask_bt = jnp.where(dones_bt, 0.0, 1.0)
    deltas_bt = bootstrapped_rewards_bt + decay_gamma * values_shifted_t * mask_bt - values_bt

    # Transpose for scan.
    bootstrapped_rewards_tb = jnp.swapaxes(bootstrapped_rewards_bt, -1, -2)
    deltas_tb = jnp.swapaxes(deltas_bt, -1, -2)
    mask_tb = jnp.swapaxes(mask_bt, -1, -2)

    # Compute returns and GAE.
    _, (returns_tb, gae_tb) = xax.scan(
        returns_and_gae_scan_fn,
        (jnp.zeros_like(rewards_bt[..., -1]), jnp.zeros_like(deltas_bt[..., -1])),
        (bootstrapped_rewards_tb, deltas_tb, mask_tb),
        reverse=True,
        jit_level=JitLevel.RL_CORE,
    )

    # Transpose back.
    returns_bt = jnp.swapaxes(returns_tb, -1, -2)
    gae_bt = jnp.swapaxes(gae_tb, -1, -2)

    # Get the value targets.
    value_targets_bt = returns_bt if monte_carlo_returns else gae_bt + values_bt

    inputs = PPOInputs(
        advantages_bt=gae_bt,
        value_targets_bt=value_targets_bt,
        gae_bt=gae_bt,
        returns_bt=returns_bt,
    )

    if normalize_advantages:
        adv_mean = inputs.advantages_bt.mean(keepdims=True)
        adv_std = inputs.advantages_bt.std(keepdims=True).clip(min=1e-6)
        inputs.advantages_bt = (inputs.advantages_bt - adv_mean) / adv_std

    return jax.lax.stop_gradient(inputs)


@xax.jit(static_argnames=["clip_param"], jit_level=JitLevel.HELPER_FUNCTIONS)
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


@xax.jit(
    static_argnames=[
        "clip_param",
        "value_loss_coef",
        "entropy_coef",
        "kl_coef",
        "log_clip_value",
        "use_clipped_value_loss",
    ],
    jit_level=JitLevel.RL_CORE,
)
def compute_ppo_loss(
    ppo_inputs: PPOInputs,
    on_policy_variables: PPOVariables,
    off_policy_variables: PPOVariables,
    *,
    clip_param: float,
    value_loss_coef: float,
    entropy_coef: float,
    kl_coef: float,
    log_clip_value: float,
    use_clipped_value_loss: bool,
) -> dict[str, Array]:
    """Compute PPO loss.

    Args:
        ppo_inputs: The pre-computed PPO inputs.
        on_policy_variables: The variables for the original policy.
        off_policy_variables: The variables for the new policy.
        clip_param: The clip parameter for PPO.
        value_loss_coef: The value loss coefficient for PPO.
        entropy_coef: The entropy coefficient for PPO.
        kl_coef: The KL divergence coefficient for PPO, to discourage large
            changes in the policy.
        log_clip_value: The log clip value for PPO, for numerical stability.
        use_clipped_value_loss: Whether to use clipped value loss.

    Returns:
        A dictionary of the various loss terms, each with shape (T,).
    """
    chex.assert_equal_shape_prefix(
        [
            on_policy_variables.log_probs,
            on_policy_variables.values,
            off_policy_variables.log_probs,
            off_policy_variables.values,
            ppo_inputs.advantages_bt,
            ppo_inputs.value_targets_bt,
        ]
        + ([] if off_policy_variables.aux_losses is None else list(off_policy_variables.aux_losses.values())),
        prefix_len=1,
    )

    # The following should not have any singleton dimensions.
    chex.assert_rank(on_policy_variables.values, 2)
    chex.assert_rank(off_policy_variables.values, 2)
    chex.assert_rank(ppo_inputs.advantages_bt, 2)
    chex.assert_rank(ppo_inputs.value_targets_bt, 2)
    if off_policy_variables.aux_losses is not None:
        for aux_loss in off_policy_variables.aux_losses.values():
            chex.assert_rank(aux_loss, 2)

    # Log probs should have an extra dimension for the number of actions.
    chex.assert_rank(on_policy_variables.log_probs, 3)
    chex.assert_rank(off_policy_variables.log_probs, 3)
    if off_policy_variables.entropy is not None:
        chex.assert_rank(off_policy_variables.entropy, 3)

    # Preventing underflow / overflow in calculating the ratio.
    log_ratio = jnp.sum(off_policy_variables.log_probs - on_policy_variables.log_probs, axis=-1)
    ratio = jnp.exp(jnp.clip(log_ratio, -log_clip_value, log_clip_value))
    clipped_ratio = jnp.clip(ratio, 1 - clip_param, 1 + clip_param)
    surrogate_1 = ratio * ppo_inputs.advantages_bt
    surrogate_2 = clipped_ratio * ppo_inputs.advantages_bt
    policy_loss = -jnp.minimum(surrogate_1, surrogate_2)

    # Computes the value loss, with or without clipping.
    if use_clipped_value_loss:
        value_objective = 0.5 * clipped_value_loss(
            target_values=on_policy_variables.values,
            values=off_policy_variables.values,
            value_targets=ppo_inputs.value_targets_bt,
            clip_param=clip_param,
        )
    else:
        value_objective = 0.5 * (ppo_inputs.value_targets_bt - off_policy_variables.values) ** 2
    value_loss = value_objective * value_loss_coef

    # Minimize the KL divergence between the two policies, to discourage large changes.
    kl_div = (on_policy_variables.log_probs - off_policy_variables.log_probs).sum(axis=-1)
    kl_loss = kl_div * kl_coef

    losses = {
        "policy": policy_loss,
        "value": value_loss,
        "kl": kl_loss,
    }

    # Maximize the entropy of the policy, to encourage exploration.
    if off_policy_variables.entropy is not None:
        entropy_loss = -off_policy_variables.entropy.sum(axis=-1) * entropy_coef
        losses["entropy"] = entropy_loss

    # Adds any additional auxiliary losses.
    if off_policy_variables.aux_losses is not None:
        for name, aux_loss_term in off_policy_variables.aux_losses.items():
            losses[name] = aux_loss_term

    return losses


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
    kl_coef: float = xax.field(
        value=1e-3,
        help="KL divergence coefficient for PPO, to discourage large changes in the policy.",
    )
    adaptive_kl: bool = xax.field(
        value=False,
        help="If true, adapt kl_coef online to target desired_kl.",
    )
    desired_kl: float = xax.field(
        value=1e-2,
        help="Target KL divergence between old and new policy when adaptive_kl is enabled.",
    )
    log_clip_value: float = xax.field(
        value=5.0,
        help="The log clip value for PPO, for numerical stability.",
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
        value=True,
        help="Whether to normalize the advantages.",
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
        trajectory: Trajectory,
        model_carry: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[PPOVariables, PyTree]:
        """Gets the variables required for computing PPO loss.

        Args:
            model: The user-provided model.
            trajectory: The trajectory to get PPO variables for.
            model_carry: The model carry from the previous rollout.
            rng: A random seed.

        Returns:
            The PPO variables and the next carry for the model.
        """

    def get_ppo_metrics(
        self,
        losses_bt: dict[str, Array],
        ppo_inputs: PPOInputs,
        on_policy_variables: PPOVariables,
        off_policy_variables: PPOVariables,
    ) -> dict[str, Array]:
        """Gets the metrics to be logged.

        If the metric is a scalar, it will be logged as a scalar. If the
        metric is a tuple, it is assumed to be a distribution in (mean, std)
        format and will be logged as a distribution.

        Args:
            losses_bt: The dictionary of losses.
            ppo_inputs: The PPO inputs.
            on_policy_variables: The variables for the original policy.
            off_policy_variables: The variables for the new policy.

        Returns:
            A dictionary of metrics to be logged.
        """
        metrics = {
            "on_policy_log_probs": on_policy_variables.log_probs.mean(0).flatten(),
            "off_policy_log_probs": off_policy_variables.log_probs.mean(0).flatten(),
            "on_policy_values": on_policy_variables.values.mean(0).flatten(),
            "off_policy_values": off_policy_variables.values.mean(0).flatten(),
            "value_targets": ppo_inputs.value_targets_bt.mean(),
            "advantages": ppo_inputs.advantages_bt.mean(),
        }
        # KL metric between on- and off-policy distributions
        kl_div = (on_policy_variables.log_probs - off_policy_variables.log_probs).sum(axis=-1).mean()
        metrics["kl"] = kl_div
        for name, loss in losses_bt.items():
            metrics[f"loss_{name}"] = loss.mean()
        if off_policy_variables.entropy is not None:
            metrics["entropy"] = off_policy_variables.entropy.mean(0).flatten()
        return metrics

    def _get_logged_trajectory_metrics(
        self,
        losses_t: dict[str, Array],
        ppo_inputs: PPOInputs,
        on_policy_variables: PPOVariables,
        off_policy_variables: PPOVariables,
    ) -> dict[str, Array]:
        """Gets the metrics to log for a single trajectory.

        If the metric is a scalar, it will be logged as a scalar. If the
        metric is a tuple, it is assumed to be a distribution in (mean, std)
        format and will be logged as a distribution.

        Args:
            losses_t: The dictionary of losses.
            ppo_inputs: The PPO inputs.
            on_policy_variables: The variables for the original policy.
            off_policy_variables: The variables for the new policy.

        Returns:
            A dictionary of metrics to be logged. Each metric should be a tensor
            with shape (T, *).
        """
        metrics = {
            "values": off_policy_variables.values,
            "value_targets": ppo_inputs.value_targets_bt,
            "advantages": ppo_inputs.advantages_bt,
            "gae": ppo_inputs.gae_bt,
            "returns": ppo_inputs.returns_bt,
        }
        for name, loss in losses_t.items():
            metrics[f"loss_{name}"] = loss
        if off_policy_variables.entropy is not None:
            metrics["entropy"] = off_policy_variables.entropy
        return metrics

    @xax.jit(
        static_argnames=["self", "model_static"],
        donate_argnames=["trajectories", "rewards", "on_policy_variables", "rng"],
        jit_level=JitLevel.RL_CORE,
    )
    def _get_ppo_loss_and_metrics(
        self,
        model_arr: PyTree,
        model_static: PyTree,
        trajectories: Trajectory,
        rewards: RewardState,
        init_carry: PyTree,
        on_policy_variables: PPOVariables,
        rng: PRNGKeyArray,
        kl_scale: float,
    ) -> tuple[Array, xax.FrozenDict[str, Array]]:
        """Computes the PPO loss and additional metrics.

        Args:
            model_arr: The mutable part of the model to optimize.
            model_static: The static part of the model to optimize.
            trajectories: The batch of trajectories to compute the loss and metrics for.
            rewards: The rewards for the trajectories.
            init_carry: The initial carry for the model.
            on_policy_variables: The PPO variables from the on-policy rollout.
            rng: A random seed.
            kl_scale: A runtime scaling factor applied to the KL loss term to support adaptive KL.

        Returns:
            A tuple containing the loss value as a scalar, a dictionary of
            metrics to log.
        """
        model = eqx.combine(model_arr, model_static)

        ppo_fn = xax.vmap(self.get_ppo_variables, in_axes=(None, 0, 0, 0), jit_level=JitLevel.RL_CORE)
        rngs = jax.random.split(rng, trajectories.done.shape[0])
        off_policy_variables, _ = ppo_fn(model, trajectories, init_carry, rngs)

        ppo_inputs = compute_ppo_inputs(
            values_bt=off_policy_variables.values,
            rewards_bt=rewards.total,
            dones_bt=trajectories.done,
            successes_bt=trajectories.success,
            decay_gamma=self.config.gamma,
            gae_lambda=self.config.lam,
            normalize_advantages=self.config.normalize_advantages,
            monte_carlo_returns=self.config.monte_carlo_returns,
        )

        losses_bt = compute_ppo_loss(
            ppo_inputs=ppo_inputs,
            on_policy_variables=on_policy_variables,
            off_policy_variables=off_policy_variables,
            clip_param=self.config.clip_param,
            value_loss_coef=self.config.value_loss_coef,
            entropy_coef=self.config.entropy_coef,
            kl_coef=self.config.kl_coef,
            log_clip_value=self.config.log_clip_value,
            use_clipped_value_loss=self.config.use_clipped_value_loss,
        )

        # Rescale KL loss dynamically to support adaptive KL without recompilation.
        losses_bt = dict(losses_bt)
        losses_bt["kl"] = losses_bt["kl"] * kl_scale

        metrics = self.get_ppo_metrics(
            losses_bt=losses_bt,
            ppo_inputs=ppo_inputs,
            on_policy_variables=on_policy_variables,
            off_policy_variables=off_policy_variables,
        )

        # Mean loss over all losses.
        loss_bt = jnp.stack(list(losses_bt.values()), axis=-1).sum(axis=-1)
        loss = loss_bt.mean()

        return loss, xax.freeze_dict(metrics)

    @xax.jit(
        static_argnames=["self", "constants"],
        donate_argnames=["trajectories", "rewards", "carry", "on_policy_variables", "rng"],
        jit_level=JitLevel.RL_CORE,
    )
    def _single_step(
        self,
        trajectories: Trajectory,
        rewards: RewardState,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        on_policy_variables: PPOVariables,
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopCarry, xax.FrozenDict[str, Array]]:
        # Gets the policy model and optimizer.
        model_arr = carry.shared_state.model_arrs[0]
        model_static = constants.constants.model_statics[0]
        optimizer = constants.optimizer[0]
        opt_state = carry.opt_state[0]

        # Computes the metrics and PPO gradients.
        loss_fn = xax.grad(
            self._get_ppo_loss_and_metrics,
            argnums=0,
            has_aux=True,
            jit_level=JitLevel.RL_CORE,
        )
        # Compute runtime KL scaling factor for adaptive KL.
        current_kl_coef = carry.shared_state.aux_values["kl_coef"] if self.config.adaptive_kl else self.config.kl_coef
        kl_scale = current_kl_coef / self.config.kl_coef

        grads, ppo_metrics = loss_fn(
            model_arr,
            model_static,
            trajectories,
            rewards,
            carry.env_states.model_carry,
            on_policy_variables,
            rng,
            kl_scale,
        )

        # Applies the gradients.
        updates, new_opt_state = optimizer.update(grads, opt_state, model_arr)
        new_model_arr = eqx.apply_updates(model_arr, updates)

        # Updates the carry with the new model and optimizer states.
        carry = replace(
            carry,
            shared_state=replace(
                carry.shared_state,
                model_arrs=xax.tuple_insert(carry.shared_state.model_arrs, 0, new_model_arr),
            ),
            opt_state=xax.tuple_insert(carry.opt_state, 0, new_opt_state),
        )

        # If enabled, adapt KL coefficient toward the desired KL.
        if self.config.adaptive_kl:
            kl_mean = ppo_metrics["kl"].mean()
            desired = self.config.desired_kl
            factor = 1.5
            new_kl_coef = jnp.where(kl_mean > desired * factor, current_kl_coef * factor, current_kl_coef)
            new_kl_coef = jnp.where(kl_mean < desired / factor, new_kl_coef / factor, new_kl_coef)
            new_kl_coef = jnp.clip(new_kl_coef, 1e-8, 1e2)

            aux_vals = {**dict(carry.shared_state.aux_values)}
            aux_vals["kl_coef"] = new_kl_coef
            carry = replace(
                carry,
                shared_state=replace(
                    carry.shared_state,
                    aux_values=xax.freeze_dict(aux_vals),
                ),
            )

        return carry, ppo_metrics

    @xax.jit(
        static_argnames=["self", "constants"],
        donate_argnames=["carry", "rng"],
        jit_level=JitLevel.RL_CORE,
    )
    def update_model(
        self,
        *,
        constants: RLLoopConstants,
        carry: RLLoopCarry,
        trajectories: Trajectory,
        rewards: RewardState,
        rng: PRNGKeyArray,
    ) -> tuple[RLLoopCarry, xax.FrozenDict[str, Array]]:
        rng, onp_rng, passes_rng = jax.random.split(rng, 3)

        # Initialize adaptive KL coefficient in aux_values if needed.
        if self.config.adaptive_kl and "kl_coef" not in carry.shared_state.aux_values:
            init_aux = {**dict(carry.shared_state.aux_values)}
            init_aux["kl_coef"] = jnp.asarray(self.config.kl_coef)
            carry = replace(
                carry,
                shared_state=replace(
                    carry.shared_state,
                    aux_values=xax.freeze_dict(init_aux),
                ),
            )

        # Gets the policy model.
        policy_model_arr = carry.shared_state.model_arrs[0]
        policy_model_static = constants.constants.model_statics[0]
        policy_model = eqx.combine(policy_model_arr, policy_model_static)

        # Runs the policy model on the trajectory to get the PPO variables.
        on_policy_rngs = jax.random.split(onp_rng, trajectories.done.shape[0])
        ppo_fn = xax.vmap(self.get_ppo_variables, in_axes=(None, 0, 0, 0), jit_level=JitLevel.RL_CORE)
        on_policy_variables, _ = ppo_fn(policy_model, trajectories, carry.env_states.model_carry, on_policy_rngs)
        on_policy_variables = jax.lax.stop_gradient(on_policy_variables)

        # Loops over the trajectory batches and applies gradient updates.
        def update_model_in_batch(
            carry: RLLoopCarry,
            xs: tuple[Array, PRNGKeyArray],
        ) -> tuple[RLLoopCarry, xax.FrozenDict[str, Array]]:
            batch_indices, rng = xs

            # Gets the current batch of trajectories and rewards.
            trajectory_batch = jax.tree.map(lambda x: x[batch_indices], trajectories)
            reward_batch = jax.tree.map(lambda x: x[batch_indices], rewards)
            env_states_batch = jax.tree.map(lambda x: x[batch_indices], carry.env_states)
            on_policy_variables_batch = jax.tree.map(lambda x: x[batch_indices], on_policy_variables)

            next_carry, metrics = self._single_step(
                trajectories=trajectory_batch,
                rewards=reward_batch,
                constants=constants,
                carry=replace(carry, env_states=env_states_batch),
                on_policy_variables=on_policy_variables_batch,
                rng=rng,
            )

            # Update the carry's shared states.
            carry = replace(
                carry,
                opt_state=next_carry.opt_state,
                shared_state=next_carry.shared_state,
            )

            return carry, metrics

        # Applies N steps of gradient updates.
        def update_model_across_batches(
            carry: RLLoopCarry,
            rng: PRNGKeyArray,
        ) -> tuple[RLLoopCarry, xax.FrozenDict[str, Array]]:
            shuffle_rng, batch_rng = jax.random.split(rng)

            # Shuffle the indices so that minibatch updates are different.
            indices = jnp.arange(trajectories.done.shape[0])
            indices = jax.random.permutation(shuffle_rng, indices, independent=False)
            indices_by_batch = indices.reshape(-1, min(self.batch_size, indices.shape[0]))

            carry, metrics = xax.scan(
                update_model_in_batch,
                carry,
                (indices_by_batch, jax.random.split(batch_rng, indices_by_batch.shape[0])),
                jit_level=JitLevel.RL_CORE,
            )

            return carry, metrics

        # Applies gradient update across all batches num_passes times.
        carry, metrics = xax.scan(
            update_model_across_batches,
            carry,
            xs=jax.random.split(passes_rng, self.config.num_passes),
            jit_level=JitLevel.HELPER_FUNCTIONS,
        )

        if carry.env_states.model_carry is not None:
            # Gets the policy model, using the latest model parameters.
            policy_model_arr = carry.shared_state.model_arrs[0]
            policy_model_static = constants.constants.model_statics[0]
            policy_model = eqx.combine(policy_model_arr, policy_model_static)

            # For the next rollout, we use the model carry from the output of the
            # model update instead of the output of the rollout. This was shown to
            # work slightly better in practice - for an  RNN model, for example,
            # after updating the model, the model carry will be new and the
            # previous rollout's model carry will be incorrect. This does perform
            # some additional computation, but the impact is small.
            off_policy_rngs = jax.random.split(rng, trajectories.done.shape[0])
            _, next_model_carrys = ppo_fn(
                policy_model,
                trajectories,
                carry.env_states.model_carry,
                off_policy_rngs,
            )

            carry = replace(
                carry,
                env_states=replace(
                    carry.env_states,
                    model_carry=next_model_carrys,
                ),
            )

        return carry, metrics
