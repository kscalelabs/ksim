"""Types used at the task level."""

from dataclasses import dataclass

import jax
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.model.formulations import ActorCriticAgent


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RolloutTimeLossComponents:
    """Components of the loss function for a rollout."""

    initial_action_log_probs: Array
    initial_values: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PPORolloutTimeLossComponents(RolloutTimeLossComponents):
    """Components of the loss function for a PPO rollout."""

    advantages: Array
    returns: Array
