"""Types used at the task level."""

from dataclasses import dataclass

import jax
from jaxtyping import Array


@jax.tree_util.register_dataclass
@dataclass
class RolloutTimeLossComponents:
    """Components of the loss function for a rollout."""

    initial_action_log_probs: Array
    initial_values: Array


@jax.tree_util.register_dataclass
@dataclass
class PPORolloutTimeLossComponents(RolloutTimeLossComponents):
    """Components of the loss function for a PPO rollout."""

    advantages: Array
    returns: Array
