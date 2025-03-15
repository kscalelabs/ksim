"""Types used at the task level."""

from dataclasses import dataclass

import jax
from jaxtyping import Array

from ksim.env.data import Transition


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RolloutTimeStats:
    """Only computed once at the end of a rollout, used accross training."""

    initial_action_log_probs: Array
    initial_values: Array
    returns: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PPORolloutTimeStats(RolloutTimeStats):
    """Components of the loss function for a PPO rollout."""

    advantages: Array
    value_targets: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RLDataset:
    """Everything you need to run an RL pass."""

    transitions: Transition
    rollout_time_stats: RolloutTimeStats
