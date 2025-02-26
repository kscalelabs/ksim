"""Includes all major typing."""

from typing import NamedTuple

from jaxtyping import Array, PyTree

ModelObs = Array | PyTree[Array]
ModelOut = Array | PyTree[Array]


class PPOBatch(NamedTuple):
    """Batch data structure for PPO training and logging."""

    observations: PyTree
    next_observations: PyTree
    actions: Array
    rewards: Array
    done: Array
    action_log_probs: Array
