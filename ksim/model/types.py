"""Typing for the model."""

from __future__ import annotations

from dataclasses import dataclass

import jax.tree_util
from flax.core import FrozenDict
from jaxtyping import Array


@jax.tree_util.register_dataclass
@dataclass
class ModelInput:
    """A consistent input structure for all models."""

    obs: FrozenDict[str, Array]
    command: FrozenDict[str, Array]
    action_history: Array | None
    recurrent_state: Array | FrozenDict[str, Array] | None
