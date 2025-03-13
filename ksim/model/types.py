"""Typing for the model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.tree_util
from flax.core import FrozenDict
from jaxtyping import Array


@jax.tree_util.register_dataclass
@dataclass
class ModelInput:
    """A consistent input structure for all models."""

    obs_proprio_vec: Array
    """Proprioceptive observations, shape: (..., proprio_history_length, proprio_dim)."""

    command_vec: Array
    """Vector-like commands, shape: (..., command_dim)."""

    obs_image_tensor: Array | None
    """Image observations, shape: (..., image_history_length, height, width, channels)."""

    command_text_tokens: Array | None
    """Text command tokens, shape: (..., sequence_length)."""

    action_history_vec: Array | None
    """Previous action, shape: (..., action_history_length, action_dim)."""

    recurrent_state: Array | FrozenDict[str, Array] | None
    """Recurrent state, shape: (..., history_length, recurrent_dim)."""


PredictionType = Literal["mean", "mean_std", "direct"]
DistributionType = Literal["gaussian", "tanh_gaussian", "categorical"]
