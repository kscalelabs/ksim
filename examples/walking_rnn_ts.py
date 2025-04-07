# mypy: disable-error-code="override"
"""Performs teacher-student fine-tuning of a walking policy using an RNN actor."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

import ksim

from .walking_rnn import (
    NUM_INPUTS,
    NUM_JOINTS,
    HumanoidWalkingRNNTask,
    HumanoidWalkingRNNTaskConfig,
    DefaultHumanoidRNNActor,
)

