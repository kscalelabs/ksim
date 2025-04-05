"""Defines a stateful version of PPO.

This interface can be used for training RNN policies.
"""

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

from ksim.task.rl import RLConfig, RLTask, RolloutConstants, RolloutVariables
from ksim.types import Rewards, SingleTrajectory, Trajectory
from ksim.task.ppo import PPOConfig, PPOVariables
