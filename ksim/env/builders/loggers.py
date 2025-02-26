"""Defines a standard interface for logging metrics."""

from abc import ABC, abstractmethod

import attrs
import jax.numpy as jnp
import xax
from jaxtyping import PyTree

from ksim.types import PPOBatch


@attrs.define(frozen=True, kw_only=True)
class LoggingData:
    trajectory: PPOBatch | None = None
    update_metrics: dict[str, jnp.ndarray] = attrs.field(factory=dict)
    gradients: PyTree | None = None
    loss: float | None = None
    training_state: xax.State | None = None


@attrs.define(frozen=True, kw_only=True)
class LogMetric(ABC):
    @abstractmethod
    def __call__(self, data: LoggingData) -> jnp.ndarray:
        """Compute the log metric from trajectory data.

        The data argument should include any fields necessary for computing
        the metric.
        """

    def get_name(self) -> str:
        """Returns a human-friendly name for the metric."""
        return xax.camelcase_to_snakecase(self.__class__.__name__)


@attrs.define(frozen=True, kw_only=True)
class EpisodeLengthLog(LogMetric):
    def __call__(self, data: LoggingData) -> jnp.ndarray:
        if data.trajectory is None:
            raise ValueError("Trajectory cannot be None for EpisodeLengthLog")
        trajectory = data.trajectory
        # Ensure we have at least one termination event.
        episode_count = jnp.sum(trajectory.done).clip(min=1)
        return jnp.sum(~trajectory.done) / episode_count


@attrs.define(frozen=True, kw_only=True)
class AverageRewardLog(LogMetric):
    def __call__(self, data: LoggingData) -> jnp.ndarray:
        if data.trajectory is None:
            raise ValueError("Trajectory cannot be None for AverageRewardLog")
        trajectory = data.trajectory
        total_reward = jnp.sum(trajectory.rewards)
        episode_count = jnp.sum(trajectory.done).clip(min=1)
        return total_reward / episode_count


@attrs.define(frozen=True)
class ModelUpdateLog(LogMetric):
    name: str = attrs.field()

    def __call__(self, data: LoggingData) -> jnp.ndarray:
        return data.update_metrics[self.name]

    def get_name(self) -> str:
        return self.name
