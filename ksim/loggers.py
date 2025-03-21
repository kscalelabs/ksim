"""Defines a standard interface for logging metrics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import attrs
import jax
import jax.numpy as jnp
import xax
from flax.core import FrozenDict

from ksim.env.data import Trajectory


@jax.tree_util.register_dataclass
@dataclass
class LoggingData:
    update_metrics: FrozenDict[str, jnp.ndarray]
    trajectory: Trajectory | None = None


@attrs.define(frozen=True, kw_only=True)
class LogMetric(ABC):
    @abstractmethod
    def __call__(self, logger: xax.Logger, data: LoggingData) -> None:
        """Compute the log metric from trajectory data.

        The data argument should include any fields necessary for computing
        the metric.
        """

    def get_name(self) -> str:
        """Returns a human-friendly name for the metric."""
        return xax.camelcase_to_snakecase(self.__class__.__name__)


@attrs.define(frozen=True, kw_only=True)
class EpisodeLengthLog(LogMetric):
    def __call__(self, logger: xax.Logger, data: LoggingData) -> None:
        if data.trajectory is None:
            raise ValueError("Trajectory cannot be None for EpisodeLengthLog")
        trajectory = data.trajectory
        # Ensure we have at least one termination event.
        episode_count = jnp.sum(trajectory.done).clip(min=1)
        logger.log_scalar(
            "average_episode_length",
            jnp.sum(~trajectory.done) / episode_count,
            namespace="📉",
        )


@attrs.define(frozen=True)
class ModelUpdateLog(LogMetric):
    def __call__(self, logger: xax.Logger, data: LoggingData) -> None:
        for key, value in data.update_metrics.items():
            assert isinstance(value, jnp.ndarray)
            logger.log_scalar(
                key,
                value,
                namespace="📉",
            )
