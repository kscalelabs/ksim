"""Defines a standard interface for logging metrics."""

from abc import ABC, abstractmethod
from typing import Generic, NamedTuple, Optional, TypeVar

import attrs
import jax.numpy as jnp
import xax
from jaxtyping import Array, PyTree

from ksim.utils.data import BuilderData


class TrajectoryData(NamedTuple):
    """Simple trajectory data structure for logging.

    TODO: This class should really be PPOBatch, but that leads to circular imports.
    So we should have a base "Batch" type in a seperate file that both PPO and
    logging can import from.
    """

    observations: PyTree
    next_observations: PyTree
    actions: Array
    rewards: Array
    done: Array
    action_log_probs: Array


@attrs.define(frozen=True, kw_only=True)
class LoggingData:
    trajectory: Optional[TrajectoryData] = None
    update_metrics: dict[str, jnp.ndarray] = attrs.field(
        factory=dict
    )  # e.g., policy_loss, value_loss, entropy, total_loss
    gradients: Optional[PyTree] = None
    loss: Optional[float] = None
    training_state: Optional[xax.State] = None


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


T = TypeVar("T", bound=LogMetric)


class LogMetricBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, data: Optional[BuilderData] = None) -> T:
        """Constructs and returns a log metric instance given some builder data."""


@attrs.define(frozen=True, kw_only=True)
class EpisodeLengthLog(LogMetric):
    def __call__(self, data: LoggingData) -> jnp.ndarray:
        """Computes the average episode length."""
        if data.trajectory is None:
            raise ValueError("Trajectory cannot be None for EpisodeLengthLog")
        trajectory = data.trajectory
        return jnp.sum(~trajectory.done) / (jnp.sum(trajectory.done) + 1)


class EpisodeLengthLogBuilder(LogMetricBuilder[EpisodeLengthLog]):
    def __init__(self) -> None:
        pass

    def __call__(self, data: Optional[BuilderData] = None) -> EpisodeLengthLog:
        return EpisodeLengthLog()


class AverageRewardLog(LogMetric):
    def __call__(self, data: LoggingData) -> jnp.ndarray:
        """Computes the average reward per episode."""
        if data.trajectory is None:
            raise ValueError("Trajectory cannot be None for AverageRewardLog")
        trajectory = data.trajectory
        total_reward = jnp.sum(trajectory.rewards)
        episode_count = jnp.sum(trajectory.done) + 1  # +1 to avoid division by zero
        return total_reward / episode_count


class AverageRewardLogBuilder(LogMetricBuilder[AverageRewardLog]):
    def __call__(self, data: Optional[BuilderData] = None) -> AverageRewardLog:
        return AverageRewardLog()


class PolicyLossLog(LogMetric):
    def __call__(self, data: LoggingData) -> jnp.ndarray:
        return data.update_metrics["policy_loss"]


class PolicyLossLogBuilder(LogMetricBuilder[PolicyLossLog]):
    def __call__(self, data: Optional[BuilderData] = None) -> PolicyLossLog:
        return PolicyLossLog()


class ValueLossLog(LogMetric):
    def __call__(self, data: LoggingData) -> jnp.ndarray:
        return data.update_metrics["value_loss"]


class ValueLossLogBuilder(LogMetricBuilder[ValueLossLog]):
    def __call__(self, data: Optional[BuilderData] = None) -> ValueLossLog:
        return ValueLossLog()


class EntropyLog(LogMetric):
    def __call__(self, data: LoggingData) -> jnp.ndarray:
        return data.update_metrics["entropy"]


class EntropyLogBuilder(LogMetricBuilder[EntropyLog]):
    def __call__(self, data: Optional[BuilderData] = None) -> EntropyLog:
        return EntropyLog()


class TotalLossLog(LogMetric):
    def __call__(self, data: LoggingData) -> jnp.ndarray:
        return data.update_metrics["total_loss"]


class TotalLossLogBuilder(LogMetricBuilder[TotalLossLog]):
    def __call__(self, data: Optional[BuilderData] = None) -> TotalLossLog:
        return TotalLossLog()
