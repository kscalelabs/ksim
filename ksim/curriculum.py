"""Defines a base curriculum class. Curricula are updated after every epoch."""

__all__ = [
    "Curriculum",
    "LinearCurriculum",
    "ConstantCurriculum",
    "EpisodeLengthCurriculum",
    "DistanceFromOriginCurriculum",
    "StepWhenSaturated",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.types import Rewards, Trajectory

T = TypeVar("T")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CurriculumState(Generic[T]):
    level: Array
    state: T


@attrs.define(frozen=True, kw_only=True)
class Curriculum(ABC, Generic[T]):
    """Base class for all curricula.

    Curricula should return a level between 0 and 1.
    """

    @abstractmethod
    def __call__(
        self,
        trajectory: Trajectory,
        rewards: Rewards,
        training_state: xax.State,
        prev_state: CurriculumState[T],
    ) -> CurriculumState[T]: ...

    @abstractmethod
    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[T]: ...


@attrs.define(frozen=True, kw_only=True)
class ConstantCurriculum(Curriculum[None]):
    """Constant curriculum."""

    level: float = attrs.field()

    def __call__(
        self,
        trajectory: Trajectory,
        rewards: Rewards,
        training_state: xax.State,
        prev_state: CurriculumState[None],
    ) -> CurriculumState[None]:
        return prev_state

    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[None]:
        return CurriculumState(level=jnp.array(self.level), state=None)


@attrs.define(frozen=True, kw_only=True)
class LinearCurriculum(Curriculum[None]):
    """Linear curriculum."""

    step_size: float = attrs.field(default=0.01)
    step_every_n_epochs: int = attrs.field(default=1)

    def __call__(
        self,
        trajectory: Trajectory,
        rewards: Rewards,
        training_state: xax.State,
        prev_state: CurriculumState[None],
    ) -> CurriculumState[None]:
        level = (training_state.num_steps // self.step_every_n_epochs) * self.step_size
        level = jnp.clip(level, 0.0, 1.0)
        return CurriculumState(level=level, state=None)

    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[None]:
        return CurriculumState(level=jnp.array(0.0), state=None)


@attrs.define(frozen=True, kw_only=True)
class EpisodeLengthCurriculum(Curriculum[None]):
    """Curriculum that updates the episode length."""

    min_length_steps: int = attrs.field()
    max_length_steps: int = attrs.field()

    def __call__(
        self,
        trajectory: Trajectory,
        rewards: Rewards,
        training_state: xax.State,
        prev_state: CurriculumState[None],
    ) -> CurriculumState[None]:
        tsz = trajectory.done.shape[-1]
        num_episodes = trajectory.done.sum(axis=-1).mean() + 1
        episode_length = tsz / num_episodes
        level = (episode_length - self.min_length_steps) / (self.max_length_steps - self.min_length_steps)
        return CurriculumState(level=jnp.clip(level, 0.0, 1.0), state=None)

    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[None]:
        return CurriculumState(level=jnp.array(0.0), state=None)

    @classmethod
    def create(
        cls,
        min_length_seconds: float,
        max_length_seconds: float,
        ctrl_dt: float,
    ) -> Self:
        min_length_steps = round(min_length_seconds / ctrl_dt)
        max_length_steps = round(max_length_seconds / ctrl_dt)
        return cls(min_length_steps=min_length_steps, max_length_steps=max_length_steps)


@attrs.define(frozen=True, kw_only=True)
class DistanceFromOriginCurriculum(Curriculum[None]):
    """Curriculum that updates the distance from the origin."""

    min_distance: float = attrs.field()
    max_distance: float = attrs.field()

    def __call__(
        self,
        trajectory: Trajectory,
        rewards: Rewards,
        training_state: xax.State,
        prev_state: CurriculumState[None],
    ) -> CurriculumState[None]:
        distance = jnp.linalg.norm(trajectory.qpos[..., :3], axis=-1).max()
        level = (distance - self.min_distance) / (self.max_distance - self.min_distance)
        return CurriculumState(level=jnp.clip(level, 0.0, 1.0), state=None)

    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[None]:
        return CurriculumState(level=jnp.array(0.0), state=None)


@attrs.define(frozen=True, kw_only=True)
class StepWhenSaturated(Curriculum[Array]):
    """Step the curriculum depending on how many deaths there are per episode.

    The logic here is that if there are fewer than `increase_threshold` deaths
    per episode on average, then we should increase the curriculum level.
    Similarly, if there are more than `decrease_threshold` deaths per episode
    on average, then we should decrease the curriculum level.

    We only allow the level to be changed every `min_level_steps` steps.
    """

    num_levels: int = attrs.field()
    increase_threshold: float = attrs.field()
    decrease_threshold: float = attrs.field()
    min_level_steps: int = attrs.field()

    def __call__(
        self,
        trajectory: Trajectory,
        rewards: Rewards,
        training_state: xax.State,
        prev_state: CurriculumState[Array],
    ) -> CurriculumState[Array]:
        level, steps = prev_state.level, prev_state.state
        level, steps = jax.lax.cond(
            steps <= 0,
            lambda: self._step_level(level, trajectory),
            lambda: (level, steps - 1),
        )
        return CurriculumState(level=level, state=steps)

    def _step_level(self, level: Array, trajectory: Trajectory) -> tuple[Array, Array]:
        deaths = trajectory.done.sum(axis=-1).mean()
        should_increase = deaths < self.increase_threshold
        should_decrease = deaths > self.decrease_threshold
        delta = 1.0 / self.num_levels
        level = jnp.where(should_increase, level + delta, jnp.where(should_decrease, level - delta, level))
        return jnp.clip(level, 0.0, 1.0), jnp.array(self.min_level_steps, dtype=jnp.int32)

    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[Array]:
        return CurriculumState(level=jnp.array(0.0), state=jnp.array(self.min_level_steps, dtype=jnp.int32))
