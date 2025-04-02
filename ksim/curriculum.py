"""Defines a base curriculum class. Curricula are updated after every epoch."""

__all__ = [
    "Curriculum",
    "LinearCurriculum",
    "EpisodeLengthCurriculum",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.types import Trajectory

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
        training_state: xax.State,
        prev_state: CurriculumState[None],
    ) -> CurriculumState[None]:
        return prev_state

    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[None]:
        return CurriculumState(level=jnp.array(self.level), state=None)


@attrs.define(frozen=True, kw_only=True)
class LinearCurriculum(Curriculum[None]):
    """Linear curriculum."""

    min_level: float = attrs.field(default=0.0)
    max_level: float = attrs.field(default=1.0)
    step_size: float = attrs.field(default=0.01)
    step_every_n_epochs: int = attrs.field(default=1)

    def __call__(
        self,
        trajectory: Trajectory,
        training_state: xax.State,
        prev_state: CurriculumState[None],
    ) -> CurriculumState[None]:
        level = (training_state.num_steps // self.step_every_n_epochs) * self.step_size
        level = jnp.clip(level, self.min_level, self.max_level)
        return CurriculumState(level=level, state=None)

    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[None]:
        return CurriculumState(level=jnp.array(self.min_level), state=None)


@attrs.define(frozen=True, kw_only=True)
class EpisodeLengthCurriculum(Curriculum[None]):
    """Curriculum that updates the episode length."""

    min_length_steps: int = attrs.field()
    max_length_steps: int = attrs.field()

    def __call__(
        self,
        trajectory: Trajectory,
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
