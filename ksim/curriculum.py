"""Defines a base curriculum class. Curricula are updated after every epoch."""

__all__ = [
    "Curriculum",
    "LinearCurriculum",
    "ConstantCurriculum",
    "EpisodeLengthCurriculum",
    "DistanceFromOriginCurriculum",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

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

    level: float = attrs.field(validator=attrs.validators.ge(0.0))

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

    step_size: float = attrs.field(default=0.01, validator=attrs.validators.ge(0.0))
    step_every_n_epochs: int = attrs.field(default=1, validator=attrs.validators.ge(1))

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
class EpisodeLengthCurriculum(Curriculum[Array]):
    """Curriculum that updates the episode length."""

    num_levels: int = attrs.field(validator=attrs.validators.ge(1))
    increase_threshold: float = attrs.field(validator=attrs.validators.ge(0.0))
    decrease_threshold: float = attrs.field(validator=attrs.validators.ge(0.0))
    min_level_steps: int = attrs.field(validator=attrs.validators.ge(0))

    def __call__(
        self,
        trajectory: Trajectory,
        rewards: Rewards,
        training_state: xax.State,
        prev_state: CurriculumState[Array],
    ) -> CurriculumState[Array]:
        step_size = 1 / self.num_levels
        episode_length = trajectory.episode_length().mean()
        steps = prev_state.state
        level = prev_state.level
        next_steps = (steps - 1).clip(min=0)
        can_step = next_steps == 0
        should_inc = (episode_length > self.increase_threshold) & can_step
        should_dec = (episode_length < self.decrease_threshold) & can_step
        next_level = jnp.where(should_inc, level + step_size, jnp.where(should_dec, level - step_size, level))
        next_level = jnp.clip(next_level, 0.0, 1.0)
        next_steps = jnp.where(should_inc | should_dec, self.min_level_steps, next_steps)
        return CurriculumState(level=next_level, state=next_steps)

    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[Array]:
        return CurriculumState(level=jnp.array(0.0), state=jnp.array(self.min_level_steps, dtype=jnp.int32))


@attrs.define(frozen=True, kw_only=True)
class DistanceFromOriginCurriculum(Curriculum[None]):
    """Curriculum that updates the distance from the origin."""

    min_distance: float = attrs.field(validator=attrs.validators.ge(0.0))
    max_distance: float = attrs.field(validator=attrs.validators.ge(0.0))

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
