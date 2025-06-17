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
from typing import Generic, TypeVar

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.types import RewardState, Trajectory

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
        rewards: RewardState,
        training_state: xax.State,
        prev_state: CurriculumState[T],
    ) -> CurriculumState[T]: ...

    @abstractmethod
    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[T]: ...


@attrs.define(frozen=True, kw_only=True)
class ConstantCurriculum(Curriculum[None]):
    """Constant curriculum."""

    level: float = attrs.field(default=1.0, validator=attrs.validators.ge(0.0))

    def __call__(
        self,
        trajectory: Trajectory,
        rewards: RewardState,
        training_state: xax.State,
        prev_state: CurriculumState[None],
    ) -> CurriculumState[None]:
        return prev_state

    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[None]:
        return CurriculumState(level=jnp.array(self.level, dtype=jnp.float32), state=None)


@attrs.define(frozen=True, kw_only=True)
class LinearCurriculum(Curriculum[None]):
    """Linear curriculum."""

    step_size: float = attrs.field(default=0.01, validator=attrs.validators.ge(0.0))
    step_every_n_epochs: int = attrs.field(default=1, validator=attrs.validators.ge(1))
    min_level: float = attrs.field(default=0.0, validator=attrs.validators.ge(0.0))

    def __call__(
        self,
        trajectory: Trajectory,
        rewards: RewardState,
        training_state: xax.State,
        prev_state: CurriculumState[None],
    ) -> CurriculumState[None]:
        level = (training_state.num_steps // self.step_every_n_epochs) * self.step_size
        level = jnp.clip(level, self.min_level, 1.0)
        level = jnp.full_like(prev_state.level, level)
        return CurriculumState(level=level, state=None)

    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[None]:
        return CurriculumState(level=jnp.array(self.min_level, dtype=jnp.float32), state=None)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EpisodeLengthCurriculumState:
    step_counter: Array
    ema_episode_length: Array


@attrs.define(frozen=True, kw_only=True)
class EpisodeLengthCurriculum(Curriculum[EpisodeLengthCurriculumState]):
    """Curriculum that updates the episode length."""

    num_levels: int = attrs.field(default=100, validator=attrs.validators.ge(1))
    increase_threshold: float = attrs.field(default=3.0, validator=attrs.validators.ge(0.0))
    decrease_threshold: float = attrs.field(default=3.0, validator=attrs.validators.ge(0.0))
    min_level_steps: int = attrs.field(default=1, validator=attrs.validators.ge(0))
    min_level: float = attrs.field(default=0.0, validator=attrs.validators.ge(0.0))
    ema_decay: float = attrs.field(default=0.9, validator=attrs.validators.ge(0.0))

    def __call__(
        self,
        trajectory: Trajectory,
        rewards: RewardState,
        training_state: xax.State,
        prev_state: CurriculumState[EpisodeLengthCurriculumState],
    ) -> CurriculumState[EpisodeLengthCurriculumState]:
        step_size = 1 / self.num_levels

        episode_length = trajectory.episode_length().mean()
        ema = self.ema_decay * prev_state.state.ema_episode_length + (1 - self.ema_decay) * episode_length

        steps = prev_state.state.step_counter
        level = prev_state.level
        can_step = steps == 0
        should_inc = (ema > self.increase_threshold) & can_step
        should_dec = (ema < self.decrease_threshold) & can_step
        next_steps = jnp.where(should_inc | should_dec, self.min_level_steps, jnp.maximum(steps - 1, 0))

        next_level = jnp.where(
            should_inc,
            level + step_size,
            jnp.where(should_dec, level - step_size, level),
        )
        next_level = jnp.clip(next_level, self.min_level, 1.0)
        next_steps = jnp.where(should_inc | should_dec, self.min_level_steps, next_steps)

        return CurriculumState(
            level=next_level,
            state=EpisodeLengthCurriculumState(
                step_counter=next_steps,
                ema_episode_length=ema,
            ),
        )

    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[EpisodeLengthCurriculumState]:
        return CurriculumState(
            level=jnp.array(self.min_level, dtype=jnp.float32),
            state=EpisodeLengthCurriculumState(
                step_counter=jnp.array(self.min_level_steps, dtype=jnp.int32),
                ema_episode_length=jnp.array(0.0, dtype=jnp.float32),
            ),
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DistanceFromOriginCurriculumState:
    step_counter: Array
    ema_distance: Array


@attrs.define(frozen=True, kw_only=True)
class DistanceFromOriginCurriculum(Curriculum[DistanceFromOriginCurriculumState]):
    """Curriculum that updates the distance from the origin based on thresholds."""

    num_levels: int = attrs.field(default=100, validator=attrs.validators.ge(1))
    increase_threshold: float = attrs.field(default=3.0, validator=attrs.validators.ge(0.0))
    decrease_threshold: float = attrs.field(default=3.0, validator=attrs.validators.ge(0.0))
    min_level_steps: int = attrs.field(default=1, validator=attrs.validators.ge(1))
    min_level: float = attrs.field(default=0.0, validator=attrs.validators.ge(0.0))
    ema_decay: float = attrs.field(default=0.9, validator=attrs.validators.ge(0.0))

    def __call__(
        self,
        trajectory: Trajectory,
        rewards: RewardState,
        training_state: xax.State,
        prev_state: CurriculumState[DistanceFromOriginCurriculumState],
    ) -> CurriculumState[DistanceFromOriginCurriculumState]:
        # Calculate current distance from origin
        distance = jnp.linalg.norm(trajectory.qpos[..., :3], axis=-1).max()
        ema = self.ema_decay * prev_state.state.ema_distance + (1 - self.ema_decay) * distance

        # Get current level and steps
        current_level = prev_state.level
        steps = jnp.maximum(prev_state.state.step_counter - 1, 0)

        # Update level based on thresholds
        new_level_if_enough_steps = jnp.where(
            ema > self.increase_threshold,
            jnp.minimum(current_level + 1.0 / self.num_levels, 1.0),
            jnp.where(
                ema < self.decrease_threshold,
                jnp.maximum(current_level - 1.0 / self.num_levels, 0.0),
                current_level,
            ),
        )

        new_level = jnp.where(steps == 0, new_level_if_enough_steps, current_level)

        # Reset steps if level changed
        steps = jnp.where(new_level != current_level, self.min_level_steps, steps)

        return CurriculumState(
            level=new_level,
            state=DistanceFromOriginCurriculumState(
                step_counter=steps,
                ema_distance=ema,
            ),
        )

    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[DistanceFromOriginCurriculumState]:
        return CurriculumState(
            level=jnp.array(self.min_level, dtype=jnp.float32),
            state=DistanceFromOriginCurriculumState(
                step_counter=jnp.array(self.min_level_steps, dtype=jnp.int32),
                ema_distance=jnp.array(0.0, dtype=jnp.float32),
            ),
        )


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
    min_level: float = attrs.field(default=0.0, validator=attrs.validators.ge(0.0))

    def __call__(
        self,
        trajectory: Trajectory,
        rewards: RewardState,
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
        level = jnp.where(
            should_increase,
            level + delta,
            jnp.where(should_decrease, level - delta, level),
        )
        return jnp.clip(level, self.min_level, 1.0), jnp.array(self.min_level_steps, dtype=jnp.int32)

    def get_initial_state(self, rng: PRNGKeyArray) -> CurriculumState[Array]:
        return CurriculumState(
            level=jnp.array(self.min_level, dtype=jnp.float32),
            state=jnp.array(self.min_level_steps, dtype=jnp.int32),
        )
