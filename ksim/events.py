"""Defines the base event classes and builders. Events are triggered during a rollout."""

__all__ = [
    "Event",
    "PushEventState",
    "PushEvent",
    "JumpEventState",
    "JumpEvent",
    "BaseEventState",
]

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.types import PhysicsData, PhysicsModel
from ksim.utils.mujoco import slice_update, update_data_field


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BaseEventState:
    curriculum_step: Array
    episode_length_percentage: Array
    time_remaining: Array


@attrs.define(frozen=True, kw_only=True)
class Event(ABC):
    """Base class for all events."""

    @abstractmethod
    def __call__(
        self,
        model: PhysicsModel,
        data: PhysicsData,
        event_state: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, PyTree]:
        """Apply the event to the data.

        Note that this function is called on every physics timestep, not
        control timestep - it is called by the engine directly.

        Args:
            model: The physics model.
            data: The physics data.
            event_state: The state of the event.
            rng: The random number generator.

        Returns:
            The updated data and event state.
        """

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def event_name(self) -> str:
        return self.get_name()

    @abstractmethod
    def update_curriculum_step(self, event_state: BaseEventState) -> BaseEventState:
        """Update the curriculum step."""

    @abstractmethod
    def get_initial_event_state(self, rng: PRNGKeyArray) -> BaseEventState:
        """Get the initial info for the event."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PushEventState(BaseEventState):
    pass


@attrs.define(frozen=True, kw_only=True)
class PushEvent(Event):
    """Randomly push the robot after some interval."""

    x_force: float = attrs.field()
    y_force: float = attrs.field()
    z_force: float = attrs.field(default=0.0)
    interval_range: tuple[float, float] = attrs.field()

    use_curriculum: bool = attrs.field(default=False)

    max_curriculum_steps: int = attrs.field(default=10)
    force_scale_per_step: float = attrs.field(default=0.1)
    episode_length_threshold: float = attrs.field(default=0.8)

    def __call__(
        self,
        model: PhysicsModel,
        data: PhysicsData,
        event_state: PushEventState,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, PushEventState]:
        event_state = self.update_curriculum_step(event_state)

        # Decrement by physics timestep.
        dt = jnp.float32(model.opt.timestep)
        time_remaining = event_state.time_remaining - dt

        # Update the data if the time remaining is less than 0.
        updated_data, time_remaining = jax.lax.cond(
            time_remaining <= 0.0,
            lambda: self._apply_random_force(data, rng, event_state),
            lambda: (data, time_remaining),
        )

        return updated_data, PushEventState(
            time_remaining=time_remaining,
            curriculum_step=event_state.curriculum_step,
            episode_length_percentage=event_state.episode_length_percentage,
        )

    def update_curriculum_step(self, event_state: BaseEventState) -> PushEventState:
        assert isinstance(event_state, PushEventState)

        if not self.use_curriculum:
            return event_state

        should_step = (event_state.episode_length_percentage < self.episode_length_threshold) * (
            event_state.curriculum_step < self.max_curriculum_steps
        )

        new_curriculum_step = jnp.where(should_step, event_state.curriculum_step + 1, event_state.curriculum_step)

        return PushEventState(
            time_remaining=event_state.time_remaining,
            curriculum_step=new_curriculum_step,
            episode_length_percentage=event_state.episode_length_percentage,
        )

    def _apply_random_force(
        self, data: PhysicsData, rng: PRNGKeyArray, event_state: PushEventState
    ) -> tuple[PhysicsData, Array]:
        # Randomly applies a force.

        curriculum_scale = 1.0 + self.force_scale_per_step * event_state.curriculum_step

        linear_force_scale = jnp.array([self.x_force, self.y_force, self.z_force]) * curriculum_scale
        random_forces = jax.random.uniform(rng, (3,), minval=-1.0, maxval=1.0)
        random_forces = random_forces * linear_force_scale
        new_qvel = slice_update(data, "qvel", slice(0, 3), random_forces)
        updated_data = update_data_field(data, "qvel", new_qvel)

        # Chooses a new remaining interval.
        minval, maxval = self.interval_range
        time_remaining = jax.random.uniform(rng, (), minval=minval, maxval=maxval)

        return updated_data, time_remaining

    def get_initial_event_state(self, rng: PRNGKeyArray) -> PushEventState:
        minval, maxval = self.interval_range
        time_remaining = jax.random.uniform(rng, (), minval=minval, maxval=maxval)
        return PushEventState(
            time_remaining=time_remaining, curriculum_step=jnp.array(0), episode_length_percentage=jnp.array(0)
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class JumpEventState(BaseEventState):
    pass


@attrs.define(frozen=True, kw_only=True)
class JumpEvent(Event):
    """Randomly jump the robot into the air."""

    jump_height_range: tuple[float, float] = attrs.field()
    interval_range: tuple[float, float] = attrs.field()

    def __call__(
        self,
        model: PhysicsModel,
        data: PhysicsData,
        event_state: JumpEventState,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, JumpEventState]:
        event_state = self.update_curriculum_step(event_state)

        # Decrement by physics timestep.
        dt = jnp.float32(model.opt.timestep)
        time_remaining = event_state.time_remaining - dt

        # Update the data if the time remaining is less than 0.
        updated_data, time_remaining = jax.lax.cond(
            time_remaining <= 0.0,
            lambda: self._apply_jump(model, data, rng),
            lambda: (data, time_remaining),
        )

        return updated_data, JumpEventState(
            time_remaining=time_remaining,
            curriculum_step=event_state.curriculum_step,
            episode_length_percentage=event_state.episode_length_percentage,
        )

    def _apply_jump(self, model: PhysicsModel, data: PhysicsData, rng: PRNGKeyArray) -> tuple[PhysicsData, Array]:
        # Implements a jump as a vertical velocity impulse. We compute the
        # required vertical velocity impulse to reach the desired jump height.
        minval, maxval = self.jump_height_range
        jump_height = jax.random.uniform(rng, (), minval=minval, maxval=maxval)
        new_qvel = slice_update(data, "qvel", 2, jnp.sqrt(2 * model.opt.gravity * jump_height))
        updated_data = update_data_field(data, "qvel", new_qvel)

        # Chooses a new remaining interval.
        minval, maxval = self.interval_range
        time_remaining = jax.random.uniform(rng, (), minval=minval, maxval=maxval)

        return updated_data, time_remaining
