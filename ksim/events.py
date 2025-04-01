"""Defines the base event classes and builders. Events are triggered during a rollout."""

__all__ = [
    "Event",
    "PushEvent",
    "JumpEvent",
]

import functools
from abc import ABC, abstractmethod

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.types import PhysicsData, PhysicsModel
from ksim.utils.mujoco import slice_update, update_data_field


@attrs.define(frozen=True, kw_only=True)
class Event(ABC):
    """Base class for all events."""

    @abstractmethod
    def __call__(
        self,
        model: PhysicsModel,
        data: PhysicsData,
        event_state: PyTree,
        curriculum_step: int,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, Array]:
        """Apply the event to the data.

        Note that this function is called on every physics timestep, not
        control timestep - it is called by the engine directly.

        Args:
            model: The physics model.
            data: The physics data.
            event_state: The state of the event.
            curriculum_step: The current curriculum step.
            rng: The random number generator.

        Returns:
            The updated data and event state.
        """

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def event_name(self) -> str:
        return self.get_name()

    def should_step_curriculum(self, ep_len_pct: Array, curr_step: int) -> Array:
        """Determine if the curriculum should step based on episode length percentage.

        Args:
            ep_len_pct: The percentage of the episode length that was completed.
            curr_step: The current curriculum step.

        Returns:
            A boolean array indicating whether to step the curriculum.
        """
        return jnp.array(False)

    @abstractmethod
    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        """Get the initial info for the event."""


@attrs.define(frozen=True, kw_only=True)
class PushEvent(Event):
    """Randomly push the robot after some interval."""

    x_force: float = attrs.field()
    y_force: float = attrs.field()
    z_force: float = attrs.field(default=0.0)
    x_angular_force: float = attrs.field(default=0.0)
    y_angular_force: float = attrs.field(default=0.0)
    z_angular_force: float = attrs.field(default=0.0)
    interval_range: tuple[float, float] = attrs.field()

    use_curriculum: bool = attrs.field(default=False)

    max_curriculum_steps: int = attrs.field(default=10)
    force_scale_per_step: float = attrs.field(default=0.1)
    episode_length_threshold: float = attrs.field(default=0.8)

    def __call__(
        self,
        model: PhysicsModel,
        data: PhysicsData,
        event_state: Array,
        curriculum_step: int,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, Array]:
        # Decrement by physics timestep.
        dt = jnp.float32(model.opt.timestep)
        time_remaining = event_state - dt

        # Update the data if the time remaining is less than 0.
        updated_data, time_remaining = jax.lax.cond(
            time_remaining <= 0.0,
            lambda: self._apply_random_force(data, rng, curriculum_step),
            lambda: (data, time_remaining),
        )

        return updated_data, time_remaining

    def should_step_curriculum(self, ep_len_pct: Array, curr_step: int) -> Array:
        """Determine if the curriculum should step based on episode length percentage.

        Args:
            ep_len_pct: The percentage of the episode length that was completed.
            curr_step: The current curriculum step.

        Returns:
            A boolean array indicating whether to step the curriculum.
        """
        if not self.use_curriculum:
            return jnp.array(False)

        should_step = jnp.logical_and(
            ep_len_pct >= self.episode_length_threshold, curr_step < self.max_curriculum_steps
        )

        return should_step

    def _apply_random_force(
        self, data: PhysicsData, rng: PRNGKeyArray, curriculum_step: int
    ) -> tuple[PhysicsData, Array]:
        # Randomly applies a force.
        curriculum_scale = 1.0 + self.force_scale_per_step * curriculum_step

        force_scales = jnp.array(
            [
                self.x_force,
                self.y_force,
                self.z_force,
                self.x_angular_force,
                self.y_angular_force,
                self.z_angular_force,
            ]
        )
        random_forces = jax.random.uniform(rng, (6,), minval=-1.0, maxval=1.0)
        random_forces = random_forces * force_scales * curriculum_scale
        new_qvel = slice_update(data, "qvel", slice(0, 6), random_forces)
        updated_data = update_data_field(data, "qvel", new_qvel)

        # Chooses a new remaining interval.
        minval, maxval = self.interval_range
        time_remaining = jax.random.uniform(rng, (), minval=minval, maxval=maxval)

        return updated_data, time_remaining

    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        minval, maxval = self.interval_range
        return jax.random.uniform(rng, (), minval=minval, maxval=maxval)


@attrs.define(frozen=True, kw_only=True)
class JumpEvent(Event):
    """Randomly jump the robot into the air."""

    jump_height_range: tuple[float, float] = attrs.field()
    interval_range: tuple[float, float] = attrs.field()

    def __call__(
        self,
        model: PhysicsModel,
        data: PhysicsData,
        event_state: Array,
        curriculum_step: int,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, Array]:
        # Decrement by physics timestep.
        dt = jnp.float32(model.opt.timestep)
        time_remaining = event_state - dt

        # Update the data if the time remaining is less than 0.
        updated_data, time_remaining = jax.lax.cond(
            time_remaining <= 0.0,
            lambda: self._apply_jump(model, data, rng),
            lambda: (data, time_remaining),
        )

        return updated_data, time_remaining

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

    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        minval, maxval = self.interval_range
        time_remaining = jax.random.uniform(rng, (), minval=minval, maxval=maxval)
        return time_remaining

    def should_step_curriculum(self, ep_len_pct: Array, curr_step: int) -> Array:
        """Determine if the curriculum should step based on episode length percentage.

        Args:
            ep_len_pct: The percentage of the episode length that was completed.
            curr_step: The current curriculum step.
        """
        return jnp.array(False)
