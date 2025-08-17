"""Defines the base event classes and builders. Events are triggered during a rollout."""

__all__ = [
    "Event",
    "LinearPushEvent",
    "JumpEvent",
]

from abc import ABC, abstractmethod

import attrs
import jax
import jax.numpy as jnp
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
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, Array]:
        """Apply the event to the data.

        Note that this function is called on every physics timestep, not
        control timestep - it is called by the engine directly.

        Args:
            model: The physics model.
            data: The physics data.
            event_state: The state of the event.
            curriculum_level: The curriculum level.
            rng: The random number generator.

        Returns:
            The updated data and event state.
        """

    @abstractmethod
    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        """Get the initial info for the event."""


@attrs.define(frozen=True, kw_only=True)
class LinearPushEvent(Event):
    """Randomly push the robot in a linear direction."""

    linvel: float = attrs.field()
    vel_range: tuple[float, float] = attrs.field(default=(0.0, 1.0))
    interval_range: tuple[float, float] = attrs.field()
    curriculum_range: tuple[float, float] = attrs.field(default=(0.0, 1.0))

    def __call__(
        self,
        model: PhysicsModel,
        data: PhysicsData,
        event_state: Array,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, Array]:
        # Decrement by physics timestep.
        dt = jnp.float32(model.opt.timestep)
        time_remaining = event_state - dt

        # Update the data if the time remaining is less than 0.
        updated_data, time_remaining = jax.lax.cond(
            time_remaining <= 0.0,
            lambda: self._apply_random_force(data, curriculum_level, rng),
            lambda: (data, time_remaining),
        )

        return updated_data, time_remaining

    def _apply_random_force(
        self,
        data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, Array]:
        urng, brng, trng = jax.random.split(rng, 3)

        # Scales the curriculum level range.
        curriculum_min, curriculum_max = self.curriculum_range
        curriculum_level = curriculum_level * (curriculum_max - curriculum_min) + curriculum_min

        push_theta = jax.random.uniform(urng, (), minval=0.0, maxval=2.0 * jnp.pi)
        push_theta = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta), 0.0])

        push_mag = jax.random.uniform(brng, (), minval=self.vel_range[0], maxval=self.vel_range[1])
        push_vel = push_theta * push_mag * curriculum_level
        new_qvel = slice_update(data, "qvel", slice(0, 3), data.qvel[:3] + push_vel)
        updated_data = update_data_field(data, "qvel", new_qvel)

        # Chooses a new remaining interval.
        minval, maxval = self.interval_range
        time_remaining = jax.random.uniform(trng, (), minval=minval, maxval=maxval)
        return updated_data, time_remaining

    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        minval, maxval = self.interval_range
        return jax.random.uniform(rng, (), minval=minval, maxval=maxval)


@attrs.define(frozen=True, kw_only=True)
class JumpEvent(Event):
    """Randomly jump the robot into the air."""

    jump_height_range: tuple[float, float] = attrs.field()
    interval_range: tuple[float, float] = attrs.field()
    curriculum_range: tuple[float, float] = attrs.field(default=(0.0, 1.0))

    def __call__(
        self,
        model: PhysicsModel,
        data: PhysicsData,
        event_state: Array,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, Array]:
        # Decrement by physics timestep.
        dt = jnp.float32(model.opt.timestep)
        time_remaining = event_state - dt

        # Update the data if the time remaining is less than 0.
        updated_data, time_remaining = jax.lax.cond(
            time_remaining <= 0.0,
            lambda: self._apply_jump(model, data, curriculum_level, rng),
            lambda: (data, time_remaining),
        )

        return updated_data, time_remaining

    def _apply_jump(
        self,
        model: PhysicsModel,
        data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, Array]:
        urng, trng = jax.random.split(rng, 2)

        # Scales the curriculum level range.
        curriculum_min, curriculum_max = self.curriculum_range
        curriculum_level = curriculum_level * (curriculum_max - curriculum_min) + curriculum_min

        # Implements a jump as a vertical velocity impulse. We compute the
        # required vertical velocity impulse to reach the desired jump height.
        minval, maxval = self.jump_height_range
        jump_height = jax.random.uniform(urng, (), minval=minval, maxval=maxval) * curriculum_level
        linvel = jnp.sqrt(2 * -model.opt.gravity * jump_height)
        angvel = jnp.zeros(3)
        vel = jnp.concatenate([linvel, angvel], axis=0)
        new_qvel = slice_update(data, "qvel", slice(0, 6), data.qvel[:6] + vel)
        updated_data = update_data_field(data, "qvel", new_qvel)

        # Chooses a new remaining interval.
        minval, maxval = self.interval_range
        time_remaining = jax.random.uniform(trng, (), minval=minval, maxval=maxval)

        return updated_data, time_remaining

    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        minval, maxval = self.interval_range
        return jax.random.uniform(rng, (), minval=minval, maxval=maxval)
