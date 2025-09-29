"""Defines the base event classes and builders. Events are triggered during a rollout."""

__all__ = [
    "Event",
    "LinearPushEvent",
    "AngularPushEvent",
    "JumpEvent",
    "ForcePushEvent",
]

from abc import ABC, abstractmethod
from typing import Self

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.scales import ConstantScale, Scale
from ksim.types import PhysicsData, PhysicsModel
from ksim.utils.mujoco import get_body_data_idx_by_name, slice_update, update_data_field


def convert_to_scale(value: float | int | Scale) -> Scale:
    if isinstance(value, (float, int)):
        return ConstantScale(scale=value)
    if isinstance(value, Scale):
        return value
    raise ValueError(f"Invalid scale: {value}")


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
class AngularPushEvent(Event):
    """Randomly push the robot in a linear direction."""

    angvel: float = attrs.field()
    vel_range: tuple[float, float] = attrs.field(default=(0.0, 1.0))
    interval_range: tuple[float, float] = attrs.field()
    scale: Scale = attrs.field(default=ConstantScale(scale=1.0), converter=convert_to_scale)

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
        frng, brng, trng = jax.random.split(rng, 3)

        # Scales the curriculum level range.
        curriculum_level = self.scale.get_scale(curriculum_level)

        flip = jax.random.bernoulli(frng, p=0.5, shape=())
        push_mag = jax.random.uniform(brng, (), minval=self.vel_range[0], maxval=self.vel_range[1]) * self.angvel
        push_mag = jnp.where(flip, -push_mag, push_mag)
        push_vel = push_mag * curriculum_level
        new_qvel = slice_update(data, "qvel", slice(5, 6), data.qvel[..., 5:6] + push_vel)
        updated_data = update_data_field(data, "qvel", new_qvel)

        # Chooses a new remaining interval.
        minval, maxval = self.interval_range
        time_remaining = jax.random.uniform(trng, (), minval=minval, maxval=maxval)
        return updated_data, time_remaining

    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        minval, maxval = self.interval_range
        return jax.random.uniform(rng, (), minval=minval, maxval=maxval)


@attrs.define(frozen=True, kw_only=True)
class LinearPushEvent(Event):
    """Randomly push the robot in a linear direction."""

    linvel: float = attrs.field()
    vel_range: tuple[float, float] = attrs.field(default=(0.0, 1.0))
    interval_range: tuple[float, float] = attrs.field()
    scale: Scale = attrs.field(default=ConstantScale(scale=1.0), converter=convert_to_scale)

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
        curriculum_level = self.scale.get_scale(curriculum_level)

        push_theta = jax.random.uniform(urng, (), minval=0.0, maxval=2.0 * jnp.pi)
        push_theta = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta), 0.0])

        push_mag = jax.random.uniform(brng, (), minval=self.vel_range[0], maxval=self.vel_range[1]) * self.linvel
        push_vel = push_theta * push_mag * curriculum_level
        new_qvel = slice_update(data, "qvel", slice(0, 3), data.qvel[..., :3] + push_vel)
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
    scale: Scale = attrs.field(default=ConstantScale(scale=1.0), converter=convert_to_scale)

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
        curriculum_level = self.scale.get_scale(curriculum_level)

        # Implements a jump as a vertical velocity impulse. We compute the
        # required vertical velocity impulse to reach the desired jump height.
        minval, maxval = self.jump_height_range
        jump_height = jax.random.uniform(urng, (), minval=minval, maxval=maxval) * curriculum_level
        linvel = jnp.sqrt(2 * -model.opt.gravity * jump_height)
        angvel = jnp.zeros(3)
        vel = jnp.concatenate([linvel, angvel], axis=0)
        new_qvel = slice_update(data, "qvel", slice(0, 6), data.qvel[..., :6] + vel)
        updated_data = update_data_field(data, "qvel", new_qvel)

        # Chooses a new remaining interval.
        minval, maxval = self.interval_range
        time_remaining = jax.random.uniform(trng, (), minval=minval, maxval=maxval)

        return updated_data, time_remaining

    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        minval, maxval = self.interval_range
        return jax.random.uniform(rng, (), minval=minval, maxval=maxval)


@attrs.define(frozen=True, kw_only=True)
class ForcePushEvent(Event):
    """Apply a continuous force to the robot over time.

    Unlike LinearPushEvent which sets velocity.
    """

    max_force: float = attrs.field()
    max_torque: float = attrs.field()
    force_range: tuple[float, float] = attrs.field(default=(0.0, 1.0))
    duration_range: tuple[float, float] = attrs.field()
    interval_range: tuple[float, float] = attrs.field()
    body_id: int = attrs.field()

    @classmethod
    def from_body_name(
        cls,
        model: PhysicsModel,
        body_name: str,
        *,
        max_force: float,
        max_torque: float,
        duration_range: tuple[float, float],
        interval_range: tuple[float, float],
        force_range: tuple[float, float] = (0.0, 1.0),
    ) -> Self:
        names_to_idxs = get_body_data_idx_by_name(model)
        if body_name not in names_to_idxs:
            raise ValueError(f"Body name {body_name} not found in model")
        body_id = names_to_idxs[body_name]
        return cls(
            max_force=max_force,
            max_torque=max_torque,
            duration_range=duration_range,
            interval_range=interval_range,
            force_range=force_range,
            body_id=body_id,
        )

    def __call__(
        self,
        model: PhysicsModel,
        data: PhysicsData,
        event_state: PyTree,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, tuple[Array, Array, Array]]:
        time_remaining, force_duration, force = event_state

        # Decrement timers by physics timestep
        dt = jnp.float32(model.opt.timestep)
        time_remaining = time_remaining - dt
        force_duration = force_duration - dt

        # Apply force and torque or zero it out based on duration
        force = force * curriculum_level
        new_xfrc = jax.lax.cond(
            force_duration > 0.0, lambda _: force * curriculum_level, lambda _: jnp.zeros_like(force), None
        )

        # Set xfrc_applied
        data = update_data_field(data, "xfrc_applied", jnp.zeros_like(data.xfrc_applied))
        data = update_data_field(
            data,
            "xfrc_applied",
            slice_update(data, "xfrc_applied", slice(self.body_id, self.body_id + 1), new_xfrc[None]),
        )

        # If interval expired, sample next force
        time_remaining, force_duration, force = jax.lax.cond(
            time_remaining <= 0.0,
            lambda _: self._sample_force(rng),
            lambda _: (time_remaining, force_duration, force),
            None,
        )

        event_state = (time_remaining, force_duration, force)
        return data, event_state

    def _sample_force(self, rng: PRNGKeyArray) -> tuple[Array, Array, Array]:
        rng_splits = jax.random.split(rng, 6)

        # Sample random force and torque
        direction = jax.random.uniform(rng_splits[0], shape=(3,), minval=-1.0, maxval=1.0)
        force_scale = jax.random.uniform(rng_splits[1], (), minval=self.force_range[0], maxval=self.force_range[1])
        force = direction / jnp.linalg.norm(direction) * force_scale * self.max_force
        torque_scale = jax.random.uniform(rng_splits[2], shape=(3,), minval=-1.0, maxval=1.0)
        torque = torque_scale / jnp.linalg.norm(torque_scale) * force_scale * self.max_torque

        # Sample one of three cases: force only, torque only, or both
        force = jax.lax.switch(
            jax.random.randint(rng_splits[3], (), 0, 3),
            [
                lambda _: jnp.concatenate([force, jnp.zeros_like(torque)]),
                lambda _: jnp.concatenate([jnp.zeros_like(force), torque]),
                lambda _: jnp.concatenate([force, torque]),
            ],
            None,
        )

        # Initial timers
        time_remaining = jax.random.uniform(
            rng_splits[4], (), minval=self.interval_range[0], maxval=self.interval_range[1]
        )
        force_duration = jax.random.uniform(
            rng_splits[5], (), minval=self.duration_range[0], maxval=self.duration_range[1]
        )

        return (time_remaining, force_duration, force)

    def get_initial_event_state(self, rng: PRNGKeyArray) -> tuple[Array, Array, Array]:
        minval, maxval = self.interval_range
        return (
            jax.random.uniform(rng, (), minval=minval, maxval=maxval),
            jnp.array(0.0, dtype=jnp.float32),
            jnp.zeros(6, dtype=jnp.float32),
        )
