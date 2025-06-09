"""Defines the base event classes and builders. Events are triggered during a rollout."""

__all__ = [
    "Event",
    "PushEvent",
    "JumpEvent",
    "JointPerturbationEvent",
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

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def event_name(self) -> str:
        return self.get_name()

    @abstractmethod
    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        """Get the initial info for the event."""


@attrs.define(frozen=True, kw_only=True)
class PushEvent(Event):
    """Randomly push the robot after some interval."""

    x_linvel: float = attrs.field()
    y_linvel: float = attrs.field()
    z_linvel: float = attrs.field(default=0.0)
    x_angvel: float = attrs.field(default=0.0)
    y_angvel: float = attrs.field(default=0.0)
    z_angvel: float = attrs.field(default=0.0)
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

        # Randomly applies a force.
        vel_scales = jnp.array(
            [
                self.x_linvel,
                self.y_linvel,
                self.z_linvel,
                self.x_angvel,
                self.y_angvel,
                self.z_angvel,
            ]
        )
        vel_min, vel_max = self.vel_range
        random_vels = jax.random.uniform(urng, shape=(6,), minval=vel_min, maxval=vel_max)
        random_flip = jax.random.bernoulli(brng, p=0.5, shape=(6,)).astype(random_vels.dtype) * 2 - 1
        random_vels = random_vels * vel_scales * curriculum_level * random_flip
        random_vels += data.qvel[:6]
        new_qvel = slice_update(data, "qvel", slice(0, 6), random_vels)
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


@attrs.define(frozen=True, kw_only=True)
class JointPerturbationEvent(Event):
    """Randomly adds some velocity to each joint."""

    std: float = attrs.field(validator=attrs.validators.gt(0.0))
    mask_prct: float = attrs.field(
        default=0.0,
        validator=attrs.validators.and_(
            attrs.validators.ge(0.0),
            attrs.validators.le(1.0),
        ),
    )
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
            lambda: self._apply_random_velocity(data, curriculum_level, rng),
            lambda: (data, time_remaining),
        )

        return updated_data, time_remaining

    def _apply_random_velocity(
        self,
        data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, Array]:
        urng, mrng, trng = jax.random.split(rng, 3)

        # Scales the curriculum level range.
        curriculum_min, curriculum_max = self.curriculum_range
        curriculum_level = curriculum_level * (curriculum_max - curriculum_min) + curriculum_min

        # Randomly applies a velocity to a single joint.
        std = self.std
        nomask_prct = 1.0 - self.mask_prct
        qfrc_applied = data.qfrc_applied[..., 6:]
        random_vels = jax.random.normal(urng, shape=qfrc_applied.shape) * std
        random_flip = jax.random.bernoulli(urng, p=0.5, shape=qfrc_applied.shape).astype(random_vels.dtype) * 2 - 1
        random_mask = jax.random.bernoulli(mrng, p=nomask_prct, shape=qfrc_applied.shape).astype(random_vels.dtype)
        random_vels = random_vels * curriculum_level * random_flip * random_mask
        qfrc_applied = slice_update(data, "qfrc_applied", slice(6, None), qfrc_applied + random_vels)
        updated_data = update_data_field(data, "qfrc_applied", qfrc_applied)

        # Chooses a new remaining interval.
        minval, maxval = self.interval_range
        time_remaining = jax.random.uniform(trng, (), minval=minval, maxval=maxval)
        return updated_data, time_remaining

    def get_initial_event_state(self, rng: PRNGKeyArray) -> Array:
        minval, maxval = self.interval_range
        return jax.random.uniform(rng, (), minval=minval, maxval=maxval)
