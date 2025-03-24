"""Defines the base event classes and builders. Events are triggered during a rollout."""

__all__ = [
    "Event",
    "PushEvent",
    "PushEventInfo",
]

from abc import ABC, abstractmethod

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.types import PhysicsData
from ksim.utils.mujoco import update_data_field


def event_probability_validator(inst: "Event", attr: attrs.Attribute, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"Event probability must be between 0.0 and 1.0, got {value}")


@attrs.define(frozen=True, kw_only=True)
class Event(ABC):
    """Base class for all events."""

    probability: float = attrs.field(validator=event_probability_validator)

    @abstractmethod
    def __call__(self, persistent_data: PyTree, data: PhysicsData, rng: PRNGKeyArray) -> tuple[PhysicsData, PyTree]:
        """Apply the event to the data."""

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @abstractmethod
    def get_initial_info(self) -> PyTree:
        """Get the initial info for the event."""


@attrs.define(frozen=True, kw_only=True)
class PushEventInfo:
    remaining_interval: Array
    linear_force: Array
    angular_force: Array


@attrs.define(frozen=True, kw_only=True)
class PushEvent(Event):
    """Event for pushing the robot."""

    linear_force_scale: float = attrs.field(default=1.0)
    angular_force_scale: float = attrs.field(default=1.0)
    interval_range: tuple[float, float] = attrs.field(default=(0.0, 0.0))

    def __call__(
        self,
        persistent_data: PushEventInfo,
        data: PhysicsData,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, PushEventInfo]:
        """Apply the event to the data.

        Persistent data has the following structure:
        (
            remaining_interval: Array,
            linear_force: (Array, Array, Array),
            angular_force: (Array, Array, Array),
        )
        """
        # Split the RNG for different operations
        rng1, rng2 = jax.random.split(rng)

        # Determine whether to reset based on interval and probability
        needs_reset = persistent_data.remaining_interval <= 0.0
        reset_prob = jax.random.uniform(rng1)
        should_reset = needs_reset & (reset_prob < self.probability)

        # Generate new values
        rng_interval, rng_linear, rng_angular = jax.random.split(rng2, 3)

        # Calculate new interval (either new random interval or decremented existing one)
        interval_range = self.interval_range
        random_interval = jax.random.randint(rng_interval, (1,), minval=interval_range[0], maxval=interval_range[1])
        continued_interval = persistent_data.remaining_interval - 1

        # Select new interval value
        new_interval = jnp.where(
            should_reset,
            random_interval,
            jnp.where(needs_reset, persistent_data.remaining_interval, continued_interval),
        )

        # Generate random forces if needed
        random_linear_force = jax.random.uniform(
            rng_linear,
            (3,),
            minval=-self.linear_force_scale,
            maxval=self.linear_force_scale,
        )
        random_angular_force = jax.random.uniform(
            rng_angular,
            (3,),
            minval=-self.angular_force_scale,
            maxval=self.angular_force_scale,
        )

        # Select forces based on reset condition
        new_linear_force = jnp.where(should_reset, random_linear_force, persistent_data.linear_force)
        new_angular_force = jnp.where(should_reset, random_angular_force, persistent_data.angular_force)

        # Apply force when appropriate (either continuing or newly reset)
        should_apply_force = (~needs_reset) | should_reset

        new_data_qvel = data.qvel
        new_data_qvel = new_data_qvel.at[0:3].set(new_linear_force)

        # Use JAX's functional update for data
        updated_data = jax.lax.cond(
            should_apply_force,
            lambda: update_data_field(data, "qvel", new_data_qvel),
            lambda: data,
        )

        return updated_data, PushEventInfo(
            remaining_interval=new_interval,
            linear_force=new_linear_force,
            angular_force=new_angular_force,
        )

    def get_initial_info(self) -> PushEventInfo:
        return PushEventInfo(
            remaining_interval=jnp.zeros(1),
            linear_force=jnp.zeros(3),
            angular_force=jnp.zeros(3),
        )
