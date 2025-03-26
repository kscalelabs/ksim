"""Defines the base event classes and builders. Events are triggered during a rollout."""

__all__ = [
    "Event",
    "PushEvent",
    "PushEventInfo",
]

import functools
from abc import ABC, abstractmethod
from typing import Self

import attrs
import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from mujoco import mjx

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
    def __call__(
        self, persistent_data: PyTree, data: PhysicsData, dt: float, rng: PRNGKeyArray
    ) -> tuple[PhysicsData, PyTree]:
        """Apply the event to the data."""

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def event_name(self) -> str:
        return self.get_name()

    @abstractmethod
    def get_initial_info(self) -> PyTree:
        """Get the initial info for the event."""


@jax.tree_util.register_pytree_node_class
@attrs.define(frozen=True, kw_only=True)
class PushEventInfo:
    remaining_interval: Array

    def tree_flatten(self) -> tuple[tuple, None]:
        return (self.remaining_interval,), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple) -> Self:
        """Reconstruct the class from flattened representation."""
        (remaining_interval,) = children
        return cls(
            remaining_interval=remaining_interval,
        )


@attrs.define(frozen=True, kw_only=True)
class PushEvent(Event):
    """Event for pushing the robot."""

    linear_force_scale: float = attrs.field(default=0.0)
    angular_force_scale: float = attrs.field(default=0.0)
    interval_range: tuple[float, float] = attrs.field(default=(0.0, 0.0))

    def __call__(
        self,
        persistent_data: PushEventInfo,
        data: PhysicsData,
        dt: float,
        rng: PRNGKeyArray,
    ) -> tuple[PhysicsData, PushEventInfo]:
        """Apply the event to the data.

        Persistent data has the following structure:
        remaining_interval: Array  # Remaining time in seconds before next push
        """
        # Split the RNG for different operations
        rng1, rng2 = jax.random.split(rng)

        needs_reset = persistent_data.remaining_interval <= 0.0
        reset_prob = jax.random.uniform(rng1)
        should_reset = needs_reset & (reset_prob < self.probability)

        rng_interval, rng_linear, rng_angular = jax.random.split(rng2, 3)

        # Calculate new interval (either new random interval or decremented existing one)
        interval_range = self.interval_range
        # Generate random interval in seconds - ensure it's float32
        random_interval = jax.random.uniform(
            rng_interval,
            (),
            minval=jnp.float32(interval_range[0]),
            maxval=jnp.float32(interval_range[1]),
        )

        # Decrement by physics timestep
        dt_float32 = jnp.float32(dt)
        continued_interval = persistent_data.remaining_interval - dt_float32

        # Select new interval value
        new_interval = jnp.where(
            should_reset,
            random_interval,
            jnp.where(needs_reset, jnp.float32(0.0), continued_interval),
        )

        # Generate random forces
        random_linear_force = (
            jax.random.uniform(
                rng_linear,
                (3,),
                minval=-self.linear_force_scale,
                maxval=self.linear_force_scale,
            )
            + data.qvel[0:3]
        )

        # Apply forces conditionally using where instead of lax.cond
        match type(data):
            case mujoco.MjData:
                new_data_qvel = data.qvel.copy()
                new_data_qvel[0:3] = jnp.where(should_reset, random_linear_force, data.qvel[0:3])
            case mjx.Data:
                new_data_qvel = data.qvel.at[0:3].set(jnp.where(should_reset, random_linear_force, data.qvel[0:3]))

        # Update data with new velocities
        updated_data = update_data_field(data, "qvel", new_data_qvel)

        return updated_data, PushEventInfo(
            remaining_interval=new_interval,
        )

    def get_initial_info(self) -> PushEventInfo:
        """Initialize with a float32 zero value for consistent typing."""
        return PushEventInfo(remaining_interval=jnp.float32(0.0))
