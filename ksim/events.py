"""Defines the base event classes and builders. Events are triggered during a rollout."""

__all__ = [
    "Event",
    "PushEvent",
    "PushEventInfo",
]

from abc import ABC, abstractmethod
from typing import Self

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


@jax.tree_util.register_pytree_node_class
@attrs.define(frozen=True, kw_only=True)
class PushEventInfo:
    remaining_interval: Array

    def tree_flatten(self) -> tuple[tuple, None]:
        return (self.remaining_interval,), None

    @classmethod
    def tree_unflatten(cls, aux_data: None, children: tuple) -> Self:
        """Reconstruct the class from flattened representation."""
        remaining_interval, = children
        return cls(
            remaining_interval=remaining_interval,
        )


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
        remaining_interval: Array  # Remaining time in seconds before next push
        """
        # Split the RNG for different operations
        rng1, rng2 = jax.random.split(rng)

        needs_reset = persistent_data.remaining_interval[0] <= 0.0
        reset_prob = jax.random.uniform(rng1)
        should_reset = needs_reset & (reset_prob < self.probability)

        rng_interval, rng_linear, rng_angular = jax.random.split(rng2, 3)

        # Calculate new interval (either new random interval or decremented existing one)
        interval_range = self.interval_range
        # Generate random interval in seconds
        random_interval = jax.random.uniform(
            rng_interval, 
            (1,), 
            minval=interval_range[0], 
            maxval=interval_range[1]
        )

        # Decrement by physics timestep (in seconds)
        continued_interval = persistent_data.remaining_interval - data.dt

        # Select new interval value
        new_interval = jnp.where(
            should_reset,
            random_interval,
            jnp.where(needs_reset, persistent_data.remaining_interval, continued_interval),
        )

        updated_data = jax.lax.cond(
            jnp.bool_(should_reset),
            lambda: self._apply_push_force(data, rng_linear, rng_angular),
            lambda: data,
        )

        return updated_data, PushEventInfo(
            remaining_interval=new_interval,
        )
    
    def _apply_push_force(self, data: PhysicsData, rng_linear: PRNGKeyArray, rng_angular: PRNGKeyArray) -> PhysicsData:
        """Generate and apply random forces to the physics data."""
        random_linear_force = jax.random.uniform(
            rng_linear,
            (3,),
            minval=-self.linear_force_scale,
            maxval=self.linear_force_scale,
        ) + data.qvel[0:3]

        random_angular_force = jax.random.uniform(
            rng_angular,
            (3,),
            minval=-self.angular_force_scale,
            maxval=self.angular_force_scale,
        )
        
        new_data_qvel = data.qvel
        new_data_qvel = new_data_qvel.at[0:3].set(random_linear_force)
        
        return update_data_field(data, "qvel", new_data_qvel)

    def get_initial_info(self) -> PushEventInfo:
        return PushEventInfo(
            remaining_interval=jnp.array(self.interval_range[0])
        )
