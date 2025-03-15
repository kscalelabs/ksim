"""Base JAX centric environment class.

Designed to be easily extensible to any physics engine that separates
model and data.
"""

from abc import ABC, abstractmethod

from jaxtyping import Array, PRNGKeyArray

from ksim.env.data import PhysicsState


class BaseEngine(ABC):
    """The role of an engine is simple: reset and step. Decoupled from data."""

    @abstractmethod
    def reset(self, rng: PRNGKeyArray) -> PhysicsState:
        """Reset the engine and return the physics model and data."""

    @abstractmethod
    def step(
        self,
        action: Array,
        physics_state: PhysicsState,
    ) -> PhysicsState:
        """Step the engine and return the physics model and data."""
