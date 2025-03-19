"""Randomize each environment when gathering trajectories."""

from abc import ABC, abstractmethod

import attrs
import jax
from jaxtyping import PRNGKeyArray

from ksim.env.data import PhysicsModel
from ksim.utils.mujoco import update_model_field


@attrs.define(frozen=True, kw_only=True)
class Randomization(ABC):
    """Randomize the joint positions of the robot."""

    @abstractmethod
    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        """Randomize the model for a single environment."""


@attrs.define(frozen=True, kw_only=True)
class WeightRandomization(Randomization):
    """Randomize the joint positions of the robot."""

    scale: float = attrs.field()

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        """Randomize the model for a single environment."""
        new_body_mass = model.body_mass * jax.random.normal(rng, model.body_mass.shape) * self.scale
        return update_model_field(model, "body_mass", new_body_mass)
