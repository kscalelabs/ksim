"""Defines the base randomization class, along with some implementations.

To be merged with the reset classes which mutates the data in place.
"""

import functools
from abc import ABC, abstractmethod

import attrs
import jax
from jax import Array as JaxArray
from omegaconf import MISSING

from ksim.env.types import PhysicsModel


@attrs.define(frozen=True, kw_only=True)
class Randomization(ABC):
    """Randomizes the observations and commands."""

    @abstractmethod
    def __call__(self, model: PhysicsModel, rng: JaxArray) -> PhysicsModel:
        """Randomizes the model."""

    @functools.cached_property
    def randomization_name(self) -> str:
        return self.name


@attrs.define(frozen=True, kw_only=True)
class StaticFrictionRandomization(Randomization):
    """Randomizes the static friction."""

    name: str = "dof_frictionloss"
    scale_lower: float = attrs.field(default=0.5)
    scale_upper: float = attrs.field(default=2.0)

    def __call__(self, model: PhysicsModel, rng: JaxArray) -> PhysicsModel:
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss * jax.random.uniform(
            key, shape=(model.nq,), minval=self.scale_lower, maxval=self.scale_upper
        )
        dof_frictionloss = model.dof_frictionloss.at[:].set(frictionloss)

        return dof_frictionloss


@attrs.define(frozen=True, kw_only=True)
class FloorFrictionRandomization(Randomization):
    """Randomizes the floor friction."""

    name: str = "geom_friction"
    scale_lower: float = attrs.field(default=0.4)
    scale_upper: float = attrs.field(default=1.0)
    floor_body_id: int = attrs.field(default=MISSING)

    def __call__(self, model: PhysicsModel, rng: JaxArray) -> PhysicsModel:
        geom_friction = model.geom_friction.at[self.floor_body_id, 0].set(
            jax.random.uniform(rng, minval=self.scale_lower, maxval=self.scale_upper)
        )
        return geom_friction


@attrs.define(frozen=True, kw_only=True)
class ArmatureRandomization(Randomization):
    """Randomizes the armature."""

    name: str = "dof_armature"
    scale_lower: float = attrs.field(default=1.0)
    scale_upper: float = attrs.field(default=1.05)

    def __call__(self, model: PhysicsModel, rng: JaxArray) -> PhysicsModel:
        armature = model.dof_armature * jax.random.uniform(
            rng, shape=(model.nq,), minval=self.scale_lower, maxval=self.scale_upper
        )
        dof_armature = model.dof_armature.at[:].set(armature)
        return dof_armature


@attrs.define(frozen=True, kw_only=True)
class LinkMassRandomization(Randomization):
    """Randomizes the link masses."""

    name: str = "body_mass"
    scale_lower: float = attrs.field(default=-1.0)
    scale_upper: float = attrs.field(default=1.0)

    def __call__(self, model: PhysicsModel, rng: JaxArray) -> PhysicsModel:
        dmass = jax.random.uniform(rng, shape=(model.nbody,), minval=self.scale_lower, maxval=self.scale_upper)
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)

        return body_mass


@attrs.define(frozen=True, kw_only=True)
class TorsoMassRandomization(Randomization):
    """Randomizes the torso mass."""

    name: str = "body_mass"
    scale_lower: float = attrs.field(default=-1.0)
    scale_upper: float = attrs.field(default=1.0)
    torso_body_id: int = attrs.field(default=MISSING)

    def __call__(self, model: PhysicsModel, rng: JaxArray) -> PhysicsModel:
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=self.scale_lower, maxval=self.scale_upper)
        body_mass = model.body_mass.at[self.torso_body_id].set(model.body_mass[self.torso_body_id] + dmass)
        return body_mass


@attrs.define(frozen=True, kw_only=True)
class JointDampingRandomization(Randomization):
    """Randomizes the joint damping."""

    name: str = "dof_damping"
    scale_lower: float = attrs.field(default=0.9)
    scale_upper: float = attrs.field(default=1.1)

    def __call__(self, model: PhysicsModel, rng: JaxArray) -> PhysicsModel:
        rng, key = jax.random.split(rng)
        kd = model.dof_damping * jax.random.uniform(
            key, shape=(model.nq,), minval=self.scale_lower, maxval=self.scale_upper
        )
        dof_damping = model.dof_damping.at[:].set(kd)
        return dof_damping
