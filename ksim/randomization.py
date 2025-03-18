"""Defines the base Randomizer class, along with some implementations.

To be merged with the reset classes which mutates the data in place.
"""

import functools
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import attrs
import jax
from jaxtyping import Array
from omegaconf import MISSING

from ksim.env.data import PhysicsData, PhysicsModel
from ksim.utils.named_access import get_body_data_idx_by_name


@attrs.define(frozen=True, kw_only=True)
class Randomizer(ABC):
    """Randomizes the observations and commands."""

    @abstractmethod
    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: Array) -> tuple[PhysicsModel, PhysicsData]:
        """Randomizes the model."""

    @functools.cached_property
    def name(self) -> str:
        return self.name


T = TypeVar("T", bound=Randomizer)


class RandomizerBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, physics_model: PhysicsModel, data: PhysicsData, rng: Array) -> T:
        """Builds a randomizer from a MuJoCo model."""


@attrs.define(frozen=True, kw_only=True)
class StaticFrictionRandomizer(Randomizer):
    """Randomizes the static friction."""

    name: str = "dof_frictionloss"
    scale_lower: float = attrs.field(default=0.5)
    scale_upper: float = attrs.field(default=2.0)

    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: Array) -> tuple[PhysicsModel, PhysicsData]:
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss + jax.random.uniform(
            key,
            shape=(model.dof_frictionloss.shape[0],),
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        dof_frictionloss = model.dof_frictionloss.at[:].set(frictionloss)
        model = model.tree_replace({self.name: dof_frictionloss})
        return model, data


@attrs.define(frozen=True, kw_only=True)
class FloorFrictionRandomizer(Randomizer):
    """Randomizes the floor friction."""

    name: str = "geom_friction"
    scale_lower: float = attrs.field(default=0.4)
    scale_upper: float = attrs.field(default=1.0)
    floor_body_id: int = attrs.field(default=MISSING)

    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: Array) -> tuple[PhysicsModel, PhysicsData]:
        geom_friction = model.geom_friction.at[self.floor_body_id, 0].set(
            jax.random.uniform(rng, minval=self.scale_lower, maxval=self.scale_upper)
        )
        model = model.tree_replace({self.name: geom_friction})
        return model, data


@attrs.define(frozen=True, kw_only=True)
class ArmatureRandomizer(Randomizer):
    """Randomizes the armature."""

    name: str = "dof_armature"
    scale_lower: float = attrs.field(default=1.0)
    scale_upper: float = attrs.field(default=1.05)

    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: Array) -> tuple[PhysicsModel, PhysicsData]:
        armature = model.dof_armature * jax.random.uniform(
            rng, shape=(model.nq,), minval=self.scale_lower, maxval=self.scale_upper
        )
        dof_armature = model.dof_armature.at[:].set(armature)
        model = model.tree_replace({self.name: dof_armature})
        return model, data


@attrs.define(frozen=True, kw_only=True)
class LinkMassRandomizer(Randomizer):
    """Randomizes the link masses."""

    name: str = "body_mass"
    scale_lower: float = attrs.field(default=0.9)
    scale_upper: float = attrs.field(default=1.1)

    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: Array) -> tuple[PhysicsModel, PhysicsData]:
        dmass = jax.random.uniform(rng, shape=(model.nbody,), minval=self.scale_lower, maxval=self.scale_upper)
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)
        model = model.tree_replace({self.name: body_mass})
        return model, data


@attrs.define(frozen=True, kw_only=True)
class TorsoMassRandomizer(Randomizer):
    """Randomizes the torso mass."""

    name: str = "body_mass"
    scale_lower: float = attrs.field(default=-1.0)
    scale_upper: float = attrs.field(default=1.0)
    torso_body_id: int = attrs.field(default=MISSING)

    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: Array) -> tuple[PhysicsModel, PhysicsData]:
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=self.scale_lower, maxval=self.scale_upper)
        body_mass = model.body_mass.at[self.torso_body_id].set(model.body_mass[self.torso_body_id] + dmass)
        model = model.tree_replace({self.name: body_mass})
        return model, data


@attrs.define(frozen=True, kw_only=True)
class TorsoMassRandomizerBuilder(RandomizerBuilder[TorsoMassRandomizer]):
    scale_lower: float = attrs.field(default=-1.0)
    scale_upper: float = attrs.field(default=1.0)
    torso_body_name: str = attrs.field(default=MISSING)

    def __call__(self, model: PhysicsModel) -> TorsoMassRandomizer:
        torso_body_id = get_body_data_idx_by_name(model)[self.torso_body_name]
        return TorsoMassRandomizer(
            scale_lower=self.scale_lower,
            scale_upper=self.scale_upper,
            torso_body_id=torso_body_id,
        )


@attrs.define(frozen=True, kw_only=True)
class JointDampingRandomizer(Randomizer):
    """Randomizes the joint damping."""

    name: str = "dof_damping"
    scale_lower: float = attrs.field(default=0.9)
    scale_upper: float = attrs.field(default=1.1)

    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: Array) -> tuple[PhysicsModel, PhysicsData]:
        rng, key = jax.random.split(rng)
        kd = model.dof_damping * jax.random.uniform(
            key, shape=(model.nq,), minval=self.scale_lower, maxval=self.scale_upper
        )
        dof_damping = model.dof_damping.at[:].set(kd)
        model = model.tree_replace({self.name: dof_damping})
        return model, data
