"""Randomize each environment when gathering trajectories."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import attrs
import jax
from jaxtyping import PRNGKeyArray
from omegaconf import MISSING

from ksim.env.data import PhysicsData, PhysicsModel
from ksim.utils.mujoco import update_model_field
from ksim.utils.named_access import get_body_data_idx_by_name


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
        new_body_mass = model.body_mass * (jax.random.uniform(rng, model.body_mass.shape) * self.scale + 1.0)
        return update_model_field(model, "body_mass", new_body_mass)


T = TypeVar("T", bound=Randomization)


class RandomizerBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, physics_model: PhysicsModel, data: PhysicsData, rng: PRNGKeyArray) -> T:
        """Builds a randomizer from a MuJoCo model."""


@attrs.define(frozen=True, kw_only=True)
class StaticFrictionRandomization(Randomization):
    """Randomizes the static friction."""

    name: str = "dof_frictionloss"
    scale_lower: float = attrs.field(default=0.5)
    scale_upper: float = attrs.field(default=2.0)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss + jax.random.uniform(
            key,
            shape=(model.dof_frictionloss.shape[0],),
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        dof_frictionloss = model.dof_frictionloss.at[:].set(frictionloss)
        return update_model_field(model, self.name, dof_frictionloss)


@attrs.define(frozen=True, kw_only=True)
class FloorFrictionRandomization(Randomization):
    """Randomizes the floor friction."""

    name: str = "geom_friction"
    scale_lower: float = attrs.field(default=0.4)
    scale_upper: float = attrs.field(default=1.0)
    floor_body_id: int = attrs.field(default=MISSING)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        geom_friction = model.geom_friction.at[self.floor_body_id, 0].set(
            jax.random.uniform(rng, minval=self.scale_lower, maxval=self.scale_upper)
        )
        return update_model_field(model, self.name, geom_friction)


@attrs.define(frozen=True, kw_only=True)
class ArmatureRandomization(Randomization):
    """Randomizes the armature."""

    name: str = "dof_armature"
    scale_lower: float = attrs.field(default=1.0)
    scale_upper: float = attrs.field(default=1.05)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        armature = model.dof_armature * jax.random.uniform(
            rng, shape=(model.nq,), minval=self.scale_lower, maxval=self.scale_upper
        )
        dof_armature = model.dof_armature.at[:].set(armature)
        return update_model_field(model, self.name, dof_armature)


@attrs.define(frozen=True, kw_only=True)
class LinkMassRandomization(Randomization):
    """Randomizes the link masses."""

    name: str = "body_mass"
    scale_lower: float = attrs.field(default=0.9)
    scale_upper: float = attrs.field(default=1.1)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        dmass = jax.random.uniform(rng, shape=(model.nbody,), minval=self.scale_lower, maxval=self.scale_upper)
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)
        return update_model_field(model, self.name, body_mass)


@attrs.define(frozen=True, kw_only=True)
class TorsoMassRandomization(Randomization):
    """Randomizes the torso mass."""

    name: str = "body_mass"
    scale_lower: float = attrs.field(default=-1.0)
    scale_upper: float = attrs.field(default=1.0)
    torso_body_id: int = attrs.field(default=MISSING)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=self.scale_lower, maxval=self.scale_upper)
        body_mass = model.body_mass.at[self.torso_body_id].set(model.body_mass[self.torso_body_id] + dmass)
        return update_model_field(model, self.name, body_mass)


@attrs.define(frozen=True, kw_only=True)
class TorsoMassRandomizerBuilder(RandomizerBuilder[TorsoMassRandomization]):
    scale_lower: float = attrs.field(default=-1.0)
    scale_upper: float = attrs.field(default=1.0)
    torso_body_name: str = attrs.field(default=MISSING)

    def __call__(self, model: PhysicsModel) -> TorsoMassRandomization:
        torso_body_id = get_body_data_idx_by_name(model)[self.torso_body_name]
        return TorsoMassRandomization(
            scale_lower=self.scale_lower,
            scale_upper=self.scale_upper,
            torso_body_id=torso_body_id,
        )


@attrs.define(frozen=True, kw_only=True)
class JointDampingRandomization(Randomization):
    """Randomizes the joint damping."""

    name: str = "dof_damping"
    scale_lower: float = attrs.field(default=0.9)
    scale_upper: float = attrs.field(default=1.1)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        rng, key = jax.random.split(rng)
        kd = model.dof_damping * jax.random.uniform(
            key, shape=(model.nq,), minval=self.scale_lower, maxval=self.scale_upper
        )
        dof_damping = model.dof_damping.at[:].set(kd)
        return update_model_field(model, self.name, dof_damping)
