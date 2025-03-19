"""Randomize each environment when gathering trajectories."""

from abc import ABC, abstractmethod

import attrs
import jax
from jaxtyping import PRNGKeyArray

from ksim.env.data import PhysicsModel
from ksim.utils.mujoco import get_body_data_idx_by_name, update_model_field


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


@attrs.define(frozen=True, kw_only=True)
class StaticFrictionRandomization(Randomization):
    """Randomizes the static friction."""

    scale_lower: float = attrs.field(default=0.5)
    scale_upper: float = attrs.field(default=2.0)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss * jax.random.uniform(
            key,
            shape=model.dof_frictionloss.shape,
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        dof_frictionloss = model.dof_frictionloss.at[:].set(frictionloss)
        return update_model_field(model, "dof_frictionloss", dof_frictionloss)


@attrs.define(frozen=True, kw_only=True)
class FloorFrictionRandomization(Randomization):
    """Randomizes the floor friction."""

    floor_body_id: int = attrs.field()
    scale_lower: float = attrs.field(default=0.4)
    scale_upper: float = attrs.field(default=1.0)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        new_geom_friction = jax.random.uniform(rng, minval=self.scale_lower, maxval=self.scale_upper)
        geom_friction = model.geom_friction.at[self.floor_body_id, 0].set(new_geom_friction)
        return update_model_field(model, "geom_friction", geom_friction)

    @classmethod
    def from_body_name(
        cls,
        model: PhysicsModel,
        floor_body_name: str,
        scale_lower: float = 0.4,
        scale_upper: float = 1.0,
    ) -> "FloorFrictionRandomization":
        names_to_idxs = get_body_data_idx_by_name(model)
        if floor_body_name not in names_to_idxs:
            raise ValueError(f"Body name {floor_body_name} not found in model")
        floor_body_id = names_to_idxs[floor_body_name]
        return cls(
            floor_body_id=floor_body_id,
            scale_lower=scale_lower,
            scale_upper=scale_upper,
        )


@attrs.define(frozen=True, kw_only=True)
class ArmatureRandomization(Randomization):
    """Randomizes the armature."""

    scale_lower: float = attrs.field(default=1.0)
    scale_upper: float = attrs.field(default=1.05)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        armature = model.dof_armature * jax.random.uniform(
            rng,
            shape=model.dof_armature.shape,
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        dof_armature = model.dof_armature.at[:].set(armature)
        return update_model_field(model, "dof_armature", dof_armature)


@attrs.define(frozen=True, kw_only=True)
class LinkMassRandomization(Randomization):
    """Randomizes the link masses."""

    scale_lower: float = attrs.field(default=0.9)
    scale_upper: float = attrs.field(default=1.1)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        dmass = jax.random.uniform(
            rng,
            shape=model.body_mass.shape,
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        body_mass = model.body_mass.at[:].set(model.body_mass * dmass)
        return update_model_field(model, "body_mass", body_mass)


@attrs.define(frozen=True, kw_only=True)
class TorsoMassRandomization(Randomization):
    """Randomizes the torso mass."""

    torso_body_id: int = attrs.field()
    scale_lower: float = attrs.field(default=0.0)
    scale_upper: float = attrs.field(default=1.0)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=self.scale_lower, maxval=self.scale_upper)
        body_mass = model.body_mass.at[self.torso_body_id].set(model.body_mass[self.torso_body_id] + dmass)
        return update_model_field(model, "body_mass", body_mass)

    @classmethod
    def from_body_name(
        cls,
        model: PhysicsModel,
        torso_body_name: str,
        scale_lower: float = 0.0,
        scale_upper: float = 1.0,
    ) -> "TorsoMassRandomization":
        names_to_idxs = get_body_data_idx_by_name(model)
        if torso_body_name not in names_to_idxs:
            raise ValueError(f"Body name {torso_body_name} not found in model")
        torso_body_id = names_to_idxs[torso_body_name]
        return cls(
            torso_body_id=torso_body_id,
            scale_lower=scale_lower,
            scale_upper=scale_upper,
        )


@attrs.define(frozen=True, kw_only=True)
class JointDampingRandomization(Randomization):
    """Randomizes the joint damping."""

    scale_lower: float = attrs.field(default=0.9)
    scale_upper: float = attrs.field(default=1.1)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        rng, key = jax.random.split(rng)
        kd = model.dof_damping * jax.random.uniform(
            key,
            shape=model.dof_damping.shape,
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        dof_damping = model.dof_damping.at[:].set(kd)
        return update_model_field(model, "dof_damping", dof_damping)
