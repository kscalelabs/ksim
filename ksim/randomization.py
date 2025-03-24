"""Randomize each environment when gathering trajectories."""

__all__ = [
    "Randomization",
    "WeightRandomization",
    "StaticFrictionRandomization",
    "FloorFrictionRandomization",
    "ArmatureRandomization",
    "TorsoMassRandomization",
    "JointDampingRandomization",
]

from abc import ABC, abstractmethod

import attrs
import jax
import jax.numpy as jnp
import mujoco
from jaxtyping import PRNGKeyArray
from mujoco import mjx

from ksim.types import PhysicsModel
from ksim.utils.mujoco import get_body_data_idx_by_name, update_model_field


@attrs.define(frozen=True, kw_only=True)
class Randomization(ABC):
    """Randomize the joint positions of the robot."""

    @abstractmethod
    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        """Randomize the model for a single environment."""


@attrs.define(frozen=True, kw_only=True)
class WeightRandomization(Randomization):
    """Randomize the body masses of the robot."""

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
        """Randomize the static friction of the robot."""
        rng, key = jax.random.split(rng)
        frictionloss = model.dof_frictionloss[6:] + jax.random.uniform(
            key,
            shape=(model.dof_frictionloss.shape[0] - 6,),
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        # Skip the first 6 DOFs (free joint)
        new_frictionloss = jnp.concatenate([model.dof_frictionloss[:6], frictionloss])
        return update_model_field(model, "dof_frictionloss", new_frictionloss)


@attrs.define(frozen=True, kw_only=True)
class FloorFrictionRandomization(Randomization):
    """Randomizes the floor friction."""

    floor_body_id: int = attrs.field()
    scale_lower: float = attrs.field(default=0.4)
    scale_upper: float = attrs.field(default=1.0)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        """Randomize the floor friction of the robot."""
        match type(model):
            case mujoco.MjModel:
                new_geom_friction = model.geom_friction.copy()
                new_geom_friction[self.floor_body_id, 0] = jax.random.uniform(
                    rng, minval=self.scale_lower, maxval=self.scale_upper
                )
            case mjx.Model:
                new_geom_friction = model.geom_friction.at[self.floor_body_id, 0].set(
                    jax.random.uniform(rng, minval=self.scale_lower, maxval=self.scale_upper)
                )
        return update_model_field(model, "geom_friction", new_geom_friction)

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
        """Randomize the armature of the robot."""
        # Skip the first 6 DOFs (free joint)
        armature = model.dof_armature[6:] * jax.random.uniform(
            rng, shape=(model.dof_armature.shape[0] - 6,), minval=self.scale_lower, maxval=self.scale_upper
        )
        new_armature = jnp.concatenate([model.dof_armature[:6], armature])
        return update_model_field(model, "dof_armature", new_armature)


@attrs.define(frozen=True, kw_only=True)
class TorsoMassRandomization(Randomization):
    """Randomizes the torso mass."""

    torso_body_id: int = attrs.field()
    scale_lower: float = attrs.field(default=-1.0)
    scale_upper: float = attrs.field(default=1.0)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        """Randomize the torso mass of the robot."""
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=self.scale_lower, maxval=self.scale_upper)
        new_body_mass = jnp.concatenate(
            [
                model.body_mass[: self.torso_body_id],
                jnp.array([dmass]),
                model.body_mass[self.torso_body_id + 1 :],
            ]
        )
        return update_model_field(model, "body_mass", new_body_mass)

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
        # Skip the first 6 DOFs (free joint)
        kd = model.dof_damping[6:] * jax.random.uniform(
            key, shape=(model.dof_damping.shape[0] - 6,), minval=self.scale_lower, maxval=self.scale_upper
        )
        dof_damping = jnp.concatenate([model.dof_damping[:6], kd])

        return update_model_field(model, "dof_damping", dof_damping)
