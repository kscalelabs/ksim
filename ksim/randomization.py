"""Randomize each environment when gathering trajectories."""

__all__ = [
    "Randomization",
    "WeightRandomization",
    "StaticFrictionRandomization",
    "FloorFrictionRandomization",
    "ArmatureRandomization",
    "TorsoMassRandomization",
    "JointDampingRandomization",
    "JointZeroPositionRandomization",
]

import functools
from abc import ABC, abstractmethod
from typing import Collection, Self

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.types import PhysicsModel
from ksim.utils.mujoco import get_body_data_idx_by_name, slice_update
from ksim.vis import Marker


@attrs.define(frozen=True, kw_only=True)
class Randomization(ABC):
    """Randomize the joint positions of the robot."""

    @abstractmethod
    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        """Randomize the model for a single environment."""

    def get_markers(self) -> Collection[Marker]:
        return []

    def get_name(self) -> str:
        """Get the name of the observation."""
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def randomization_name(self) -> str:
        return self.get_name()


@attrs.define(frozen=True, kw_only=True)
class WeightRandomization(Randomization):
    """Randomize the body masses of the robot."""

    scale: float = attrs.field()

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        return {"body_mass": model.body_mass * (jax.random.uniform(rng, model.body_mass.shape) * self.scale + 1.0)}


@attrs.define(frozen=True, kw_only=True)
class StaticFrictionRandomization(Randomization):
    """Randomizes the static friction."""

    scale_lower: float = attrs.field(default=0.5)
    scale_upper: float = attrs.field(default=2.0)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
            rng,
            shape=(model.dof_frictionloss.shape[0] - 6,),
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        # Skip the first 6 DOFs (free joint)
        new_frictionloss = jnp.concatenate([model.dof_frictionloss[:6], frictionloss])
        return {"dof_frictionloss": new_frictionloss}


@attrs.define(frozen=True, kw_only=True)
class FloorFrictionRandomization(Randomization):
    """Randomizes the floor friction."""

    floor_body_id: int = attrs.field()
    scale_lower: float = attrs.field(default=0.4)
    scale_upper: float = attrs.field(default=1.0)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        arr_inds = (self.floor_body_id, 0)
        rand_vals = jax.random.uniform(rng, minval=self.scale_lower, maxval=self.scale_upper)
        new_geom_friction = slice_update(model, "geom_friction", arr_inds, rand_vals)
        return {"geom_friction": new_geom_friction}

    @classmethod
    def from_body_name(
        cls,
        model: PhysicsModel,
        floor_body_name: str,
        scale_lower: float = 0.4,
        scale_upper: float = 1.0,
    ) -> Self:
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

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        # Skip the first 6 DOFs (free joint)
        armature = model.dof_armature[6:] * jax.random.uniform(
            rng, shape=(model.dof_armature.shape[0] - 6,), minval=self.scale_lower, maxval=self.scale_upper
        )
        new_armature = jnp.concatenate([model.dof_armature[:6], armature])
        return {"dof_armature": new_armature}


@attrs.define(frozen=True, kw_only=True)
class TorsoMassRandomization(Randomization):
    """Randomizes the torso mass."""

    torso_body_id: int = attrs.field()
    scale_lower: float = attrs.field(default=-1.0)
    scale_upper: float = attrs.field(default=1.0)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        new_mass = model.body_mass[self.torso_body_id] + jax.random.uniform(
            rng, minval=self.scale_lower, maxval=self.scale_upper
        )
        new_body_mass = jnp.concatenate(
            [
                model.body_mass[: self.torso_body_id],
                jnp.array([new_mass]),
                model.body_mass[self.torso_body_id + 1 :],
            ]
        )
        return {"body_mass": new_body_mass}

    @classmethod
    def from_body_name(
        cls,
        model: PhysicsModel,
        torso_body_name: str,
        scale_lower: float = 0.0,
        scale_upper: float = 1.0,
    ) -> Self:
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

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        # Skip the first 6 DOFs (free joint)
        kd = model.dof_damping[6:] * jax.random.uniform(
            rng,
            shape=(model.dof_damping.shape[0] - 6,),
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        dof_damping = jnp.concatenate([model.dof_damping[:6], kd])
        return {"dof_damping": dof_damping}


@attrs.define(frozen=True, kw_only=True)
class JointZeroPositionRandomization(Randomization):
    """Randomizes the joint zero position."""

    scale_lower: float = attrs.field(default=-0.1)
    scale_upper: float = attrs.field(default=0.1)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        qpos_0 = model.qpos0
        new_qpos = jax.random.uniform(
            rng,
            shape=(model.qpos0.shape[0] - 7,),
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        # Skip the first 7 DOFs (free joint - xyz + quat)
        qpos_0 = slice_update(model, "qpos0", slice(7, None), new_qpos)
        return {"qpos0": qpos_0}
