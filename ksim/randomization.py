"""Randomize each environment when gathering trajectories."""

__all__ = [
    "PhysicsRandomizer",
    "StaticFrictionRandomizer",
    "FloorFrictionRandomizer",
    "ArmatureRandomizer",
    "MassAdditionRandomizer",
    "MassMultiplicationRandomizer",
    "AllBodiesMassMultiplicationRandomizer",
    "JointDampingRandomizer",
    "JointZeroPositionRandomizer",
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
from ksim.utils.mujoco import get_body_data_idx_by_name, get_geom_data_idx_by_name, slice_update
from ksim.vis import Marker


@attrs.define(frozen=True, kw_only=True)
class PhysicsRandomizer(ABC):
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
class StaticFrictionRandomizer(PhysicsRandomizer):
    """Randomizes the static friction."""

    scale_lower: float = attrs.field(default=0.5)
    scale_upper: float = attrs.field(default=2.0)

    freejoint_first: bool = attrs.field(default=True)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        if self.freejoint_first:
            frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
                rng,
                shape=(model.dof_frictionloss.shape[0] - 6,),
                minval=self.scale_lower,
                maxval=self.scale_upper,
            )
            new_frictionloss = jnp.concatenate([model.dof_frictionloss[:6], frictionloss])
        else:
            new_frictionloss = model.dof_frictionloss * jax.random.uniform(
                rng,
                shape=(model.dof_frictionloss.shape[0],),
                minval=self.scale_lower,
                maxval=self.scale_upper,
            )
        return {"dof_frictionloss": new_frictionloss}


@attrs.define(frozen=True, kw_only=True)
class FloorFrictionRandomizer(PhysicsRandomizer):
    """Randomizes the floor friction."""

    floor_geom_id: int = attrs.field()
    scale_lower: float = attrs.field(default=0.4)
    scale_upper: float = attrs.field(default=1.0)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        arr_inds = (self.floor_geom_id, 0)
        rand_vals = jax.random.uniform(rng, minval=self.scale_lower, maxval=self.scale_upper)
        new_geom_friction = slice_update(model, "geom_friction", arr_inds, rand_vals)
        return {"geom_friction": new_geom_friction}

    @classmethod
    def from_geom_name(
        cls,
        model: PhysicsModel,
        floor_geom_name: str,
        scale_lower: float = 0.4,
        scale_upper: float = 1.0,
    ) -> Self:
        names_to_idxs = get_geom_data_idx_by_name(model)
        if floor_geom_name not in names_to_idxs:
            raise ValueError(f"Geom name {floor_geom_name} not found in model. Choices are {names_to_idxs.keys()}")
        floor_geom_id = names_to_idxs[floor_geom_name]
        return cls(
            floor_geom_id=floor_geom_id,
            scale_lower=scale_lower,
            scale_upper=scale_upper,
        )


@attrs.define(frozen=True, kw_only=True)
class ArmatureRandomizer(PhysicsRandomizer):
    """Randomizes the armature."""

    scale_lower: float = attrs.field(default=0.95)
    scale_upper: float = attrs.field(default=1.05)

    freejoint_first: bool = attrs.field(default=True)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        # Skip the first 6 DOFs (free joint)
        if self.freejoint_first:
            armature = model.dof_armature[6:] * jax.random.uniform(
                rng,
                shape=(model.dof_armature.shape[0] - 6,),
                minval=self.scale_lower,
                maxval=self.scale_upper,
            )
            new_armature = jnp.concatenate([model.dof_armature[:6], armature])
        else:
            new_armature = model.dof_armature * jax.random.uniform(
                rng,
                shape=(model.dof_armature.shape[0],),
                minval=self.scale_lower,
                maxval=self.scale_upper,
            )
        return {"dof_armature": new_armature}


@attrs.define(frozen=True, kw_only=True)
class MassAdditionRandomizer(PhysicsRandomizer):
    """Randomizes the mass of some body."""

    body_id: int = attrs.field()
    scale_lower: float = attrs.field(default=-1.0)
    scale_upper: float = attrs.field(default=1.0)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        new_mass = model.body_mass[self.body_id] + jax.random.uniform(
            rng, minval=self.scale_lower, maxval=self.scale_upper
        )
        new_body_mass = jnp.concatenate(
            [
                model.body_mass[: self.body_id],
                jnp.array([new_mass]),
                model.body_mass[self.body_id + 1 :],
            ]
        )
        return {"body_mass": new_body_mass}

    @classmethod
    def from_body_name(
        cls,
        model: PhysicsModel,
        body_name: str,
        scale_lower: float = 0.0,
        scale_upper: float = 1.0,
    ) -> Self:
        names_to_idxs = get_body_data_idx_by_name(model)
        if body_name not in names_to_idxs:
            raise ValueError(f"Body name {body_name} not found in model")
        body_id = names_to_idxs[body_name]
        return cls(
            body_id=body_id,
            scale_lower=scale_lower,
            scale_upper=scale_upper,
        )


@attrs.define(frozen=True, kw_only=True)
class MassMultiplicationRandomizer(PhysicsRandomizer):
    """Randomizes the mass of some body."""

    body_id: int = attrs.field()
    scale_lower: float = attrs.field(default=0.95)
    scale_upper: float = attrs.field(default=1.05)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        new_mass = model.body_mass[self.body_id] * jax.random.uniform(
            rng,
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        new_body_mass = jnp.concatenate(
            [
                model.body_mass[: self.body_id],
                jnp.array([new_mass]),
                model.body_mass[self.body_id + 1 :],
            ]
        )
        return {"body_mass": new_body_mass}

    @classmethod
    def from_body_name(
        cls,
        model: PhysicsModel,
        body_name: str,
        scale_lower: float = 0.0,
        scale_upper: float = 1.0,
    ) -> Self:
        names_to_idxs = get_body_data_idx_by_name(model)
        if body_name not in names_to_idxs:
            raise ValueError(f"Body name {body_name} not found in model")
        body_id = names_to_idxs[body_name]
        return cls(
            body_id=body_id,
            scale_lower=scale_lower,
            scale_upper=scale_upper,
        )


@attrs.define(frozen=True, kw_only=True)
class AllBodiesMassMultiplicationRandomizer(PhysicsRandomizer):
    """Randomizes the mass of all bodies."""

    scale_lower: float = attrs.field(default=0.98)
    scale_upper: float = attrs.field(default=1.02)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        random_mass = jax.random.uniform(
            rng,
            shape=(model.nbody,),
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        new_body_mass = model.body_mass * random_mass
        return {"body_mass": new_body_mass}


@attrs.define(frozen=True, kw_only=True)
class JointDampingRandomizer(PhysicsRandomizer):
    """Randomizes the joint damping."""

    scale_lower: float = attrs.field(default=0.9)
    scale_upper: float = attrs.field(default=1.1)

    freejoint_first: bool = attrs.field(default=True)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        # Skip the first 6 DOFs (free joint)
        if self.freejoint_first:
            kd = model.dof_damping[6:] * jax.random.uniform(
                rng,
                shape=(model.dof_damping.shape[0] - 6,),
                minval=self.scale_lower,
                maxval=self.scale_upper,
            )
            dof_damping = jnp.concatenate([model.dof_damping[:6], kd])
        else:
            dof_damping = model.dof_damping * jax.random.uniform(
                rng,
                shape=(model.dof_damping.shape[0],),
                minval=self.scale_lower,
                maxval=self.scale_upper,
            )
        return {"dof_damping": dof_damping}


@attrs.define(frozen=True, kw_only=True)
class JointZeroPositionRandomizer(PhysicsRandomizer):
    """Randomizes the joint zero position."""

    scale_lower: float = attrs.field(default=-0.01)
    scale_upper: float = attrs.field(default=0.01)

    freejoint_first: bool = attrs.field(default=True)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        if self.freejoint_first:
            new_qpos = model.qpos0[7:] + jax.random.uniform(
                rng,
                shape=(model.qpos0.shape[0] - 7,),
                minval=self.scale_lower,
                maxval=self.scale_upper,
            )
            new_qpos = jnp.concatenate([model.qpos0[:7], new_qpos])
        else:
            new_qpos = model.qpos0 + jax.random.uniform(
                rng,
                shape=(model.qpos0.shape[0],),
                minval=self.scale_lower,
                maxval=self.scale_upper,
            )
        return {"qpos0": new_qpos}
