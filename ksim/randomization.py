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
    "IMUAlignmentRandomizer",
]

import functools
from abc import ABC, abstractmethod
from typing import Self

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.types import PhysicsModel
from ksim.utils.mujoco import (
    get_body_data_idx_by_name,
    get_geom_data_idx_by_name,
    get_site_data_idx_by_name,
    slice_update,
)


@attrs.define(frozen=True, kw_only=True)
class PhysicsRandomizer(ABC):
    """Randomize the joint positions of the robot."""

    @abstractmethod
    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        """Randomize the model for a single environment."""

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

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        frictionloss = model.dof_frictionloss[6:] * jax.random.uniform(
            rng,
            shape=(model.dof_frictionloss.shape[0] - 6,),
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        new_frictionloss = jnp.concatenate([model.dof_frictionloss[:6], frictionloss])
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

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        armature = model.dof_armature[6:] * jax.random.uniform(
            rng,
            shape=(model.dof_armature.shape[0] - 6,),
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        new_armature = jnp.concatenate([model.dof_armature[:6], armature])
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
class JointZeroPositionRandomizer(PhysicsRandomizer):
    """Randomizes the joint zero position."""

    scale_lower: float = attrs.field(default=-0.01)
    scale_upper: float = attrs.field(default=0.01)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        new_qpos = model.qpos0[7:] + jax.random.uniform(
            rng,
            shape=(model.qpos0.shape[0] - 7,),
            minval=self.scale_lower,
            maxval=self.scale_upper,
        )
        new_qpos = jnp.concatenate([model.qpos0[:7], new_qpos])
        return {"qpos0": new_qpos}


@attrs.define(frozen=True, kw_only=True)
class IMUAlignmentRandomizer(PhysicsRandomizer):
    site_name: str = "imu_site"
    tilt_std_rad: float = 0.0349066  # 1σ for roll & pitch (2 deg in rad)
    yaw_std_rad: float | None = None  # 1σ for yaw if desired
    translate_std_m: float | None = None  # 1σ in metres

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        # sample small rotations from normal distribution
        rng, sub = jax.random.split(rng)
        rx, ry = jax.random.normal(sub, (2,)) * self.tilt_std_rad

        rz = jnp.array(0.0)
        if self.yaw_std_rad is not None:
            rng, sub = jax.random.split(rng)
            rz = jax.random.normal(sub) * self.yaw_std_rad

        # Small angle approximation: q ≈ [1, θx/2, θy/2, θz/2].
        h = jnp.array(0.5)
        qx = jnp.array([1.0, rx * h, 0.0, 0.0])
        qy = jnp.array([1.0, 0.0, ry * h, 0.0])
        qz = jnp.array([1.0, 0.0, 0.0, rz * h])

        # Compose rotations: Z(yaw) * Y(pitch) * X(roll).
        # This applies roll first in body frame, then pitch, then yaw.
        q_offset = xax.quat_mul(qz, xax.quat_mul(qy, qx))
        q_offset = q_offset / jnp.linalg.norm(q_offset)

        # Apply rotation offset to IMU site.
        site_id = get_site_data_idx_by_name(model)[self.site_name]
        new_site_quat = model.site_quat.at[site_id].set(xax.quat_mul(q_offset, model.site_quat[site_id]))
        updates = {"site_quat": new_site_quat}

        # Translation noise.
        if self.translate_std_m is not None and self.translate_std_m > 0.0:
            rng, sub = jax.random.split(rng)
            dpos = jax.random.normal(sub, (3,)) * self.translate_std_m
            updates["site_pos"] = model.site_pos.at[site_id].add(dpos)

        return updates
