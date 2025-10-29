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
    "IMUAlignmentRandomizer",
    "COMRandomizer",
    "AllBodiesCOMRandomizer",
    "AllBodiesInertiaRandomizer",
    "CollisionBodyRandomizer",
]

import functools
import math
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
class IMUAlignmentRandomizer(PhysicsRandomizer):
    site_name: str = attrs.field(default="imu_site")
    tilt_std_rad: float = attrs.field(default=math.radians(2))  # 1σ for roll & pitch (2 deg in rad)
    yaw_std_rad: float | None = attrs.field(default=None)  # 1σ for yaw if desired
    translate_std_m: float | None = attrs.field(default=None)  # 1σ in metres

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


@attrs.define(frozen=True, kw_only=True)
class COMRandomizer(PhysicsRandomizer):
    """Randomizes the center of mass position (ipos) for a specific body in the bot."""

    body_id: int = attrs.field()
    scale: float = attrs.field(default=0.01)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        # Sample random offsets for x, y, z coordinates
        rng, sub = jax.random.split(rng)
        ipos_offset = jax.random.uniform(
            sub,
            shape=(3,),
            minval=-self.scale,
            maxval=self.scale,
        )

        # Create new ipos array with the offset added to the specified body
        new_ipos = jnp.concatenate(
            [
                model.body_ipos[: self.body_id],
                jnp.array([model.body_ipos[self.body_id] + ipos_offset]),
                model.body_ipos[self.body_id + 1 :],
            ]
        )
        return {"body_ipos": new_ipos}

    @classmethod
    def from_body_name(
        cls,
        model: PhysicsModel,
        body_name: str,
        scale: float = 0.01,
    ) -> Self:
        names_to_idxs = get_body_data_idx_by_name(model)
        if body_name not in names_to_idxs:
            raise ValueError(f"Body name {body_name} not found in model")
        body_id = names_to_idxs[body_name]
        return cls(
            body_id=body_id,
            scale=scale,
        )


@attrs.define(frozen=True, kw_only=True)
class AllBodiesCOMRandomizer(PhysicsRandomizer):
    """Randomizes the center of mass positions (ipos) for all bodies in the bot."""

    scale: float = attrs.field(default=0.01)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        # Sample random offsets for x, y, z coordinates for all bodies
        rng, sub = jax.random.split(rng)
        ipos_offsets = jax.random.uniform(
            sub,
            shape=(model.nbody, 3),
            minval=-self.scale,
            maxval=self.scale,
        )

        new_ipos = model.body_ipos + ipos_offsets
        return {"body_ipos": new_ipos}


@attrs.define(frozen=True, kw_only=True)
class AllBodiesInertiaRandomizer(PhysicsRandomizer):
    """Randomizes the inertia and mass for all bodies in the model.

    Does not update the inverse weights.
    """

    scale: float = attrs.field(default=0.02)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        rng, sub = jax.random.split(rng)
        mass_scaling = 1.0 + jax.random.uniform(sub, shape=(model.nbody,), minval=-self.scale, maxval=self.scale)
        inertia_perturbation = 1.0 + jax.random.uniform(
            sub, shape=(model.nbody, 3), minval=-self.scale, maxval=self.scale
        )

        new_mass = model.body_mass * mass_scaling
        new_inertia = (
            model.body_inertia * mass_scaling[:, None] * inertia_perturbation
        )  # scale with mass and change shape a bit

        updates = {
            "body_mass": new_mass,
            "body_inertia": new_inertia,
        }

        return updates


@attrs.define(frozen=True, kw_only=True)
class CollisionBodyRandomizer(PhysicsRandomizer):
    """Randomizes collision body capsule geometries.

    Jitters capsule positions independently in x, y, z and varies their radius and length.
    Useful for domain randomization of collision geometries to improve sim-to-real transfer.

    Note: When randomizing capsule length (half-length), the capsule grows/shrinks symmetrically
    from its center position along its longitudinal axis - both ends move equally.
    """

    geom_ids: tuple[int, ...] = attrs.field()
    radius_scale: float = attrs.field(default=0.05)  # scales from (1-scale) to (1+scale)
    length_scale: float = attrs.field(default=0.05)  # scales from (1-scale) to (1+scale)
    position_jitter_x: float = attrs.field(default=0.001)  # in meters, uniform range
    position_jitter_y: float = attrs.field(default=0.001)  # in meters, uniform range
    position_jitter_z: float = attrs.field(default=0.001)  # in meters, uniform range

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> dict[str, Array]:
        updates = {}

        # Randomize radius and length independently for capsules
        # For capsules: geom_size[geom_id, 0] = radius, geom_size[geom_id, 1] = half-length
        rng, radius_rng = jax.random.split(rng)
        radius_scales = jax.random.uniform(
            radius_rng,
            shape=(len(self.geom_ids),),
            minval=1.0 - self.radius_scale,
            maxval=1.0 + self.radius_scale,
        )

        rng, length_rng = jax.random.split(rng)
        length_scales = jax.random.uniform(
            length_rng,
            shape=(len(self.geom_ids),),
            minval=1.0 - self.length_scale,
            maxval=1.0 + self.length_scale,
        )

        new_geom_size = model.geom_size
        for i, geom_id in enumerate(self.geom_ids):
            # Scale radius (index 0) and half-length (index 1) independently
            new_radius = model.geom_size[geom_id, 0] * radius_scales[i]
            new_half_length = model.geom_size[geom_id, 1] * length_scales[i]
            new_geom_size = new_geom_size.at[geom_id, 0].set(new_radius)
            new_geom_size = new_geom_size.at[geom_id, 1].set(new_half_length)
        updates["geom_size"] = new_geom_size

        # Randomize position - jitter independently in x, y, z with uniform distribution
        rng, pos_rng = jax.random.split(rng)
        position_offsets = jax.random.uniform(
            pos_rng,
            shape=(len(self.geom_ids), 3),
            minval=jnp.array([-self.position_jitter_x, -self.position_jitter_y, -self.position_jitter_z]),
            maxval=jnp.array([self.position_jitter_x, self.position_jitter_y, self.position_jitter_z]),
        )

        new_geom_pos = model.geom_pos
        for i, geom_id in enumerate(self.geom_ids):
            new_geom_pos = new_geom_pos.at[geom_id].add(position_offsets[i])

        updates["geom_pos"] = new_geom_pos

        return updates

    @classmethod
    def from_geom_names(
        cls,
        model: PhysicsModel,
        geom_names: list[str] | tuple[str, ...],
        radius_scale: float = 0.05,
        length_scale: float = 0.05,
        position_jitter_x: float = 0.001,
        position_jitter_y: float = 0.001,
        position_jitter_z: float = 0.001,
    ) -> Self:
        """Create a CollisionBodyRandomizer from collision geom names.

        Args:
            model: The physics model
            geom_names: List or tuple of collision geom names to randomize
            radius_scale: Symmetric scaling range for radius, samples from (1-scale) to (1+scale) (default: 0.05)
            length_scale: Symmetric scaling range for length, samples from (1-scale) to (1+scale) (default: 0.05)
            position_jitter_x: Uniform jitter range in meters along x-axis, ±value (default: 0.001)
            position_jitter_y: Uniform jitter range in meters along y-axis, ±value (default: 0.001)
            position_jitter_z: Uniform jitter range in meters along z-axis, ±value (default: 0.001)

        Returns:
            CollisionBodyRandomizer instance

        Raises:
            ValueError: If any geom name is not found in the model
        """
        names_to_idxs = get_geom_data_idx_by_name(model)
        geom_ids = []
        for geom_name in geom_names:
            if geom_name not in names_to_idxs:
                available_geoms = list(names_to_idxs.keys())
                raise ValueError(f"Geom name '{geom_name}' not found in model. Available geoms: {available_geoms}")
            geom_ids.append(names_to_idxs[geom_name])

        return cls(
            geom_ids=tuple(geom_ids),
            radius_scale=radius_scale,
            length_scale=length_scale,
            position_jitter_x=position_jitter_x,
            position_jitter_y=position_jitter_y,
            position_jitter_z=position_jitter_z,
        )
