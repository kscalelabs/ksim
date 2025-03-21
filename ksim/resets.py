"""Defines the base reset classes and builders."""

import functools
import logging
from abc import ABC, abstractmethod

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import PRNGKeyArray
from mujoco import mjx

from ksim.env.data import PhysicsData, PhysicsModel
from ksim.utils.mujoco import update_data_field

logger = logging.getLogger(__name__)


@attrs.define(frozen=True, kw_only=True)
class Reset(ABC):
    """Base class for resets."""

    @abstractmethod
    def __call__(self, data: mjx.Data, rng: PRNGKeyArray) -> mjx.Data:
        """Resets the environment."""

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def reset_name(self) -> str:
        return self.get_name()


@attrs.define(frozen=True, kw_only=True)
class HFieldXYPositionReset(Reset):
    """Resets the robot's XY position within a bounded region, with height from aheightfield."""

    bounds: tuple[float, float, float, float]
    padded_bounds: tuple[float, float, float, float]
    x_range: float = attrs.field(default=5.0)
    y_range: float = attrs.field(default=5.0)
    hfield_data: jnp.ndarray
    robot_base_height: float = attrs.field(default=0.0)

    def __call__(self, data: mjx.Data, rng: PRNGKeyArray) -> mjx.Data:
        x_bound, y_bound, z_top, _ = self.bounds

        # Unpack padded bounds.
        lower_x, upper_x, lower_y, upper_y = self.padded_bounds

        keyx, keyy = jax.random.split(rng)
        offset_x = jax.random.uniform(keyx, (1,), minval=-self.x_range, maxval=self.x_range)
        offset_y = jax.random.uniform(keyy, (1,), minval=-self.y_range, maxval=self.y_range)

        new_x = jnp.clip(offset_x, lower_x, upper_x)
        new_y = jnp.clip(offset_y, lower_y, upper_y)

        qpos_j = data.qpos
        qpos_j = qpos_j.at[0:1].set(new_x)
        qpos_j = qpos_j.at[1:2].set(new_y)

        # Map new XY to heightfield indices.
        nx, ny = self.hfield_data.shape
        x_idx = jnp.clip(
            (((new_x.squeeze() + x_bound) / (2 * x_bound)) * (nx - 1)).astype(jnp.int32),
            0,
            nx - 1,
        )
        y_idx = jnp.clip(
            (((new_y.squeeze() + y_bound) / (2 * y_bound)) * (ny - 1)).astype(jnp.int32),
            0,
            ny - 1,
        )

        # Get the height from the heightfield and add the z offset.
        z = self.hfield_data[x_idx, y_idx]
        qpos_j = qpos_j.at[2:3].set(z + z_top + self.robot_base_height)
        data = update_data_field(data, "qpos", qpos_j)
        return data

    def __hash__(self) -> int:
        array_bytes = self.hfield_data.tobytes()
        return hash(
            (
                self.bounds,
                self.padded_bounds,
                self.x_range,
                self.y_range,
                self.hfield_data.shape,
                array_bytes,
            )
        )


@attrs.define(frozen=True, kw_only=True)
class PlaneXYPositionReset(Reset):
    """Resets the robot's XY position and sets Z to a fixed plane height."""

    bounds: tuple[float, float, float]
    padded_bounds: tuple[float, float, float, float]
    x_range: float = attrs.field(default=5.0)
    y_range: float = attrs.field(default=5.0)
    robot_base_height: float = attrs.field(default=0.0)

    def __call__(self, data: mjx.Data, rng: PRNGKeyArray) -> mjx.Data:
        _, _, z_pos = self.bounds

        lower_x, upper_x, lower_y, upper_y = self.padded_bounds

        keyx, keyy = jax.random.split(rng)
        offset_x = jax.random.uniform(keyx, (1,), minval=-self.x_range, maxval=self.x_range)
        offset_y = jax.random.uniform(keyy, (1,), minval=-self.y_range, maxval=self.y_range)

        new_x = jnp.clip(offset_x, lower_x, upper_x)
        new_y = jnp.clip(offset_y, lower_y, upper_y)

        qpos_j = data.qpos
        qpos_j = qpos_j.at[0:1].set(new_x)
        qpos_j = qpos_j.at[1:2].set(new_y)
        qpos_j = qpos_j.at[2:3].set(z_pos + self.robot_base_height)
        data = update_data_field(data, "qpos", qpos_j)
        return data


@attrs.define(frozen=True, kw_only=True)
class RandomJointPositionReset(Reset):
    """Resets the joint positions of the robot to random values."""

    scale: float = attrs.field(default=0.01)

    def __call__(self, data: PhysicsData, rng: PRNGKeyArray) -> PhysicsData:
        qpos = data.qpos
        qpos = qpos + jax.random.uniform(rng, qpos.shape, minval=-self.scale, maxval=self.scale)
        data = update_data_field(data, "qpos", qpos)
        return data


@attrs.define(frozen=True, kw_only=True)
class RandomJointVelocityReset(Reset):
    """Resets the joint velocities of the robot to random values."""

    scale: float = attrs.field(default=0.01)

    def __call__(self, data: PhysicsData, rng: PRNGKeyArray) -> PhysicsData:
        qvel = data.qvel
        qvel = qvel + jax.random.uniform(rng, qvel.shape, minval=-self.scale, maxval=self.scale)
        data = update_data_field(data, "qvel", qvel)
        return data


def get_xy_position_reset(
    physics_model: PhysicsModel,
    x_range: float = 5.0,
    y_range: float = 5.0,
    x_edge_padding: float = 5.0,
    y_edge_padding: float = 5.0,
    robot_base_height: float = 0.0,
) -> Reset:
    """Returns an XY position reset based on whether the config has a heightfield or plane floor.

    Args:
        physics_model: The MuJoCo model to use.
        x_range: The range of the XY position reset.
        y_range: The range of the XY position reset.
        x_edge_padding: The padding of the XY position reset.
        y_edge_padding: The padding of the XY position reset.
        robot_base_height: The height of the robot's base.
    """

    def compute_padded_bounds(
        x_bound: float, y_bound: float, x_edge_padding: float, y_edge_padding: float
    ) -> tuple[float, float, float, float]:
        """Helper to compute effective (padded) bounds from based on edge padding values."""
        lower_x = -x_bound + x_edge_padding
        upper_x = x_bound - x_edge_padding
        lower_y = -y_bound + y_edge_padding
        upper_y = y_bound - y_edge_padding
        return (lower_x, upper_x, lower_y, upper_y)

    # Return HFieldXYPositionReset if there is a heightfield.
    if hasattr(physics_model, "hfield_size") and physics_model.hfield_size.size > 0:
        x_bound, y_bound, z_top, z_bottom = physics_model.hfield_size.flatten().tolist()
        nx = (
            int(physics_model.hfield_nrow.item())
            if hasattr(physics_model.hfield_nrow, "item")
            else int(physics_model.hfield_nrow)
        )
        ny = (
            int(physics_model.hfield_ncol.item())
            if hasattr(physics_model.hfield_ncol, "item")
            else int(physics_model.hfield_ncol)
        )
        hfield_data = physics_model.hfield_data.reshape(nx, ny)
        logger.info("Using heightfield based floor with shape: %s", hfield_data.shape)
        padded_bounds = compute_padded_bounds(x_bound, y_bound, x_edge_padding, y_edge_padding)
        return HFieldXYPositionReset(
            bounds=(x_bound, y_bound, z_top, z_bottom),
            padded_bounds=padded_bounds,
            x_range=x_range,
            y_range=y_range,
            hfield_data=hfield_data,
            robot_base_height=robot_base_height,
        )

    # Return PlaneXYPositionReset if there is a plane floor.
    else:
        plane_indices = [i for i, t in enumerate(physics_model.geom_type) if t == 0]
        if not plane_indices:
            raise ValueError("No heightfield or plane geom found in the model. MuJoCo scene missing floor!")
        floor_idx = plane_indices[0]
        x_bound, y_bound = 5.0, 5.0
        z_pos = physics_model.geom_pos[floor_idx][2]
        logger.info("Using plane based floor with bounds: %s, %s, %s", x_bound, y_bound, z_pos)
        padded_bounds = compute_padded_bounds(x_bound, y_bound, x_edge_padding, y_edge_padding)
        return PlaneXYPositionReset(
            bounds=(x_bound, y_bound, z_pos),
            padded_bounds=padded_bounds,
            x_range=x_range,
            y_range=y_range,
            robot_base_height=robot_base_height,
        )


# This should be here but applied to the model
@attrs.define(frozen=True, kw_only=True)
class WeightRandomization(Randomization):
    """Randomize the body masses of the robot."""

    scale: float = attrs.field()

    def initial_randomization(self, rng: PRNGKeyArray) -> tuple[PhysicsModel, PhysicsData]:
        """Randomize the model for a single environment."""

    def __call__(
        self, randomization_state: FrozenDict[str, Array], model: PhysicsModel, data: PhysicsData, rng: PRNGKeyArray
    ) -> tuple[PhysicsModel, PhysicsData]:
        """Randomize the model for a single environment."""
        new_body_mass = model.body_mass * (jax.random.uniform(rng, model.body_mass.shape) * self.scale + 1.0)
        return randomization_state, model, update_data_field(data, "body_mass", new_body_mass)


@attrs.define(frozen=True, kw_only=True)
class StaticFrictionRandomization(Randomization):
    """Randomizes the static friction."""

    scale_lower: float = attrs.field(default=0.5)
    scale_upper: float = attrs.field(default=2.0)

    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: PRNGKeyArray) -> PhysicsModel:
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
        return update_model_field(model, "dof_frictionloss", new_frictionloss), data


@attrs.define(frozen=True, kw_only=True)
class FloorFrictionRandomization(Randomization):
    """Randomizes the floor friction."""

    floor_body_id: int = attrs.field()
    scale_lower: float = attrs.field(default=0.4)
    scale_upper: float = attrs.field(default=1.0)

    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: PRNGKeyArray) -> PhysicsModel:
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
        return update_model_field(model, "geom_friction", new_geom_friction), data

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

    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: PRNGKeyArray) -> PhysicsModel:
        """Randomize the armature of the robot."""
        # Skip the first 6 DOFs (free joint)
        armature = model.dof_armature[6:] * jax.random.uniform(
            rng, shape=(model.dof_armature.shape[0] - 6,), minval=self.scale_lower, maxval=self.scale_upper
        )
        new_armature = jnp.concatenate([model.dof_armature[:6], armature])
        return update_model_field(model, "dof_armature", new_armature), data


@attrs.define(frozen=True, kw_only=True)
class TorsoMassRandomization(Randomization):
    """Randomizes the torso mass."""

    torso_body_id: int = attrs.field()
    scale_lower: float = attrs.field(default=-1.0)
    scale_upper: float = attrs.field(default=1.0)

    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: PRNGKeyArray) -> PhysicsModel:
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
        return update_model_field(model, "body_mass", new_body_mass), data

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

    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: PRNGKeyArray) -> PhysicsModel:
        rng, key = jax.random.split(rng)
        # Skip the first 6 DOFs (free joint)
        kd = model.dof_damping[6:] * jax.random.uniform(
            key, shape=(model.dof_damping.shape[0] - 6,), minval=self.scale_lower, maxval=self.scale_upper
        )
        dof_damping = jnp.concatenate([model.dof_damping[:6], kd])

        return update_model_field(model, "dof_damping", dof_damping), data
