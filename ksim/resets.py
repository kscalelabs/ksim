"""Defines the base reset classes and builders."""

__all__ = [
    "Reset",
    "HFieldXYPositionReset",
    "InitialMotionStateReset",
    "PlaneXYPositionReset",
    "RandomJointPositionReset",
    "RandomJointVelocityReset",
    "RandomBaseVelocityXYReset",
    "get_xy_position_reset",
    "RandomHeadingReset",
    "RandomHeightReset",
]

import functools
import logging
from abc import ABC, abstractmethod

import attrs
import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx

from ksim.types import PhysicsData, PhysicsModel
from ksim.utils.mujoco import (
    get_joint_names_in_order,
    get_position_limits,
    slice_update,
    update_data_field,
)
from ksim.utils.priors import MotionReferenceData

logger = logging.getLogger(__name__)


@attrs.define(frozen=True, kw_only=True)
class Reset(ABC):
    """Base class for resets."""

    @abstractmethod
    def __call__(self, data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsData:
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
    hfield_data: xax.HashableArray
    robot_base_height: float = attrs.field(default=0.0)

    def __call__(self, data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsData:
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
        nx, ny = self.hfield_data.array.shape
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
        z = self.hfield_data.array[x_idx, y_idx]
        qpos_j = qpos_j.at[2:3].set(z + z_top + self.robot_base_height)
        data = update_data_field(data, "qpos", qpos_j)
        return data


@attrs.define(frozen=True, kw_only=True)
class PlaneXYPositionReset(Reset):
    """Resets the robot's XY position and sets Z to a fixed plane height."""

    bounds: tuple[float, float, float]
    padded_bounds: tuple[float, float, float, float]
    x_range: float = attrs.field(default=5.0)
    y_range: float = attrs.field(default=5.0)
    robot_base_height: float = attrs.field(default=0.0)

    def __call__(self, data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsData:
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
    zeros: tuple[float, ...] | None = attrs.field(default=None)

    def __call__(self, data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsData:
        pos = jax.random.uniform(rng, data.qpos[7:].shape, minval=-self.scale, maxval=self.scale) * curriculum_level
        if self.zeros is not None:
            pos = pos + jnp.array(self.zeros)
        new_qpos = data.qpos[7:] + pos
        new_qpos = jnp.concatenate([data.qpos[:7], new_qpos])
        data = update_data_field(data, "qpos", new_qpos)
        return data

    @classmethod
    def create(
        cls,
        physics_model: PhysicsModel,
        zeros: dict[str, float] | None = None,
        scale: float = 0.01,
    ) -> "RandomJointPositionReset":
        joint_names = get_joint_names_in_order(physics_model)[1:]  # Remove the first joint (root)
        zeros_values = [0.0 for _ in range(len(joint_names))]
        if zeros is not None:
            joint_limits = get_position_limits(physics_model)
            for name, value in zeros.items():
                if (index := joint_names.index(name)) < 0:
                    raise ValueError(f"Joint {name} not found in model. Choices are: {joint_names}")
                joint_min, joint_max = joint_limits[name]
                if value < joint_min or value > joint_max:
                    raise ValueError(f"Zero value {value} for joint {name} is out of range {joint_min}, {joint_max}")
                zeros_values[index] = value
        return cls(scale=scale, zeros=tuple(zeros_values))


@attrs.define(frozen=True, kw_only=True)
class RandomJointVelocityReset(Reset):
    """Resets the joint velocities of the robot to random values."""

    scale: float = attrs.field(default=0.01)

    def __call__(self, data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsData:
        noise = jax.random.uniform(rng, data.qvel[6:].shape, minval=-self.scale, maxval=self.scale) * curriculum_level
        new_qvel = data.qvel[6:] + noise
        new_qvel = jnp.concatenate([data.qvel[:6], new_qvel])
        data = update_data_field(data, "qvel", new_qvel)
        return data


@attrs.define(frozen=True, kw_only=True)
class RandomBaseVelocityXYReset(Reset):
    """Resets the base velocity of the robot to random values."""

    scale: float = attrs.field(default=0.01)

    def __call__(self, data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsData:
        qvel = data.qvel
        noise = jax.random.uniform(rng, qvel[0:2].shape, minval=-self.scale, maxval=self.scale) * curriculum_level
        match type(data):
            case mujoco.MjData:
                qvel[0:2] = noise
            case mjx.Data:
                qvel.at[0:2].set(noise)
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
            hfield_data=xax.hashable_array(hfield_data),
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


@attrs.define(frozen=True, kw_only=True)
class InitialMotionStateReset(Reset):
    """Resets the initial state to the one from the motion bank."""

    reference_motion: MotionReferenceData
    freejoint: bool = attrs.field(default=False)

    def __call__(self, data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsData:
        frame_index = jax.random.randint(rng, (1,), 0, self.reference_motion.num_frames)[0]
        qpos = self.reference_motion.get_qpos_at_step(frame_index)
        qvel = self.reference_motion.get_qvel_at_step(frame_index)

        if self.freejoint:
            data = update_data_field(data, "qpos", qpos)
            data = update_data_field(data, "qvel", qvel)
        else:
            new_qpos = jnp.concatenate([data.qpos[:7], qpos[7:]])
            data = update_data_field(data, "qpos", new_qpos)
            new_qvel = jnp.concatenate([data.qvel[:6], qvel[7:]])
            data = update_data_field(data, "qvel", new_qvel)

        data = update_data_field(data, "time", frame_index * self.reference_motion.ctrl_dt)
        return data


@attrs.define(frozen=True, kw_only=True)
class RandomHeadingReset(Reset):
    """Resets the heading of the robot to a random value."""

    def __call__(self, data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsData:
        angle = jax.random.uniform(rng, data.qpos.shape[:-1], minval=-jnp.pi, maxval=jnp.pi)
        euler = jnp.stack([jnp.zeros_like(angle), jnp.zeros_like(angle), angle], axis=-1)
        quat = xax.euler_to_quat(euler)
        qpos = slice_update(data, "qpos", slice(3, 7), quat)
        data = update_data_field(data, "qpos", qpos)
        return data


@attrs.define(frozen=True, kw_only=True)
class RandomHeightReset(Reset):
    """Resets the height of the robot to a random value."""

    range: tuple[float, float] = attrs.field(default=(0.0, 0.1))

    def __call__(self, data: PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsData:
        qpos = data.qpos
        min_height, max_height = self.range
        new_z = jax.random.uniform(rng, qpos[..., 2:3].shape, minval=min_height, maxval=max_height)
        qpos = slice_update(data, "qpos", slice(2, 3), qpos[..., 2:3] + new_z)
        data = update_data_field(data, "qpos", qpos)
        return data
