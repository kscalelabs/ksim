"""Defines the base reset classes and builders."""

import functools
import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import PRNGKeyArray
from mujoco import mjx

from ksim.utils.data import BuilderData

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


T = TypeVar("T", bound=Reset)


class ResetBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, data: BuilderData) -> T:
        """Builds a reset from a MuJoCo model."""


@attrs.define(frozen=True, kw_only=True)
class HFieldXYPositionReset(Reset):
    """Resets the robot's XY position within a bounded region, with height from aheightfield."""

    bounds: tuple[float, float, float, float]
    padded_bounds: tuple[float, float, float, float]
    x_range: float = attrs.field(default=5.0)
    y_range: float = attrs.field(default=5.0)
    hfield_data: jnp.ndarray

    def __call__(self, data: mjx.Data, rng: PRNGKeyArray) -> mjx.Data:
        x_bound, y_bound, z_top, _ = self.bounds

        # Unpack padded bounds.
        lower_x, upper_x, lower_y, upper_y = self.padded_bounds

        rng_split, keyx, keyy = jax.random.split(rng, 3)
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
        qpos_j = qpos_j.at[2:3].set(z + z_top)

        return data.replace(qpos=qpos_j)

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

        rng_split, keyx, keyy = jax.random.split(rng, 3)
        offset_x = jax.random.uniform(keyx, (1,), minval=-self.x_range, maxval=self.x_range)
        offset_y = jax.random.uniform(keyy, (1,), minval=-self.y_range, maxval=self.y_range)

        new_x = jnp.clip(offset_x, lower_x, upper_x)
        new_y = jnp.clip(offset_y, lower_y, upper_y)

        qpos_j = data.qpos
        qpos_j = qpos_j.at[0:1].set(new_x)
        qpos_j = qpos_j.at[1:2].set(new_y)
        qpos_j = qpos_j.at[2:3].set(z_pos + self.robot_base_height)
        return data.replace(qpos=qpos_j)


@attrs.define(frozen=True, kw_only=True)
class XYPositionResetBuilder(ResetBuilder[Reset]):
    """Builds an XY position reset based on whether the config has a heightfield or plane floor."""

    x_range: float = attrs.field(default=5.0)
    y_range: float = attrs.field(default=5.0)
    x_edge_padding: float = attrs.field(default=5.0)
    y_edge_padding: float = attrs.field(default=5.0)

    def __call__(self, data: BuilderData) -> Reset:
        def compute_padded_bounds(
            x_bound: float, y_bound: float, x_edge_padding: float, y_edge_padding: float
        ) -> tuple[float, float, float, float]:
            """Helper to compute effective (padded) bounds from based on edge padding values."""
            lower_x = -x_bound + x_edge_padding
            upper_x = x_bound - x_edge_padding
            lower_y = -y_bound + y_edge_padding
            upper_y = y_bound - y_edge_padding
            return (lower_x, upper_x, lower_y, upper_y)

        # Get robot base height.
        try:
            names = data.model.names.decode().split("\x00")
            base_idx = names.index("floating_base_link") - 1  # Adjust for body index.
            robot_base_height = data.model.body_pos[base_idx][2]
            logger.info(f"Robot base height: {robot_base_height}")
        except ValueError:
            raise ValueError("Could not find floating_base_link in the model.")

        # Return HFieldXYPositionReset if there is a heightfield.
        if hasattr(data.model, "hfield_size") and data.model.hfield_size.size > 0:
            x_bound, y_bound, z_top, z_bottom = data.model.hfield_size.flatten().tolist()
            nx = (
                int(data.robot_model.hfield_nrow.item())
                if hasattr(data.robot_model.hfield_nrow, "item")
                else int(data.robot_model.hfield_nrow)
            )
            ny = (
                int(data.robot_model.hfield_ncol.item())
                if hasattr(data.robot_model.hfield_ncol, "item")
                else int(data.robot_model.hfield_ncol)
            )
            hfield_data = data.model.hfield_data.reshape(nx, ny)
            logger.info(f"Using heightfield based floor with shape: {hfield_data.shape}")
            padded_bounds = compute_padded_bounds(
                x_bound, y_bound, self.x_edge_padding, self.y_edge_padding
            )
            return HFieldXYPositionReset(
                bounds=(x_bound, y_bound, z_top, z_bottom),
                padded_bounds=padded_bounds,
                x_range=self.x_range,
                y_range=self.y_range,
                hfield_data=hfield_data,
            )

        else:
            plane_indices = [i for i, t in enumerate(data.model.geom_type) if t == 0]
            if not plane_indices:
                raise ValueError(
                    "No heightfield or plane geom found in the model. MuJoCo scene missing floor!"
                )
            floor_idx = plane_indices[0]
            x_bound, y_bound = 5.0, 5.0
            z_pos = data.model.geom_pos[floor_idx][2]
            logger.info(f"Using plane based floor with bounds: {x_bound}, {y_bound}, {z_pos}")
            padded_bounds = compute_padded_bounds(
                x_bound, y_bound, self.x_edge_padding, self.y_edge_padding
            )
            return PlaneXYPositionReset(
                bounds=(x_bound, y_bound, z_pos),
                padded_bounds=padded_bounds,
                x_range=self.x_range,
                y_range=self.y_range,
                robot_base_height=robot_base_height,
            )


@attrs.define(frozen=True, kw_only=True)
class RandomizeJointPositions(Reset):
    """Resets the joint positions of the robot to random values."""

    scale: float = attrs.field(default=0.01)

    def __call__(self, data: mjx.Data, rng: PRNGKeyArray) -> mjx.Data:
        qpos = data.qpos
        qpos = qpos + jax.random.uniform(rng, qpos.shape, minval=-self.scale, maxval=self.scale)
        data = data.replace(qpos=qpos)
        return data


@attrs.define(frozen=True, kw_only=True)
class RandomizeJointVelocities(Reset):
    """Resets the joint velocities of the robot to random values."""

    scale: float = attrs.field(default=0.01)

    def __call__(self, data: mjx.Data, rng: PRNGKeyArray) -> mjx.Data:
        qvel = data.qvel
        qvel = qvel + jax.random.uniform(rng, qvel.shape, minval=-self.scale, maxval=self.scale)
        data = data.replace(qvel=qvel)
        return data
