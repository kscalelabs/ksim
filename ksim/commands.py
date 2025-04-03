"""Defines the base command class."""

__all__ = [
    "Command",
    "LinearVelocityCommand",
    "AngularVelocityCommand",
    "LinearVelocityStepCommand",
    "AngularVelocityStepCommand",
    "CartesianBodyTargetCommand",
    "GlobalBodyQuaternionCommand",
]

import functools
from abc import ABC, abstractmethod
from typing import Collection, Self

import attrs
import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.types import PhysicsData, PhysicsModel, Trajectory
from ksim.utils.mujoco import get_body_data_idx_from_name
from ksim.utils.types import CartesianIndex, dimension_index_validator
from ksim.vis import Marker


@attrs.define(frozen=True, kw_only=True)
class Command(ABC):
    """Base class for commands."""

    @abstractmethod
    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        """Returns the initial command.

        Args:
            physics_data: The current physics data.
            curriculum_level: The current curriculum level, a value between
                zero and one that indicates the difficulty of the task.
            rng: The random number generator.

        Returns:
            The initial command, with shape (command_dim).
        """

    @abstractmethod
    def __call__(
        self,
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        """Updates the command.

        Args:
            prev_command: The previous command.
            physics_data: The current physics data.
            curriculum_level: The current curriculum level, a value between
                zero and one that indicates the difficulty of the task.
            rng: The random number generator.

        Returns:
            The command to perform, with shape (command_dim).
        """

    def get_markers(self) -> Collection[Marker]:
        """Get the visualizations for the command.

        Args:
            command: The command to get the visualizations for.

        Returns:
            The visualizations to add to the scene.
        """
        return []

    def get_name(self) -> str:
        """Get the name of the command."""
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def command_name(self) -> str:
        return self.get_name()


@attrs.define(frozen=True)
class LinearVelocityCommand(Command):
    """Command to move the robot in a straight line.

    By convention, X is forward and Y is left. The switching probability is the
    probability of resampling the command at each step. The zero probability is
    the probability of the command being zero - this can be used to turn off
    any command.
    """

    range: tuple[float, float] = attrs.field()
    index: CartesianIndex | None = attrs.field(default=None, validator=dimension_index_validator)
    zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)
    vis_height: float = attrs.field(default=1.0)
    vis_scale: float = attrs.field(default=0.05)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng, rng_zero = jax.random.split(rng)
        minval, maxval = self.range
        value = jax.random.uniform(rng, (1,), minval=minval, maxval=maxval)
        zero_mask = jax.random.bernoulli(rng_zero, self.zero_prob)
        return jnp.where(zero_mask, 0.0, value)

    def __call__(
        self,
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)

    def get_name(self) -> str:
        return f"{super().get_name()}{'' if self.index is None else f'_{self.index}'}"


@attrs.define(frozen=True)
class AngularVelocityCommand(Command):
    """Command to turn the robot."""

    scale: float = attrs.field()
    index: CartesianIndex | None = attrs.field(default=None, validator=dimension_index_validator)
    zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        """Returns (1,) array with angular velocity."""
        rng_a, rng_b = jax.random.split(rng)
        zero_mask = jax.random.bernoulli(rng_a, self.zero_prob)
        cmd = jax.random.uniform(rng_b, (1,), minval=-self.scale, maxval=self.scale)
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

    def __call__(
        self,
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)

    def get_name(self) -> str:
        return f"{super().get_name()}{'' if self.index is None else f'_{self.index}'}"


@attrs.define(frozen=True)
class LinearVelocityStepCommand(Command):
    """This is the same as LinearVelocityCommand, but it is discrete."""

    x_range: tuple[float, float] = attrs.field()
    y_range: tuple[float, float] = attrs.field()
    x_fwd_prob: float = attrs.field()
    y_fwd_prob: float = attrs.field()
    x_zero_prob: float = attrs.field(default=0.0)
    y_zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)
    vis_height: float = attrs.field(default=1.0)
    vis_scale: float = attrs.field(default=0.05)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_x, rng_y, rng_zero_x, rng_zero_y = jax.random.split(rng, 4)
        (xmin, xmax), (ymin, ymax) = self.x_range, self.y_range
        x = jax.random.bernoulli(rng_x, self.x_fwd_prob, (1,)) * (xmax - xmin) + xmin
        y = jax.random.bernoulli(rng_y, self.y_fwd_prob, (1,)) * (ymax - ymin) + ymin
        x_zero_mask = jax.random.bernoulli(rng_zero_x, self.x_zero_prob)
        y_zero_mask = jax.random.bernoulli(rng_zero_y, self.y_zero_prob)
        return jnp.concatenate(
            [
                jnp.where(x_zero_mask, 0.0, x),
                jnp.where(y_zero_mask, 0.0, y),
            ]
        )

    def __call__(
        self,
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)


@attrs.define(frozen=True)
class AngularVelocityStepCommand(Command):
    """This is the same as AngularVelocityCommand, but it is discrete."""

    scale: float = attrs.field()
    prob: float = attrs.field(default=0.5)
    zero_prob: float = attrs.field(default=0.0)
    switch_prob: float = attrs.field(default=0.0)

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        cmd = (jax.random.bernoulli(rng_a, self.prob, (1,)) * 2 - 1) * self.scale
        zero_mask = jax.random.bernoulli(rng_b, self.zero_prob)
        return jnp.where(zero_mask, jnp.zeros_like(cmd), cmd)

    def __call__(
        self,
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)


@attrs.define(kw_only=True)
class CartesianBodyTargetMarker(Marker):
    command_name: str = attrs.field()

    def __attrs_post_init__(self) -> None:
        if self.target_name is None or self.target_type != "body":
            raise ValueError("Base body name must be provided. Make sure to create with `get`.")

    def update(self, trajectory: Trajectory) -> None:
        """Update the marker position and rotation."""
        self.pos = trajectory.command[self.command_name]

    @classmethod
    def get(
        cls, command_name: str, base_body_name: str, radius: float, rgba: tuple[float, float, float, float]
    ) -> Self:
        return cls(
            command_name=command_name,
            target_name=base_body_name,
            target_type="body",
            geom=mujoco.mjtGeom.mjGEOM_SPHERE,
            scale=(radius, radius, radius),
            rgba=rgba,
        )


@attrs.define(frozen=True)
class CartesianBodyTargetCommand(Command):
    """Samples a target xyz position along a sphere from a pivot point.

    E.g. sample a sphere centered around the shoulder, where the sampled point
    is the relative xpos with respect to the pelvis. This point will move along
    with the base but only the base.
    """

    pivot_body_name: str = attrs.field()
    base_body_name: str = attrs.field()
    pivot_id: int = attrs.field()
    base_id: int = attrs.field()
    sample_sphere_radius: float = attrs.field()
    positive_x: bool = attrs.field()
    positive_y: bool = attrs.field()
    positive_z: bool = attrs.field()
    switch_prob: float = attrs.field()
    vis_radius: float = attrs.field()
    vis_color: tuple[float, float, float, float] = attrs.field()

    def _sample_sphere(self, rng: PRNGKeyArray) -> Array:
        # Sample a random unit vector symmetrically.
        vec = jax.random.normal(rng, (3,))
        vec /= jnp.linalg.norm(vec)

        # Generate a random radius with the proper distribution.
        u = jax.random.uniform(rng, (1,))
        r = self.sample_sphere_radius * (u ** (1 / 3))

        x, y, z = vec * r
        x = jnp.where(self.positive_x and x > 0.0, x, -x)
        y = jnp.where(self.positive_y and y > 0.0, y, -y)
        z = jnp.where(self.positive_z and z > 0.0, z, -z)

        return jnp.array([x, y, z])

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        sphere_sample = self._sample_sphere(rng)
        pivot_pos = jnp.array(physics_data.xpos[self.pivot_id])
        base_pos = jnp.array(physics_data.xpos[self.base_id])
        return pivot_pos + sphere_sample - base_pos

    def __call__(
        self,
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)

    def get_markers(self) -> Collection[Marker]:
        return [CartesianBodyTargetMarker.get(self.command_name, self.base_body_name, self.vis_radius, self.vis_color)]

    def get_name(self) -> str:
        return f"{super().get_name()}_{self.pivot_body_name}"

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        pivot_name: str,
        base_name: str,
        sample_sphere_radius: float,
        positive_x: bool = True,
        positive_y: bool = True,
        positive_z: bool = True,
        switch_prob: float = 0.1,
        vis_radius: float = 0.05,
        vis_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.8),
    ) -> Self:
        pivot_id = get_body_data_idx_from_name(model, pivot_name)
        base_id = get_body_data_idx_from_name(model, base_name)
        return cls(
            pivot_body_name=pivot_name,
            base_body_name=base_name,
            pivot_id=pivot_id,
            base_id=base_id,
            sample_sphere_radius=sample_sphere_radius,
            positive_x=positive_x,
            positive_y=positive_y,
            positive_z=positive_z,
            switch_prob=switch_prob,
            vis_radius=vis_radius,
            vis_color=vis_color,
        )


@attrs.define(kw_only=True)
class GlobalBodyQuaternionMarker(Marker):
    command_name: str = attrs.field()

    def __attrs_post_init__(self) -> None:
        if self.target_name is None or self.target_type != "body":
            raise ValueError("Base body name must be provided. Make sure to create with `get`.")

    def update(self, trajectory: Trajectory) -> None:
        """Update the marker rotation."""
        command = trajectory.command[self.command_name]
        # Check if command is zeros (null quaternion)
        is_null = jnp.all(jnp.isclose(command, 0.0))

        # Only update orientation if command is not null
        if not is_null:
            self.geom = mujoco.mjtGeom.mjGEOM_ARROW
            self.orientation = command
        else:
            self.geom = mujoco.mjtGeom.mjGEOM_SPHERE

    @classmethod
    def get(
        cls,
        command_name: str,
        base_body_name: str,
        size: float,
        magnitude: float,
        rgba: tuple[float, float, float, float],
    ) -> Self:
        return cls(
            command_name=command_name,
            target_name=base_body_name,
            target_type="body",
            geom=mujoco.mjtGeom.mjGEOM_ARROW,
            scale=(size, size, magnitude),
            rgba=rgba,
        )


@attrs.define(frozen=True)
class GlobalBodyQuaternionCommand(Command):
    """Samples a target quaternion orientation for a body.

    This command samples random quaternions to specify target orientations
    for a body in global coordinates, with an option to sample a null quaternion.
    """

    base_body_name: str = attrs.field()
    base_id: int = attrs.field()
    switch_prob: float = attrs.field()
    null_prob: float = attrs.field()  # Probability of sampling null quaternion
    vis_magnitude: float = attrs.field()
    vis_size: float = attrs.field()
    vis_color: tuple[float, float, float, float] = attrs.field()

    def initial_command(
        self,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        is_null = jax.random.bernoulli(rng_a, self.null_prob)
        quat = jax.random.normal(rng_b, (4,))
        random_quat = quat / jnp.linalg.norm(quat)
        return jnp.where(is_null, jnp.zeros(4), random_quat)

    def __call__(
        self,
        prev_command: Array,
        physics_data: PhysicsData,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        rng_a, rng_b = jax.random.split(rng)
        switch_mask = jax.random.bernoulli(rng_a, self.switch_prob)
        new_commands = self.initial_command(physics_data, curriculum_level, rng_b)
        return jnp.where(switch_mask, new_commands, prev_command)

    def get_markers(self) -> Collection[Marker]:
        return [
            GlobalBodyQuaternionMarker.get(
                self.command_name, self.base_body_name, self.vis_size, self.vis_magnitude, self.vis_color
            )
        ]

    def get_name(self) -> str:
        return f"{super().get_name()}_{self.base_body_name}"

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        base_name: str,
        switch_prob: float = 0.1,
        null_prob: float = 0.1,
        vis_magnitude: float = 0.5,
        vis_size: float = 0.05,
        vis_color: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.8),
    ) -> Self:
        base_id = get_body_data_idx_from_name(model, base_name)
        return cls(
            base_body_name=base_name,
            base_id=base_id,
            switch_prob=switch_prob,
            null_prob=null_prob,
            vis_magnitude=vis_magnitude,
            vis_size=vis_size,
            vis_color=vis_color,
        )
