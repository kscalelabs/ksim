"""Randomize each environment when gathering trajectories."""

import functools
from abc import ABC, abstractmethod

import attrs
import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import PRNGKeyArray
from mujoco import mjx

from ksim.env.data import PhysicsData, PhysicsModel
from ksim.utils.mujoco import get_body_data_idx_by_name, update_data_field, update_model_field


@attrs.define(frozen=True, kw_only=True)
class Randomization(ABC):
    """Randomize the joint positions of the robot."""

    @abstractmethod
    def initial_randomization(self, rng: PRNGKeyArray) -> tuple[PhysicsModel, PhysicsData]:
        """Randomize the model for a single environment."""

    @abstractmethod
    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: PRNGKeyArray) -> tuple[PhysicsModel, PhysicsData]:
        """Randomize the model for a single environment."""

    def get_name(self) -> str:
        """Get the name of the command."""
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def randomization_name(self) -> str:
        return self.get_name()


@attrs.define(frozen=True, kw_only=True)
class WeightRandomization(Randomization):
    """Randomize the body masses of the robot."""

    scale: float = attrs.field()

    def initial_randomization(self, rng: PRNGKeyArray) -> tuple[PhysicsModel, PhysicsData]:
        """Randomize the model for a single environment."""

    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: PRNGKeyArray) -> tuple[PhysicsModel, PhysicsData]:
        """Randomize the model for a single environment."""
        new_body_mass = model.body_mass * (jax.random.uniform(rng, model.body_mass.shape) * self.scale + 1.0)
        return update_model_field(model, "body_mass", new_body_mass), data


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


@attrs.define(frozen=True, kw_only=True)
class ForceRandomization(Randomization):
    """Randomize the force of the robot."""

    push_magnitude_range: tuple[float, float] = attrs.field()
    push_interval_range: tuple[float, float] = attrs.field()
    dt: float = attrs.field()

    def initial_randomization(self, rng: PRNGKeyArray) -> tuple[PhysicsModel, PhysicsData]:
        """Randomize the force of the robot."""
        rng, push1_rng, push2_rng = jax.random.split(rng, 3)
        push_interval = jax.random.uniform(
            push2_rng,
            minval=self.push_interval_range[0],
            maxval=self.push_interval_range[1],
        )
        push_step = jnp.array(0)  # random given the trajectory length
        push_interval_steps = jnp.round(push_interval / self.dt).astype(jnp.int32)
        return jnp.array([push_interval_steps, push_step])

    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: PRNGKeyArray) -> tuple[PhysicsModel, PhysicsData]:
        """Push the model for a single environment."""
        push_step = 1
        push_interval_steps = 200
        rng, push1_rng, push2_rng = jax.random.split(rng, 3)
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jnp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self.push_magnitude_range[0],
            maxval=self.push_magnitude_range[1],
        )
        push = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)])
        # push *= (
        #     jnp.mod(data.info["push_step"] + 1, data.info["push_interval_steps"])
        #     == 0
        # )
        push = push * push_magnitude + data.qvel[:2]
        new_qvel = jnp.concatenate([push, data.qvel[2:]])
        return model, update_data_field(data, "qvel", new_qvel)
