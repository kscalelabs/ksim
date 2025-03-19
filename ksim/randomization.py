"""Randomize each environment when gathering trajectories."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import attrs
import jax
import jax.numpy as jnp
import mujoco
from jaxtyping import PRNGKeyArray
from mujoco import mjx
from omegaconf import MISSING

from ksim.env.data import PhysicsModel
from ksim.utils.mujoco import get_body_data_idx_by_name, update_model_field


@attrs.define(frozen=True, kw_only=True)
class Randomization(ABC):
    """Randomize the joint positions of the robot."""

    @abstractmethod
    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        """Randomize the model for a single environment."""


T = TypeVar("T", bound=Randomization)


class RandomizerBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, physics_model: PhysicsModel, rng: PRNGKeyArray) -> T:
        """Builds a randomizer from a physical model."""


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

    name: str = "dof_frictionloss"
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
        match type(model):
            case mujoco.MjModel:
                new_frictionloss = jnp.concatenate([model.dof_frictionloss[:6], frictionloss])
            case mjx.Model:
                new_frictionloss = model.dof_frictionloss.at[6:].set(frictionloss)

        return update_model_field(model, self.name, new_frictionloss)


@attrs.define(frozen=True, kw_only=True)
class FloorFrictionRandomization(Randomization):
    """Randomizes the floor friction."""

    name: str = "geom_friction"
    scale_lower: float = attrs.field(default=0.4)
    scale_upper: float = attrs.field(default=1.0)
    floor_body_id: int = attrs.field(default=MISSING)

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
        return update_model_field(model, self.name, new_geom_friction)


@attrs.define(frozen=True, kw_only=True)
class ArmatureRandomization(Randomization):
    """Randomizes the armature."""

    name: str = "dof_armature"
    scale_lower: float = attrs.field(default=1.0)
    scale_upper: float = attrs.field(default=1.05)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        """Randomize the armature of the robot."""
        # Skip the first 6 DOFs (free joint)
        armature = model.dof_armature[6:] * jax.random.uniform(
            rng, shape=(model.dof_armature.shape[0] - 6,), minval=self.scale_lower, maxval=self.scale_upper
        )
        match type(model):
            case mujoco.MjModel:
                new_armature = jnp.concatenate([model.dof_armature[:6], armature])
            case mjx.Model:
                new_armature = model.dof_armature.at[6:].set(armature)

        return update_model_field(model, self.name, new_armature)


@attrs.define(frozen=True, kw_only=True)
class TorsoMassRandomization(Randomization):
    """Randomizes the torso mass."""

    name: str = "body_mass"
    scale_lower: float = attrs.field(default=-1.0)
    scale_upper: float = attrs.field(default=1.0)
    torso_body_id: int = attrs.field(default=MISSING)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        """Randomize the torso mass of the robot."""
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=self.scale_lower, maxval=self.scale_upper)
        match type(model):
            case mujoco.MjModel:
                new_body_mass = jnp.concatenate(
                    [
                        model.body_mass[: self.torso_body_id],
                        jnp.array([dmass]),
                        model.body_mass[self.torso_body_id + 1 :],
                    ]
                )
            case mjx.Model:
                new_body_mass = model.body_mass.at[self.torso_body_id].set(model.body_mass[self.torso_body_id] + dmass)

        return update_model_field(model, self.name, new_body_mass)


@attrs.define(frozen=True, kw_only=True)
class TorsoMassRandomizerBuilder(RandomizerBuilder[TorsoMassRandomization]):
    scale_lower: float = attrs.field(default=-1.0)
    scale_upper: float = attrs.field(default=1.0)
    torso_body_name: str = attrs.field(default=MISSING)

    def __call__(self, model: PhysicsModel) -> TorsoMassRandomization:
        """Build a torso mass randomizer from a physical model."""
        torso_body_id = get_body_data_idx_by_name(model)[self.torso_body_name]
        return TorsoMassRandomization(
            scale_lower=self.scale_lower,
            scale_upper=self.scale_upper,
            torso_body_id=torso_body_id,
        )


@attrs.define(frozen=True, kw_only=True)
class JointDampingRandomization(Randomization):
    """Randomizes the joint damping."""

    name: str = "dof_damping"
    scale_lower: float = attrs.field(default=0.9)
    scale_upper: float = attrs.field(default=1.1)

    def __call__(self, model: PhysicsModel, rng: PRNGKeyArray) -> PhysicsModel:
        rng, key = jax.random.split(rng)
        # Skip the first 6 DOFs (free joint)
        kd = model.dof_damping[6:] * jax.random.uniform(
            key, shape=(model.dof_damping.shape[0] - 6,), minval=self.scale_lower, maxval=self.scale_upper
        )
        match type(model):
            case mujoco.MjModel:
                dof_damping = jnp.concatenate([model.dof_damping[:6], kd])
            case mjx.Model:
                dof_damping = model.dof_damping.at[6:].set(kd)

        return update_model_field(model, self.name, dof_damping)
