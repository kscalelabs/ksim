"""Defines a base interface for defining reward functions."""

__all__ = [
    "Reward",
    "TerminationPenalty",
    "LinearVelocityZPenalty",
    "AngularVelocityXYPenalty",
    "JointVelocityPenalty",
    "LinearVelocityTrackingReward",
    "BaseHeightReward",
    "ActionSmoothnessPenalty",
    "ActuatorForcePenalty",
]

import functools
import logging
from abc import ABC, abstractmethod

import attrs
import jax.numpy as jnp
import xax
from jaxtyping import Array

from ksim.types import Trajectory

logger = logging.getLogger(__name__)


def reward_scale_validator(inst: "Reward", attr: attrs.Attribute, value: float) -> None:
    # Reward function classes should end with either "Reward" or "Penalty",
    # which we use here to check if the scale is positive or negative.
    if inst.reward_name.lower().endswith("reward"):
        if value < 0:
            raise RuntimeError(f"Reward function {inst.reward_name} has a negative scale {value}")
    elif inst.reward_name.lower().endswith("penalty"):
        if value > 0:
            raise RuntimeError(f"Penalty function {inst.reward_name} has a positive scale {value}")
    else:
        logger.warning("Reward function %s does not end with 'Reward' or 'Penalty': %f", inst.reward_name, value)


@attrs.define(frozen=True, kw_only=True)
class Reward(ABC):
    """Base class for defining reward functions."""

    scale: float = attrs.field(validator=reward_scale_validator)

    @abstractmethod
    def __call__(self, trajectory: Trajectory) -> Array: ...

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def reward_name(self) -> str:
        return self.get_name()


@attrs.define(frozen=True, kw_only=True)
class TerminationPenalty(Reward):
    """Penalty for terminating the episode."""

    def __call__(self, trajectory: Trajectory) -> Array:
        return trajectory.done


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityZPenalty(Reward):
    """Penalty for how fast the robot is moving in the z-direction."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: Trajectory) -> Array:
        lin_vel_z = trajectory.qvel[..., 2]
        return xax.get_norm(lin_vel_z, self.norm)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityXYPenalty(Reward):
    """Penalty for how fast the robot is rotating in the xy-plane."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: Trajectory) -> Array:
        ang_vel_xy = trajectory.qvel[..., 3:5]
        return xax.get_norm(ang_vel_xy, self.norm).mean(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class JointVelocityPenalty(Reward):
    """Penalty for how fast the joint angular velocities are changing."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: Trajectory) -> Array:
        joint_vel = trajectory.qvel[..., 6:]
        return xax.get_norm(joint_vel, self.norm).mean(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(Reward):
    """Reward for tracking the linear velocity command."""

    norm: xax.NormType = attrs.field(default="l2")
    command_name: str = attrs.field(default="linear_velocity_command")

    def __call__(self, trajectory: Trajectory) -> Array:
        lin_vel_cmd = trajectory.command[self.command_name]
        lin_vel_x_cmd = lin_vel_cmd[..., 0]
        lin_vel_y_cmd = lin_vel_cmd[..., 1]
        lin_vel_x = trajectory.qvel[..., 1]
        lin_vel_y = trajectory.qvel[..., 2]
        return xax.get_norm(lin_vel_x - lin_vel_x_cmd, self.norm) + xax.get_norm(lin_vel_y - lin_vel_y_cmd, self.norm)


@attrs.define(frozen=True, kw_only=True)
class BaseHeightReward(Reward):
    """Reward for tracking the base height target."""

    height_target: float = attrs.field()
    norm: xax.NormType = attrs.field(default="l1")
    sensitivity: float = attrs.field(default=5.0)

    def __call__(self, trajectory: Trajectory) -> Array:
        base_height = trajectory.qpos[..., 2]
        return jnp.exp(-xax.get_norm(base_height - self.height_target, self.norm) * self.sensitivity)


@attrs.define(frozen=True, kw_only=True)
class ActionSmoothnessPenalty(Reward):
    """Penalty for large changes between consecutive actions."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: Trajectory) -> Array:
        current_actions = trajectory.action

        # Shift actions to get previous actions (pad with first action)
        previous_actions = jnp.concatenate(
            [
                current_actions[..., :1, :],  # First action
                current_actions[..., :-1, :],  # Previous actions for remaining timesteps
            ],
            axis=-2,
        )

        action_deltas = current_actions - previous_actions

        return xax.get_norm(action_deltas, self.norm).mean(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class ActuatorForcePenalty(Reward):
    """Penalty for high actuator forces."""

    norm: xax.NormType = attrs.field(default="l1")
    observation_name: str = attrs.field(default="actuator_force_observation")

    def __call__(self, trajectory: Trajectory) -> Array:
        if self.observation_name not in trajectory.obs:
            raise ValueError(f"Observation {self.observation_name} not found; add it as an observation in your task.")
        return xax.get_norm(trajectory.obs[self.observation_name], self.norm).mean(axis=-1)
