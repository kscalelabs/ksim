"""Defines a base interface for defining reward functions."""

__all__ = [
    "MonotonicFn",
    "norm_to_reward",
    "Reward",
    "StayAliveReward",
    "LinearVelocityPenalty",
    "AngularVelocityPenalty",
    "JointVelocityPenalty",
    "LinearVelocityTrackingReward",
    "AngularVelocityTrackingReward",
    "BaseHeightReward",
    "BaseHeightRangeReward",
    "ActionSmoothnessPenalty",
    "ActuatorForcePenalty",
    "BaseJerkZPenalty",
    "ActuatorJerkPenalty",
    "AvoidLimitsReward",
    "ActionNearPositionPenalty",
    "FeetLinearVelocityTrackingPenalty",
    "FeetFlatReward",
    "FeetPhaseReward",
    "FeetNoContactReward",
    "CartesianBodyTargetReward",
    "CartesianBodyTargetPenalty",
    "CartesianBodyTargetVectorReward",
    "ContinuousCartesianBodyTargetReward",
    "GlobalBodyQuaternionReward",
]

import functools
import logging
from abc import ABC, abstractmethod
from typing import Collection, Literal, Self

import attrs
import chex
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array

from ksim.types import PhysicsModel, Trajectory
from ksim.utils.mujoco import get_body_data_idx_from_name
from ksim.utils.types import CartesianIndex, cartesian_index_to_dim, dimension_index_validator, norm_validator
from ksim.vis import Marker

logger = logging.getLogger(__name__)

MonotonicFn = Literal["exp", "inv"]


def norm_to_reward(value: Array, temp: float, monotonic_fn: MonotonicFn) -> Array:
    """Helper function for converting from a norm to a reward.

    Args:
        value: The value (usually a norm) to convert to a reward.
        temp: The temperature to use for the conversion. Higher temperatures
            will make the reward drop off less steeply.
        monotonic_fn: The monotonic function to use for the conversion.

    Returns:
        The reward.
    """
    match monotonic_fn:
        case "exp":
            return jnp.exp(-value / temp)
        case "inv":
            return 1.0 / (value / temp + 1.0)
        case "sigmoid":
            return 1.0 / (1.0 + jnp.exp(-value / temp))
        case _:
            raise ValueError(f"Invalid monotonic function: {monotonic_fn}")


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
    def __call__(self, trajectory: Trajectory) -> Array:
        """Get the reward for a single trajectory.

        You may assume that the dimensionality is (time, *leaf_dims) accross all
        leaves of the trajectory.

        Args:
            trajectory: The trajectory to get the reward for.

        Returns:
            An array of shape (time, *leaf_dims) containing the reward for each
            timestep.
        """

    def get_markers(self) -> Collection[Marker]:
        return []

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def reward_name(self) -> str:
        return self.get_name()


@attrs.define(frozen=True, kw_only=True)
class StayAliveReward(Reward):
    """Reward for staying alive.

    This provides a reward for staying alive, with a negative penalty on
    termination. These values are balanced by the balance parameter - a larger
    value will increase the relative penalty for termination.
    """

    balance: float = attrs.field(default=10.0)

    def __call__(self, trajectory: Trajectory) -> Array:
        return jnp.where(trajectory.done, -1.0, 1.0 / self.balance)


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityPenalty(Reward):
    """Penalty for how fast the robot is moving in the z-direction."""

    index: CartesianIndex = attrs.field(validator=dimension_index_validator)
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def get_name(self) -> str:
        return f"{self.index}_{super().get_name()}"

    def __call__(self, trajectory: Trajectory) -> Array:
        dim = cartesian_index_to_dim(self.index)
        lin_vel = trajectory.qvel[..., dim]
        return xax.get_norm(lin_vel, self.norm)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityPenalty(Reward):
    """Penalty for how fast the robot is rotating in the xy-plane."""

    index: CartesianIndex = attrs.field(validator=dimension_index_validator)
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def __call__(self, trajectory: Trajectory) -> Array:
        dim = cartesian_index_to_dim(self.index) + 3
        ang_vel = trajectory.qvel[..., dim]
        return xax.get_norm(ang_vel, self.norm)


@attrs.define(frozen=True, kw_only=True)
class JointVelocityPenalty(Reward):
    """Penalty for how fast the joint angular velocities are changing."""

    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    freejoint_first: bool = attrs.field(default=True)

    def __call__(self, trajectory: Trajectory) -> Array:
        if self.freejoint_first:
            joint_vel = trajectory.qvel[..., 6:]
            return xax.get_norm(joint_vel, self.norm).mean(axis=-1)
        else:
            return xax.get_norm(trajectory.qvel, self.norm).mean(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(Reward):
    """Penalty for deviating from the linear velocity command."""

    index: CartesianIndex = attrs.field(validator=dimension_index_validator)
    command_name: str = attrs.field()
    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    temp: float = attrs.field(default=1.0)
    monotonic_fn: MonotonicFn = attrs.field(default="inv")

    def __call__(self, trajectory: Trajectory) -> Array:
        dim = cartesian_index_to_dim(self.index)
        lin_vel_cmd = trajectory.command[self.command_name].squeeze(-1)
        lin_vel = trajectory.qvel[..., dim]
        norm = xax.get_norm(lin_vel - lin_vel_cmd, self.norm)
        return norm_to_reward(norm, self.temp, self.monotonic_fn)

    def get_name(self) -> str:
        return f"{self.index}_{super().get_name()}"


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityTrackingReward(Reward):
    """Penalty for deviating from the angular velocity command."""

    index: CartesianIndex = attrs.field(validator=dimension_index_validator)
    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    command_name: str = attrs.field(default="angular_velocity_command")
    temp: float = attrs.field(default=1.0)
    monotonic_fn: MonotonicFn = attrs.field(default="inv")

    def __call__(self, trajectory: Trajectory) -> Array:
        dim = cartesian_index_to_dim(self.index) + 3
        ang_vel_cmd = trajectory.command[self.command_name].squeeze(-1)
        ang_vel = trajectory.qvel[..., dim]
        norm = xax.get_norm(ang_vel - ang_vel_cmd, self.norm)
        return norm_to_reward(norm, self.temp, self.monotonic_fn)

    def get_name(self) -> str:
        return f"{self.index}_{super().get_name()}"


@attrs.define(frozen=True, kw_only=True)
class BaseHeightReward(Reward):
    """Penalty for deviating from the base height target."""

    height_target: float = attrs.field()
    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    temp: float = attrs.field(default=1.0)
    monotonic_fn: MonotonicFn = attrs.field(default="inv")

    def __call__(self, trajectory: Trajectory) -> Array:
        base_height = trajectory.qpos[..., 2]
        return norm_to_reward(xax.get_norm(base_height - self.height_target, self.norm), self.temp, self.monotonic_fn)


@attrs.define(frozen=True, kw_only=True)
class BaseHeightRangeReward(Reward):
    """Incentivizes keeping the base height within a certain range."""

    z_lower: float = attrs.field()
    z_upper: float = attrs.field()
    dropoff: float = attrs.field()

    def __call__(self, trajectory: Trajectory) -> Array:
        base_height = trajectory.qpos[..., 2]
        too_low = self.z_lower - base_height
        too_high = base_height - self.z_upper
        return (1.0 - jnp.maximum(too_low, too_high).clip(min=0.0) * self.dropoff).clip(min=0.0)


@attrs.define(frozen=True, kw_only=True)
class ActionSmoothnessPenalty(Reward):
    """Penalty for large changes between consecutive actions."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

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

    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    observation_name: str = attrs.field(default="actuator_force_observation")

    def __call__(self, trajectory: Trajectory) -> Array:
        if self.observation_name not in trajectory.obs:
            raise ValueError(f"Observation {self.observation_name} not found; add it as an observation in your task.")
        return xax.get_norm(trajectory.obs[self.observation_name], self.norm).mean(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class BaseJerkZPenalty(Reward):
    """Penalty for high base jerk."""

    ctrl_dt: float = attrs.field()
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)
    acc_obs_name: str = attrs.field(default="base_linear_acceleration_observation")

    def __call__(self, trajectory: Trajectory) -> Array:
        if self.acc_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.acc_obs_name} not found; add it as an observation in your task.")
        acc = trajectory.obs[self.acc_obs_name]
        acc = acc[None]
        acc_z = acc[..., 2]
        # First value will always be 0, because the acceleration is not changing.
        prev_acc_z = jnp.concatenate([acc_z[..., :1], acc_z[..., :-1]], axis=-1)
        # We multiply by ctrl_dt instead of dividing because we want the scale
        # for the penalty to be roughly the same magnitude as a velocity
        # penalty.
        jerk_z = (acc_z - prev_acc_z) * self.ctrl_dt
        return xax.get_norm(jerk_z, self.norm).squeeze(0)


@attrs.define(frozen=True, kw_only=True)
class ActuatorJerkPenalty(Reward):
    """Penalty for high actuator jerks."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)
    acc_obs_name: str = attrs.field(default="actuator_acceleration_observation")
    ctrl_dt: float = attrs.field()

    def __call__(self, trajectory: Trajectory) -> Array:
        if self.acc_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.acc_obs_name} not found; add it as an observation in your task.")
        acc = trajectory.obs[self.acc_obs_name]
        acc = acc[None]
        # First value will always be 0, because the acceleration is not changing.
        prev_acc = jnp.concatenate([acc[..., :1], acc[..., :-1]], axis=-1)
        # We multiply by ctrl_dt instead of dividing because we want the scale
        # for the penalty to be roughly the same magnitude as a velocity
        # penalty.
        jerk = (acc - prev_acc) * self.ctrl_dt
        return xax.get_norm(jerk, self.norm).mean(axis=-1).squeeze(0)


def joint_limits_validator(inst: "AvoidLimitsReward", attr: attrs.Attribute, value: xax.HashableArray) -> None:
    arr = value.array
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Joint range must have shape (n_joints, 2), got {arr.shape}")
    if not jnp.all(arr[..., 0] <= arr[..., 1]):
        raise ValueError(f"Joint range must be sorted, got {arr}")
    if not arr.dtype == jnp.float32:
        raise ValueError(f"Joint range must be a float array, got {arr.dtype}")


def joint_limited_validator(inst: "AvoidLimitsReward", attr: attrs.Attribute, value: xax.HashableArray) -> None:
    arr = value.array
    if arr.ndim != 1:
        raise ValueError(f"Joint limited must have shape (n_joints,), got {arr.shape}")
    if arr.dtype != jnp.bool_:
        raise ValueError(f"Joint limited must be a boolean array, got {arr.dtype}")


@attrs.define(frozen=True, kw_only=True)
class AvoidLimitsReward(Reward):
    """Reward for being too close to the joint limits."""

    joint_limits: xax.HashableArray = attrs.field(validator=joint_limits_validator)
    joint_limited: xax.HashableArray = attrs.field(validator=joint_limited_validator)

    def __call__(self, trajectory: Trajectory) -> Array:
        joint_pos = trajectory.qpos[..., 7:]
        joint_limits = self.joint_limits.array
        joint_limited = self.joint_limited.array
        in_bounds = (joint_pos > joint_limits[..., 0]) & (joint_pos < joint_limits[..., 1])
        reward = jnp.where(joint_limited, in_bounds, 0)
        return reward.all(axis=-1).astype(trajectory.qpos.dtype)

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        scale: float,
        padding: float = 0.05,
    ) -> Self:
        joint_range = model.jnt_range[1:].astype(jnp.float32)
        joint_min = joint_range[..., 0]
        joint_max = joint_range[..., 1]
        joint_padding = (joint_max - joint_min) * padding
        joint_min = joint_min + joint_padding
        joint_max = joint_max - joint_padding

        return cls(
            joint_limits=xax.hashable_array(jnp.stack([joint_min, joint_max], axis=-1)),
            joint_limited=xax.hashable_array(model.jnt_limited[1:].astype(jnp.bool_)),
            scale=scale,
        )


def joint_threshold_validator(
    inst: "ActionNearPositionPenalty",
    attr: attrs.Attribute,
    value: xax.HashableArray,
) -> None:
    arr = value.array
    if arr.ndim != 1:
        raise ValueError(f"Joint threshold must have shape (n_joints,), got {arr.shape}")
    if arr.dtype != jnp.float32:
        raise ValueError(f"Joint threshold must be a float array, got {arr.dtype}")


@attrs.define(frozen=True, kw_only=True)
class ActionNearPositionPenalty(Reward):
    """Penalizes the action for being too far from the target position.

    Note that this penalty only makes sense if you are using a position
    controller model, where actions correspond to positions.
    """

    joint_threshold: xax.HashableArray = attrs.field(validator=joint_threshold_validator)

    def __call__(self, trajectory: Trajectory) -> Array:
        current_position = trajectory.qpos[..., 7:]
        action = trajectory.action
        out_of_bounds = jnp.abs(current_position - action) > self.joint_threshold.array
        return out_of_bounds.astype(trajectory.qpos.dtype).mean(axis=-1)

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        scale: float,
        threshold: float = 0.25,
    ) -> Self:
        joint_range = model.jnt_range[1:].astype(jnp.float32)
        joint_min = joint_range[..., 0]
        joint_max = joint_range[..., 1]
        joint_threshold = (joint_max - joint_min) * threshold

        return cls(
            joint_threshold=xax.hashable_array(joint_threshold),
            scale=scale,
        )


@attrs.define(frozen=True, kw_only=True)
class FeetLinearVelocityTrackingPenalty(Reward):
    """Explicit penalty for tracking the linear velocity of the feet.

    This reward provides an explicit penalty to incentivize the robot to move
    it's feet in the direction of the velocity command.

    This penalty expects a reference linear velocity command, as well as the
    feet velocity observations.
    """

    ctrl_dt: float = attrs.field()
    command_name: str = attrs.field()
    obs_name: str = attrs.field(default="feet_position_observation")
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def __call__(self, trajectory: Trajectory) -> Array:
        cmd = trajectory.command[self.command_name]
        chex.assert_shape(cmd, (..., 2))
        lin_vel_x_cmd = cmd[..., 0]
        lin_vel_y_cmd = cmd[..., 1]

        obs = trajectory.obs[self.obs_name]
        chex.assert_shape(obs, (..., 2, 3))

        def get_vel_from_pos(pos: Array) -> Array:
            next_pos = jnp.concatenate([pos[..., 1:], pos[..., -1:]], axis=-1)
            return (next_pos - pos) / self.ctrl_dt

        left_vel_x = get_vel_from_pos(obs[..., 0, 0])
        left_vel_y = get_vel_from_pos(obs[..., 0, 1])
        right_vel_x = get_vel_from_pos(obs[..., 1, 0])
        right_vel_y = get_vel_from_pos(obs[..., 1, 1])

        # Mean of the two foot velocities should be close to the command.
        lin_vel_x_mean = (left_vel_x + right_vel_x) / 2
        lin_vel_y_mean = (left_vel_y + right_vel_y) / 2

        lin_vel_x_penalty = xax.get_norm(lin_vel_x_mean - lin_vel_x_cmd, self.norm)
        lin_vel_y_penalty = xax.get_norm(lin_vel_y_mean - lin_vel_y_cmd, self.norm)
        penalty = lin_vel_x_penalty + lin_vel_y_penalty

        # Don't penalize after falling over.
        penalty = jnp.where(trajectory.done, 0.0, penalty)

        return penalty


@attrs.define(frozen=True, kw_only=True)
class FeetFlatReward(Reward):
    """Reward for keeping the feet parallel to the relevant plane."""

    obs_name: str = attrs.field(default="feet_orientation_observation")
    plane: tuple[float, float, float] = attrs.field(default=(0.0, 0.0, 1.0))
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def __call__(self, trajectory: Trajectory) -> Array:
        feet_quat = trajectory.obs[self.obs_name]
        chex.assert_shape(feet_quat, (..., 2, 4))
        unit_vec = jnp.array(self.plane, dtype=feet_quat.dtype)
        unit_vec = xax.rotate_vector_by_quat(unit_vec, feet_quat)
        unit_vec_x, unit_vec_y, unit_vec_z = unit_vec[..., 0], unit_vec[..., 1], unit_vec[..., 2]

        # Z should be 1, and X and Y should be 0.
        return (
            xax.get_norm(unit_vec_z, self.norm)
            - xax.get_norm(unit_vec_x, self.norm)
            - xax.get_norm(unit_vec_y, self.norm)
        ).min(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class FeetPhaseReward(Reward):
    """Reward for tracking the desired foot height."""

    ctrl_dt: float = attrs.field()
    gait_freq: float = attrs.field(default=1.5)
    feet_pos_obs_name: str = attrs.field(default="feet_position_observation")
    foot_pos_obs_idx: CartesianIndex = attrs.field(default="z")
    max_foot_height: float = attrs.field(default=0.12)
    foot_default_height: float = attrs.field(default=0.04)

    def __call__(self, trajectory: Trajectory) -> Array:
        if self.feet_pos_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.feet_pos_obs_name} not found; add it as an observation in your task.")
        foot_pos = trajectory.obs[self.feet_pos_obs_name]
        chex.assert_shape(foot_pos, (..., 2, None))

        # Derives the gait phase as a function of time.
        t = jnp.arange(trajectory.done.shape[-1]) * self.ctrl_dt
        gait_phase = 2 * jnp.pi * self.gait_freq * t
        gait_phase = jnp.mod(gait_phase + jnp.pi, 2 * jnp.pi) - jnp.pi
        phase = jnp.stack([gait_phase, gait_phase + jnp.pi], axis=-1)

        foot_idx = cartesian_index_to_dim(self.foot_pos_obs_idx)
        foot_z_tf = foot_pos[..., foot_idx]
        ideal_z_tf = self.gait_phase(phase, swing_height=self.max_foot_height)
        ideal_z_tf = ideal_z_tf + self.foot_default_height

        error = jnp.sum(jnp.square(foot_z_tf - ideal_z_tf), axis=-1)
        reward = jnp.exp(-error / 0.01)

        return reward

    def gait_phase(
        self,
        phi: Array | float,
        swing_height: Array | float = 0.08,
    ) -> Array:
        """Interpolation logic for the gait phase with clear separation."""
        stance_phase = phi > 0
        swing_phase = ~stance_phase  # phi <= 0

        # Calculate s for all elements
        s = (phi + jnp.pi) / jnp.pi
        s = jnp.clip(s, 0, 1)

        # Calculate potential Z values for all elements
        z_rising = xax.cubic_bezier_interpolation(0.0, swing_height, 2.0 * s)
        z_falling = xax.cubic_bezier_interpolation(swing_height, 0.0, 2.0 * s - 1.0)
        potential_swing_z = jnp.where(s <= 0.5, z_rising, z_falling)

        # Calculate the final Z value using where based on swing phase
        final_z = jnp.where(swing_phase, potential_swing_z, 0.0)

        return final_z


@attrs.define(frozen=True, kw_only=True)
class FeetNoContactReward(Reward):
    """Reward for keeping the feet off the ground.

    This reward incentivizes the robot to keep at least one foot off the ground
    for at least `window_size` steps at a time. If the foot touches the ground
    again within `window_size` steps, the reward for the entire "off the ground"
    period is reset to 0.
    """

    window_size: int = attrs.field()
    obs_name: str = attrs.field(default="feet_contact_observation")

    def __call__(self, trajectory: Trajectory) -> Array:
        feet_contact = trajectory.obs[self.obs_name]
        chex.assert_shape(feet_contact, (..., 2))

        def count_scan_fn(carry: Array, contact: Array) -> tuple[Array, Array]:
            carry = jnp.where(contact, 0, carry + 1)
            return carry, carry

        _, counts = jax.lax.scan(count_scan_fn, jnp.zeros_like(feet_contact[0]), feet_contact, reverse=True)

        def reward_scan_fn(carry: Array, counts: Array) -> tuple[Array, Array]:
            carry = jnp.where(counts == 0, 0, jnp.where(carry == 0, counts, carry))
            return carry, carry

        _, counts = jax.lax.scan(reward_scan_fn, counts[0], counts)

        no_contact = counts >= self.window_size
        return no_contact.any(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class CartesianBodyTargetReward(Reward):
    """Rewards the closeness of the body to the target position."""

    tracked_body_idx: int = attrs.field()
    base_body_idx: int = attrs.field()
    command_name: str = attrs.field()
    sensitivity: float = attrs.field()
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def __call__(self, trajectory: Trajectory) -> Array:
        body_pos = trajectory.xpos[..., self.tracked_body_idx, :] - trajectory.xpos[..., self.base_body_idx, :]
        target_pos = trajectory.command[self.command_name]
        return jnp.exp(-xax.get_norm(body_pos - target_pos, self.norm) * self.sensitivity).mean(axis=-1)

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        command_name: str,
        tracked_body_name: str,
        base_body_name: str,
        norm: xax.NormType = "l2",
        scale: float = 1.0,
        sensitivity: float = 1.0,
    ) -> Self:
        body_idx = get_body_data_idx_from_name(model, tracked_body_name)
        base_idx = get_body_data_idx_from_name(model, base_body_name)
        return cls(
            tracked_body_idx=body_idx,
            base_body_idx=base_idx,
            norm=norm,
            scale=scale,
            sensitivity=sensitivity,
            command_name=command_name,
        )


@attrs.define(frozen=True, kw_only=True)
class CartesianBodyTargetVectorReward(Reward):
    """Rewards the alignment of the body's velocity vector to the direction of the target."""

    tracked_body_idx: int = attrs.field()
    base_body_idx: int = attrs.field()
    command_name: str = attrs.field()
    dt: float = attrs.field()
    normalize_velocity: bool = attrs.field()
    distance_threshold: float = attrs.field()
    epsilon: float = attrs.field(default=1e-6)

    def __call__(self, trajectory: Trajectory) -> Array:
        body_pos_TL = trajectory.xpos[..., self.tracked_body_idx, :] - trajectory.xpos[..., self.base_body_idx, :]

        body_pos_right_shifted_TL = jnp.roll(body_pos_TL, shift=1, axis=0)

        # Zero out the first velocity
        body_pos_right_shifted_TL = body_pos_right_shifted_TL.at[0].set(body_pos_TL[0])

        body_vel_TL = (body_pos_TL - body_pos_right_shifted_TL) / self.dt

        target_vector = trajectory.command[self.command_name] - body_pos_TL
        normalized_target_vector = target_vector / (
            jnp.linalg.norm(target_vector, axis=-1, keepdims=True) + self.epsilon
        )

        # Threshold to only apply reward to the body when it is far from the target.
        distance_scalar = jnp.linalg.norm(target_vector, axis=-1)
        far_from_target = distance_scalar > self.distance_threshold

        if self.normalize_velocity:
            normalized_body_vel = body_vel_TL / (jnp.linalg.norm(body_vel_TL, axis=-1, keepdims=True) + self.epsilon)
            original_products = normalized_body_vel * normalized_target_vector
        else:
            original_products = body_vel_TL * normalized_target_vector

        # This will give maximum reward if near the target (and velocity is normalized)
        return jnp.where(far_from_target, jnp.sum(original_products, axis=-1), 1.1)

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        command_name: str,
        tracked_body_name: str,
        base_body_name: str,
        dt: float,
        normalize_velocity: bool = True,
        scale: float = 1.0,
        epsilon: float = 1e-6,
        distance_threshold: float = 0.1,
    ) -> Self:
        body_idx = get_body_data_idx_from_name(model, tracked_body_name)
        base_idx = get_body_data_idx_from_name(model, base_body_name)
        return cls(
            tracked_body_idx=body_idx,
            base_body_idx=base_idx,
            scale=scale,
            command_name=command_name,
            dt=dt,
            normalize_velocity=normalize_velocity,
            epsilon=epsilon,
            distance_threshold=distance_threshold,
        )


@attrs.define(frozen=True, kw_only=True)
class CartesianBodyTargetPenalty(Reward):
    """Penalizes larger distances between the body and the target position."""

    tracked_body_idx: int = attrs.field()
    base_body_idx: int = attrs.field()
    command_name: str = attrs.field()
    norm: xax.NormType = attrs.field()

    def __call__(self, trajectory: Trajectory) -> Array:
        body_pos = trajectory.xpos[..., self.tracked_body_idx, :] - trajectory.xpos[..., self.base_body_idx, :]
        target_pos = trajectory.command[self.command_name]
        return xax.get_norm(body_pos - target_pos, self.norm).mean(axis=-1)

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        command_name: str,
        tracked_body_name: str,
        base_body_name: str,
        norm: xax.NormType = "l2",
        scale: float = 1.0,
    ) -> Self:
        body_idx = get_body_data_idx_from_name(model, tracked_body_name)
        base_idx = get_body_data_idx_from_name(model, base_body_name)
        return cls(
            tracked_body_idx=body_idx,
            base_body_idx=base_idx,
            norm=norm,
            scale=scale,
            command_name=command_name,
        )


@attrs.define(frozen=True, kw_only=True)
class ContinuousCartesianBodyTargetReward(Reward):
    """Rewards the closeness of the body to the target position more for the longer it has been doing so."""

    tracked_body_idx: int = attrs.field()
    base_body_idx: int = attrs.field()
    command_name: str = attrs.field()
    norm: xax.NormType = attrs.field()
    sensitivity: float = attrs.field()
    threshold: float = attrs.field()
    time_bonus_scale: float = attrs.field()
    time_sensitivity: float = attrs.field()

    def __call__(self, trajectory: Trajectory) -> Array:
        body_pos = trajectory.xpos[..., self.tracked_body_idx, :] - trajectory.xpos[..., self.base_body_idx, :]
        target_pos = trajectory.command[self.command_name]

        error = xax.get_norm(body_pos - target_pos, self.norm)
        base_reward = jnp.exp(-error * self.sensitivity)
        under_threshold = error < self.threshold

        def count_scan_fn(carry: Array, x: Array) -> tuple[Array, Array]:
            x = x.astype(jnp.int32)
            # Reset counter to 0 if not under threshold, otherwise increment
            count = jnp.where(x, carry + 1, 0)
            return count, count

        _, consecutive_steps = jax.lax.scan(
            count_scan_fn, init=jnp.zeros_like(under_threshold[0], dtype=jnp.int32), xs=under_threshold
        )

        # time_bonus = jnp.exp(consecutive_steps * self.time_sensitivity) * self.time_bonus_scale
        time_bonus = consecutive_steps * self.time_bonus_scale
        return (base_reward + time_bonus).mean(axis=-1)

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        command_name: str,
        tracked_body_name: str,
        base_body_name: str,
        norm: xax.NormType = "l2",
        scale: float = 1.0,
        sensitivity: float = 1.0,
        time_sensitivity: float = 0.01,
        threshold: float = 0.25,
        time_bonus_scale: float = 0.1,
    ) -> Self:
        body_idx = get_body_data_idx_from_name(model, tracked_body_name)
        base_idx = get_body_data_idx_from_name(model, base_body_name)
        return cls(
            tracked_body_idx=body_idx,
            base_body_idx=base_idx,
            norm=norm,
            scale=scale,
            sensitivity=sensitivity,
            time_sensitivity=time_sensitivity,
            command_name=command_name,
            threshold=threshold,
            time_bonus_scale=time_bonus_scale,
        )


@attrs.define(frozen=True, kw_only=True)
class GlobalBodyQuaternionReward(Reward):
    """Rewards the closeness of the body orientation to the target quaternion."""

    tracked_body_idx: int = attrs.field()
    base_body_idx: int = attrs.field()
    command_name: str = attrs.field()
    norm: xax.NormType = attrs.field()
    sensitivity: float = attrs.field()

    def __call__(self, trajectory: Trajectory) -> Array:
        body_quat = trajectory.xquat[..., self.tracked_body_idx, :]
        target_quat = trajectory.command[self.command_name]

        is_null = jnp.all(jnp.isclose(target_quat, 0.0), axis=-1, keepdims=True)

        err = jnp.where(is_null, 0.0, xax.get_norm(body_quat - target_quat, self.norm))
        return jnp.exp(-err * self.sensitivity).mean(axis=-1)

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        command_name: str,
        tracked_body_name: str,
        base_body_name: str,
        norm: xax.NormType = "l2",
        scale: float = 1.0,
        sensitivity: float = 1.0,
    ) -> Self:
        body_idx = get_body_data_idx_from_name(model, tracked_body_name)
        base_idx = get_body_data_idx_from_name(model, base_body_name)
        return cls(
            tracked_body_idx=body_idx,
            base_body_idx=base_idx,
            norm=norm,
            scale=scale,
            sensitivity=sensitivity,
            command_name=command_name,
        )
