"""Defines a base interface for defining reward functions."""

__all__ = [
    "MonotonicFn",
    "norm_to_reward",
    "Reward",
    "StayAliveReward",
    "LinearVelocityPenalty",
    "AngularVelocityPenalty",
    "JointVelocityPenalty",
    "BaseHeightReward",
    "BaseHeightRangeReward",
    "ActionSmoothnessPenalty",
    "ActuatorForcePenalty",
    "BaseJerkZPenalty",
    "ActuatorJerkPenalty",
    "AvoidLimitsReward",
    "ObservationMeanPenalty",
    "ActionNearPositionPenalty",
    "FeetLinearVelocityTrackingPenalty",
    "FeetFlatReward",
    "FeetNoContactReward",
    "PositionTrackingReward",
    "JoystickReward",
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
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.types import PhysicsModel, Trajectory
from ksim.utils.mujoco import get_body_data_idx_from_name
from ksim.utils.types import (
    CartesianIndex,
    cartesian_index_to_dim,
    dimension_index_validator,
    norm_validator,
)
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
    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        """Get the reward for a single trajectory.

        You may assume that the dimensionality is (time, *leaf_dims) accross all
        leaves of the trajectory.

        Args:
            trajectory: The trajectory to get the reward for.
            reward_carry: The reward carry for the trajectory.

        Returns:
            An array of shape (time, *leaf_dims) containing the reward for each
            timestep.
        """

    def initial_carry(self, rng: PRNGKeyArray) -> PyTree:
        """Initial reward carry for the trajectory, optionally overridable.

        Some rewards require information from the same episode in a previous
        rollout. E.g. a reward could require the last time the robot was in
        contact with the ground. This function simply returns the initial reward
        carry, which is `None` by default.
        """
        return None

    def get_markers(self) -> Collection[Marker]:
        """Get the markers for the reward, optionally overridable."""
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
    success_reward: float = attrs.field(default=0.0)

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        reward = jnp.where(
            trajectory.done, jnp.where(trajectory.success, self.success_reward, -1.0), 1.0 / self.balance
        )
        return reward, None


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityPenalty(Reward):
    """Penalty for how fast the robot is moving in the z-direction."""

    index: CartesianIndex = attrs.field(validator=dimension_index_validator)
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        dim = cartesian_index_to_dim(self.index)
        lin_vel = trajectory.qvel[..., dim]
        return xax.get_norm(lin_vel, self.norm), None

    def get_name(self) -> str:
        return f"{self.index}_{super().get_name()}"


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityPenalty(Reward):
    """Penalty for how fast the robot is rotating in the xy-plane."""

    index: CartesianIndex = attrs.field(validator=dimension_index_validator)
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        dim = cartesian_index_to_dim(self.index) + 3
        ang_vel = trajectory.qvel[..., dim]
        return xax.get_norm(ang_vel, self.norm), None

    def get_name(self) -> str:
        return f"{self.index}_{super().get_name()}"


@attrs.define(frozen=True, kw_only=True)
class JointVelocityPenalty(Reward):
    """Penalty for how fast the joint angular velocities are changing."""

    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    freejoint_first: bool = attrs.field(default=True)

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        if self.freejoint_first:
            joint_vel = trajectory.qvel[..., 6:]
            return xax.get_norm(joint_vel, self.norm).mean(axis=-1), None
        else:
            return xax.get_norm(trajectory.qvel, self.norm).mean(axis=-1), None


@attrs.define(frozen=True, kw_only=True)
class BaseHeightReward(Reward):
    """Penalty for deviating from the base height target."""

    height_target: float = attrs.field()
    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    temp: float = attrs.field(default=1.0)
    monotonic_fn: MonotonicFn = attrs.field(default="inv")

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        base_height = trajectory.qpos[..., 2]
        reward = norm_to_reward(xax.get_norm(base_height - self.height_target, self.norm), self.temp, self.monotonic_fn)
        return reward, None


@attrs.define(frozen=True, kw_only=True)
class BaseHeightRangeReward(Reward):
    """Incentivizes keeping the base height within a certain range."""

    z_lower: float = attrs.field()
    z_upper: float = attrs.field()
    dropoff: float = attrs.field()

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        base_height = trajectory.qpos[..., 2]
        too_low = self.z_lower - base_height
        too_high = base_height - self.z_upper
        reward = (1.0 - jnp.maximum(too_low, too_high).clip(min=0.0) * self.dropoff).clip(min=0.0)
        return reward, None


@attrs.define(frozen=True, kw_only=True)
class ActionSmoothnessPenalty(Reward):
    """Penalty for large changes between consecutive actions."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
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
        reward = xax.get_norm(action_deltas, self.norm).mean(axis=-1)
        return reward, None


@attrs.define(frozen=True, kw_only=True)
class ActuatorForcePenalty(Reward):
    """Penalty for high actuator forces."""

    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    observation_name: str = attrs.field(default="actuator_force_observation")

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        if self.observation_name not in trajectory.obs:
            raise ValueError(f"Observation {self.observation_name} not found; add it as an observation in your task.")
        reward = xax.get_norm(trajectory.obs[self.observation_name], self.norm).mean(axis=-1)
        return reward, None


@attrs.define(frozen=True, kw_only=True)
class BaseJerkZPenalty(Reward):
    """Penalty for high base jerk."""

    ctrl_dt: float = attrs.field()
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)
    acc_obs_name: str = attrs.field(default="base_linear_acceleration_observation")

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
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
        reward = xax.get_norm(jerk_z, self.norm).squeeze(0)
        return reward, None


@attrs.define(frozen=True, kw_only=True)
class ActuatorJerkPenalty(Reward):
    """Penalty for high actuator jerks."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)
    acc_obs_name: str = attrs.field(default="actuator_acceleration_observation")
    ctrl_dt: float = attrs.field()

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
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
        reward = xax.get_norm(jerk, self.norm).mean(axis=-1).squeeze(0)
        return reward, None


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

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        joint_pos = trajectory.qpos[..., 7:]
        joint_limits = self.joint_limits.array
        joint_limited = self.joint_limited.array
        in_bounds = (joint_pos > joint_limits[..., 0]) & (joint_pos < joint_limits[..., 1])
        reward = jnp.where(joint_limited, in_bounds, 0)
        return reward.all(axis=-1).astype(trajectory.qpos.dtype), None

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
class ObservationMeanPenalty(Reward):
    """Penalty for the mean of an observation."""

    observation_name: str = attrs.field()

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        reward = trajectory.obs[self.observation_name].mean(axis=-1)
        return reward, None

    def get_name(self) -> str:
        return f"{super().get_name()}_{self.observation_name}"


@attrs.define(frozen=True, kw_only=True)
class ActionNearPositionPenalty(Reward):
    """Penalizes the action for being too far from the target position.

    Note that this penalty only makes sense if you are using a position
    controller model, where actions correspond to positions.
    """

    joint_threshold: xax.HashableArray = attrs.field(validator=joint_threshold_validator)

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        current_position = trajectory.qpos[..., 7:]
        action = trajectory.action
        out_of_bounds = jnp.abs(current_position - action) > self.joint_threshold.array
        return out_of_bounds.astype(trajectory.qpos.dtype).mean(axis=-1), None

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

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
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

        return penalty, None


@attrs.define(frozen=True, kw_only=True)
class FeetFlatReward(Reward):
    """Reward for keeping the feet parallel to the relevant plane."""

    obs_name: str = attrs.field(default="feet_orientation_observation")
    plane: tuple[float, float, float] = attrs.field(default=(0.0, 0.0, 1.0))
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
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
        ).min(axis=-1), None


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

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
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
        return no_contact.any(axis=-1), None


@attrs.define(frozen=True, kw_only=True)
class PositionTrackingReward(Reward):
    """Rewards the closeness of the body to the target position more for the longer it has been doing so."""

    tracked_body_idx: int = attrs.field()
    base_body_idx: int = attrs.field()
    command_name: str = attrs.field()
    body_name: str = attrs.field()
    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    temp: float = attrs.field(default=1.0)
    monotonic_fn: MonotonicFn = attrs.field(default="inv")

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        body_pos = trajectory.xpos[..., self.tracked_body_idx, :]
        base_pos = trajectory.xpos[..., self.base_body_idx, :]
        target_pos = trajectory.command[self.command_name][..., :3]
        error = xax.get_norm((body_pos - base_pos) - target_pos, self.norm).sum(-1)
        reward = norm_to_reward(error, self.temp, self.monotonic_fn)
        return reward, None

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        command_name: str,
        tracked_body_name: str,
        base_body_name: str,
        norm: xax.NormType = "l1",
        temp: float = 1.0,
        monotonic_fn: MonotonicFn = "inv",
        scale: float = 1.0,
    ) -> Self:
        body_idx = get_body_data_idx_from_name(model, tracked_body_name)
        base_body_idx = get_body_data_idx_from_name(model, base_body_name)
        return cls(
            tracked_body_idx=body_idx,
            base_body_idx=base_body_idx,
            norm=norm,
            command_name=command_name,
            body_name=tracked_body_name,
            scale=scale,
            temp=temp,
            monotonic_fn=monotonic_fn,
        )

    def get_name(self) -> str:
        return f"{self.body_name}_{super().get_name()}"


@attrs.define(frozen=True, kw_only=True)
class JoystickReward(Reward):
    """Defines a reward for following joystick controls.

    Command mapping:

        0 = stand still
        1 = walk forward
        2 = walk backward
        3 = turn left
        4 = turn right
    """

    linear_velocity_clip_max: float = attrs.field(validator=attrs.validators.gt(0.0))
    angular_velocity_clip_max: float = attrs.field(validator=attrs.validators.gt(0.0))
    command_name: str = attrs.field(default="joystick_command")
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)
    norm_penalty: float = attrs.field(default=0.01)

    def __call__(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        command = trajectory.command[self.command_name]
        chex.assert_shape(command, (..., 1))
        command = command.squeeze(-1)

        # Gets the velocity of the robot.
        xvel = trajectory.qvel[..., 0]
        yvel = trajectory.qvel[..., 1]
        zvel = trajectory.qvel[..., 2]
        dxvel = trajectory.qvel[..., 4]
        dyvel = trajectory.qvel[..., 5]
        dzvel = trajectory.qvel[..., 6]

        reward = jnp.where(
            command == 1,
            # Forward
            xvel.clip(max=self.linear_velocity_clip_max)
            - xax.get_norm(
                jnp.stack([yvel, zvel, dxvel, dyvel, dzvel], axis=-1),
                self.norm,
            ).mean(-1)
            * self.norm_penalty,
            jnp.where(
                command == 2,
                # Backward
                (-xvel).clip(max=self.linear_velocity_clip_max)
                - xax.get_norm(
                    jnp.stack([yvel, zvel, dxvel, dyvel, dzvel], axis=-1),
                    self.norm,
                ).mean(-1)
                * self.norm_penalty,
                jnp.where(
                    command == 3,
                    # Turn left
                    dzvel.clip(max=self.angular_velocity_clip_max)
                    - xax.get_norm(
                        jnp.stack([xvel, yvel, zvel, dxvel, dyvel], axis=-1),
                        self.norm,
                    ).mean(-1)
                    * self.norm_penalty,
                    jnp.where(
                        command == 4,
                        # Turn right
                        (-dzvel).clip(max=self.angular_velocity_clip_max)
                        - xax.get_norm(
                            jnp.stack([xvel, yvel, zvel, dxvel, dyvel], axis=-1),
                            self.norm,
                        ).mean(-1)
                        * self.norm_penalty,
                        # Stationary penalty.
                        1.0
                        - xax.get_norm(
                            jnp.stack([xvel, yvel, zvel, dxvel, dyvel, dzvel], axis=-1),
                            self.norm,
                        ).mean(-1)
                        * self.norm_penalty,
                    ),
                ),
            ),
        )
        return reward, None
