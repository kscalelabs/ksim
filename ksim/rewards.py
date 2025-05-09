"""Defines a base interface for defining reward functions."""

__all__ = [
    "MonotonicFn",
    "norm_to_reward",
    "Reward",
    "StatefulReward",
    "StayAliveReward",
    "LinearVelocityReward",
    "LinearVelocityPenalty",
    "NaiveForwardReward",
    "AngularVelocityReward",
    "AngularVelocityPenalty",
    "XYAngularVelocityPenalty",
    "JointVelocityPenalty",
    "BaseHeightReward",
    "BaseHeightRangeReward",
    "ActionSmoothnessPenalty",
    "ActuatorForcePenalty",
    "ActuatorRelativeForcePenalty",
    "BaseJerkZPenalty",
    "ActuatorJerkPenalty",
    "ActionInBoundsReward",
    "AvoidLimitsPenalty",
    "ActionNearPositionPenalty",
    "JointDeviationPenalty",
    "FeetLinearVelocityTrackingPenalty",
    "FlatBodyReward",
    "FeetNoContactReward",
    "PositionTrackingReward",
    "UprightReward",
    "HeadingTrackingReward",
    "HeadingVelocityReward",
    "JoystickPenalty",
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
from ksim.utils.mujoco import (
    get_body_data_idx_from_name,
    get_heading,
    get_qpos_data_idxs_by_name,
    get_velocity_in_frame,
)
from ksim.utils.types import (
    CartesianIndex,
    cartesian_index_to_dim,
    dimension_index_tuple_validator,
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
    scale_by_curriculum: bool = attrs.field(default=False)

    @abstractmethod
    def get_reward(self, trajectory: Trajectory) -> Array:
        """Get the reward for a single trajectory.

        Args:
            trajectory: The trajectory to get the reward for.

        Returns:
            An array of shape (time) containing the reward for each timestep.
        """

    def get_markers(self) -> Collection[Marker]:
        """Get the markers for the reward, optionally overridable."""
        return []

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def reward_name(self) -> str:
        return self.get_name()


@attrs.define(frozen=True, kw_only=True)
class StatefulReward(Reward):
    """Reward that requires state from the previous timestep."""

    def get_reward(self, trajectory: Trajectory) -> Array:
        raise NotImplementedError("StatefulReward should use `get_reward_stateful` instead.")

    @abstractmethod
    def initial_carry(self, rng: PRNGKeyArray) -> PyTree:
        """Initial reward carry for the trajectory, optionally overridable.

        Some rewards require information from the same episode in a previous
        rollout. E.g. a reward could require the last time the robot was in
        contact with the ground. This function simply returns the initial reward
        carry, which is `None` by default.
        """
        return None

    @abstractmethod
    def get_reward_stateful(self, trajectory: Trajectory, reward_carry: PyTree) -> tuple[Array, PyTree]:
        """Get the reward for a single trajectory.

        This is the same as `get_reward`, but it also takes in the reward carry
        from the previous timestep.

        Args:
            trajectory: The trajectory to get the reward for.
            reward_carry: The reward carry from the previous timestep.

        Returns:
            A tuple containing the reward for the current timestep and the
            reward carry for the next timestep.
        """


@attrs.define(frozen=True, kw_only=True)
class StayAliveReward(Reward):
    """Reward for staying alive.

    This provides a reward for staying alive, with a negative penalty on
    termination. These values are balanced by the balance parameter - a larger
    value will increase the relative penalty for termination.
    """

    balance: float = attrs.field(default=10.0)
    success_reward: float | None = attrs.field(default=None)

    def get_reward(self, trajectory: Trajectory) -> Array:
        reward = jnp.where(
            trajectory.done,
            jnp.where(
                trajectory.success,
                1.0 / self.balance if self.success_reward is None else self.success_reward,
                -1.0,
            ),
            1.0 / self.balance,
        )
        return reward


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityReward(Reward):
    """Penalty for how fast the robot is moving in the z-direction."""

    index: CartesianIndex | tuple[CartesianIndex, ...] = attrs.field(validator=dimension_index_tuple_validator)
    clip_min: float | None = attrs.field(default=None)
    clip_max: float | None = attrs.field(default=None)
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)
    in_robot_frame: bool = attrs.field(default=True)

    def get_reward(self, trajectory: Trajectory) -> Array:
        indices = self.index if isinstance(self.index, tuple) else (self.index,)
        dims = tuple(cartesian_index_to_dim(index) for index in indices)
        linvel = trajectory.qvel[..., :3]
        if self.in_robot_frame:
            # Same as reading from a velocimeter attached to base.
            linvel = xax.rotate_vector_by_quat(linvel, trajectory.qpos[..., 3:7], inverse=True)
        dimvel = linvel[..., dims].clip(self.clip_min, self.clip_max).mean(axis=-1)
        return xax.get_norm(dimvel, self.norm)

    def get_name(self) -> str:
        indices = self.index if isinstance(self.index, tuple) else (self.index,)
        return f"{''.join(indices)}_{super().get_name()}"


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityPenalty(LinearVelocityReward): ...


@attrs.define(frozen=True, kw_only=True)
class NaiveForwardReward(LinearVelocityReward):
    index: tuple[CartesianIndex, ...] = attrs.field(default=("x",), validator=dimension_index_tuple_validator)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityReward(Reward):
    """Penalty for how fast the robot is rotating in the xy-plane."""

    index: CartesianIndex | tuple[CartesianIndex, ...] = attrs.field(validator=dimension_index_tuple_validator)
    clip_min: float | None = attrs.field(default=None)
    clip_max: float | None = attrs.field(default=None)
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)
    in_robot_frame: bool = attrs.field(default=True)

    def get_reward(self, trajectory: Trajectory) -> Array:
        indices = self.index if isinstance(self.index, tuple) else (self.index,)
        dims = tuple(cartesian_index_to_dim(index) for index in indices)
        angvel = trajectory.qvel[..., 3:6]
        if self.in_robot_frame:
            angvel = xax.rotate_vector_by_quat(angvel, trajectory.qpos[..., 3:7], inverse=True)
        dimvel = angvel[..., dims].clip(self.clip_min, self.clip_max).mean(axis=-1)
        return xax.get_norm(dimvel, self.norm)

    def get_name(self) -> str:
        indices = self.index if isinstance(self.index, tuple) else (self.index,)
        return f"{''.join(indices)}_{super().get_name()}"


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityPenalty(AngularVelocityReward): ...


@attrs.define(frozen=True, kw_only=True)
class XYAngularVelocityPenalty(AngularVelocityReward):
    index: CartesianIndex | tuple[CartesianIndex, ...] = attrs.field(
        default=("x", "y"),
        validator=dimension_index_tuple_validator,
    )


@attrs.define(frozen=True, kw_only=True)
class JointVelocityPenalty(Reward):
    """Penalty for how fast the joint angular velocities are changing."""

    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    freejoint_first: bool = attrs.field(default=True)

    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.freejoint_first:
            joint_vel = trajectory.qvel[..., 6:]
            return xax.get_norm(joint_vel, self.norm).mean(axis=-1)
        else:
            return xax.get_norm(trajectory.qvel, self.norm).mean(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class BaseHeightReward(Reward):
    """Penalty for deviating from the base height target."""

    height_target: float = attrs.field()
    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    temp: float = attrs.field(default=1.0)
    monotonic_fn: MonotonicFn = attrs.field(default="inv")

    def get_reward(self, trajectory: Trajectory) -> Array:
        base_height = trajectory.qpos[..., 2]
        reward = norm_to_reward(xax.get_norm(base_height - self.height_target, self.norm), self.temp, self.monotonic_fn)
        return reward


@attrs.define(frozen=True, kw_only=True)
class BaseHeightRangeReward(Reward):
    """Incentivizes keeping the base height within a certain range."""

    z_lower: float = attrs.field()
    z_upper: float = attrs.field()
    dropoff: float = attrs.field()

    def get_reward(self, trajectory: Trajectory) -> Array:
        base_height = trajectory.qpos[..., 2]
        too_low = self.z_lower - base_height
        too_high = base_height - self.z_upper
        reward = (1.0 - jnp.maximum(too_low, too_high).clip(min=0.0) * self.dropoff).clip(min=0.0)
        return reward


@attrs.define(frozen=True, kw_only=True)
class ActionSmoothnessPenalty(Reward):
    """Penalty for large changes between consecutive actions."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
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
        return reward


@attrs.define(frozen=True, kw_only=True)
class ActuatorForcePenalty(Reward):
    """Penalty for high actuator forces."""

    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    observation_name: str = attrs.field(default="actuator_force_observation")

    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.observation_name not in trajectory.obs:
            raise ValueError(f"Observation {self.observation_name} not found; add it as an observation in your task.")
        reward = xax.get_norm(trajectory.obs[self.observation_name], self.norm).mean(axis=-1)
        return reward


@attrs.define(frozen=True, kw_only=True)
class ActuatorRelativeForcePenalty(Reward):
    """Same as ActuatorForcePenalty but scaled by the maximum force on each actuator."""

    magnitudes: tuple[float, ...] = attrs.field()
    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    observation_name: str = attrs.field(default="actuator_force_observation")

    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.observation_name not in trajectory.obs:
            raise ValueError(f"Observation {self.observation_name} not found; add it as an observation in your task.")
        reward = xax.get_norm(trajectory.obs[self.observation_name], self.norm)
        reward = reward / jnp.array(self.magnitudes)
        return reward.mean(axis=-1)

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        norm: xax.NormType = "l1",
        observation_name: str = "actuator_force_observation",
        scale: float = -1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        act_force_limited = jnp.array(model.jnt_actfrclimited)[..., 1:]
        if not act_force_limited.all().item():
            raise ValueError("Actuator force limits must be set for all actuators.")

        act_force = jnp.array(model.jnt_actfrcrange)[..., 1:, :]
        act_force_min, act_force_max = act_force[..., 0], act_force[..., 1]
        act_force_magnitude = (act_force_max - act_force_min) // 2
        chex.assert_shape(act_force_magnitude, (None,))
        magnitudes = tuple(act_force_magnitude.tolist())

        return cls(
            observation_name=observation_name,
            magnitudes=magnitudes,
            norm=norm,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class BaseJerkZPenalty(Reward):
    """Penalty for high base jerk."""

    ctrl_dt: float = attrs.field()
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)
    acc_obs_name: str = attrs.field(default="base_linear_acceleration_observation")

    def get_reward(self, trajectory: Trajectory) -> Array:
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
        return reward


@attrs.define(frozen=True, kw_only=True)
class ActuatorJerkPenalty(Reward):
    """Penalty for high actuator jerks."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)
    acc_obs_name: str = attrs.field(default="actuator_acceleration_observation")
    ctrl_dt: float = attrs.field()

    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.acc_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.acc_obs_name} not found; add it as an observation in your task.")
        acc = trajectory.obs[self.acc_obs_name]
        acc = acc[None]
        # First value will always be 0, because the acceleration is not changing.
        prev_acc = jnp.concatenate([acc[..., :1], acc[..., :-1]], axis=-1)
        # We multiply by ctrl_dt here we want the scale for the penalty to be
        # roughly the same magnitude as a velocity penalty.
        jerk = (acc - prev_acc) * self.ctrl_dt * self.ctrl_dt
        reward = xax.get_norm(jerk, self.norm).mean(axis=-1).squeeze(0)
        return reward


def joint_limits_validator(inst: "ActionInBoundsReward", attr: attrs.Attribute, value: xax.HashableArray) -> None:
    arr = value.array
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Joint range must have shape (n_joints, 2), got {arr.shape}")
    if not jnp.all(arr[..., 0] <= arr[..., 1]):
        raise ValueError(f"Joint range must be sorted, got {arr}")
    if not arr.dtype == jnp.float32:
        raise ValueError(f"Joint range must be a float array, got {arr.dtype}")


@attrs.define(frozen=True, kw_only=True)
class ActionInBoundsReward(Reward):
    """Reward for the actions being within the joint limits.

    Note that this penalty only makes sense if you are using a position
    controller model, where actions correspond to positions.
    """

    joint_limits: xax.HashableArray = attrs.field(validator=joint_limits_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        action = trajectory.action
        joint_limits = self.joint_limits.array
        in_bounds = (action > joint_limits[..., 0]) & (action < joint_limits[..., 1])
        return in_bounds.all(axis=-1).astype(trajectory.qpos.dtype)

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        factor: float = 0.05,
        scale: float = 1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        jnt_range = jnp.array(model.jnt_range)
        jnt_limited = jnp.array(model.jnt_limited, dtype=jnp.bool_)
        jnt_min, jnt_max = jnt_range[..., 0], jnt_range[..., 1]
        jnt_diff = (jnt_max - jnt_min) * factor
        jnt_min = jnt_min - jnt_diff
        jnt_max = jnt_max + jnt_diff
        jnt_min = jnp.where(jnt_limited, jnt_min, -jnp.inf)
        jnt_max = jnp.where(jnt_limited, jnt_max, jnp.inf)
        joint_limits = jnp.stack([jnt_min, jnt_max], axis=-1)
        return cls(
            joint_limits=xax.hashable_array(joint_limits[..., 1:, :]),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class AvoidLimitsPenalty(ActionInBoundsReward):
    """Penalty for being too close to the joint limits."""

    joint_limits: xax.HashableArray = attrs.field(validator=joint_limits_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        joint_pos = trajectory.qpos[..., 7:]
        joint_limits = self.joint_limits.array
        penalty = -jnp.clip(joint_pos - joint_limits[..., 0], None, 0.0)
        penalty += jnp.clip(joint_pos - joint_limits[..., 1], 0.0, None)
        return jnp.sum(penalty, axis=-1)


@attrs.define(frozen=True, kw_only=True)
class ObservationMeanPenalty(Reward):
    """Penalty for the mean of an observation."""

    observation_name: str = attrs.field()

    def get_reward(self, trajectory: Trajectory) -> Array:
        reward = trajectory.obs[self.observation_name].mean(axis=-1)
        return reward

    def get_name(self) -> str:
        return f"{super().get_name()}_{self.observation_name}"


@attrs.define(frozen=True, kw_only=True)
class ActionNearPositionPenalty(Reward):
    """Penalizes the action for being too far from the target position.

    Note that this penalty only makes sense if you are using a position
    controller model, where actions correspond to positions.
    """

    joint_threshold: float = attrs.field(default=0.0, validator=attrs.validators.ge(0.0))
    backoff_scale: float = attrs.field(default=1.0)
    norm: xax.NormType = attrs.field(default="l2")

    def get_reward(self, trajectory: Trajectory) -> Array:
        current_position = trajectory.qpos[..., 7:]
        action = trajectory.action
        diff = xax.get_norm(current_position - action, self.norm)
        out_of_bounds = (diff - self.joint_threshold).clip(min=self.joint_threshold)
        return (out_of_bounds * self.backoff_scale).astype(trajectory.qpos.dtype).mean(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class JointDeviationPenalty(Reward):
    """Penalty for joint deviations from target positions."""

    norm: xax.NormType = attrs.field(default="l2")
    joint_indices: tuple[int, ...] = attrs.field()
    joint_targets: tuple[float, ...] = attrs.field()
    joint_weights: tuple[float, ...] | None = attrs.field(default=None)

    def get_reward(self, trajectory: Trajectory) -> Array:
        diff = (
            trajectory.qpos[..., jnp.array(self.joint_indices) + 7]
            - jnp.array(self.joint_targets)[jnp.array(self.joint_indices)]
        )

        if self.joint_weights is not None:
            diff *= jnp.array(self.joint_weights)

        penalty = xax.get_norm(diff, self.norm).sum(axis=-1)
        return penalty

    @classmethod
    def create(
        cls,
        physics_model: PhysicsModel,
        joint_names: tuple[str, ...],
        joint_targets: tuple[float, ...],
        scale: float = -1.0,
        joint_weights: tuple[float, ...] | None = None,
        scale_by_curriculum: bool = False,
    ) -> Self:
        joint_to_idx = get_qpos_data_idxs_by_name(physics_model)
        joint_indices = tuple([int(joint_to_idx[name][0]) - 7 for name in joint_names])
        return cls(
            joint_indices=joint_indices,
            joint_targets=joint_targets,
            joint_weights=joint_weights,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
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

    def get_reward(self, trajectory: Trajectory) -> Array:
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
class FlatBodyReward(Reward):
    """Reward for keeping the body parallel to the ground."""

    body_indices: tuple[int, ...] = attrs.field()
    plane: tuple[float, float, float] = attrs.field(default=(0.0, 0.0, 1.0))
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        body_quat = trajectory.xquat[..., self.body_indices, :]
        plane_vec = jnp.array(self.plane, dtype=body_quat.dtype)
        unit_vec = xax.rotate_vector_by_quat(plane_vec, body_quat, inverse=True)
        return jnp.einsum("...i,...i->...", unit_vec, plane_vec).mean(axis=-1)

    @classmethod
    def create(
        cls,
        physics_model: PhysicsModel,
        body_names: tuple[str, ...],
        plane: tuple[float, float, float] = (0.0, 0.0, 1.0),
        norm: xax.NormType = "l2",
        scale: float = 1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        return cls(
            body_indices=tuple([get_body_data_idx_from_name(physics_model, name) for name in body_names]),
            plane=plane,
            norm=norm,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


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

    def get_reward(self, trajectory: Trajectory) -> Array:
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
class PositionTrackingReward(Reward):
    """RewardState the closeness of the body to the target position more for the longer it has been doing so."""

    tracked_body_idx: int = attrs.field()
    base_body_idx: int = attrs.field()
    command_name: str = attrs.field()
    body_name: str = attrs.field()
    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    temp: float = attrs.field(default=1.0)
    monotonic_fn: MonotonicFn = attrs.field(default="inv")

    def get_reward(self, trajectory: Trajectory) -> Array:
        body_pos = trajectory.xpos[..., self.tracked_body_idx, :]
        base_pos = trajectory.xpos[..., self.base_body_idx, :]
        target_pos = trajectory.command[self.command_name][..., :3]
        error = xax.get_norm((body_pos - base_pos) - target_pos, self.norm).sum(-1)
        reward = norm_to_reward(error, self.temp, self.monotonic_fn)
        return reward

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
        scale_by_curriculum: bool = False,
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
            scale_by_curriculum=scale_by_curriculum,
            temp=temp,
            monotonic_fn=monotonic_fn,
        )

    def get_name(self) -> str:
        return f"{self.body_name}_{super().get_name()}"


@attrs.define(frozen=True, kw_only=True)
class UprightReward(Reward):
    """Reward for staying upright."""

    def get_reward(self, trajectory: Trajectory) -> Array:
        gravity = jnp.array([0.0, 0.0, 1.0])
        quat = trajectory.qpos[..., 3:7]
        return xax.rotate_vector_by_quat(gravity, quat, inverse=True)[..., 2]


@attrs.define(frozen=True, kw_only=True)
class HeadingTrackingReward(Reward):
    """Reward for tracking the heading vector."""

    index: CartesianIndex | tuple[CartesianIndex, ...] = attrs.field(
        default=("x", "y"),
        validator=dimension_index_tuple_validator,
    )
    command_name: str = attrs.field(default="start_quaternion_command")

    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.command_name not in trajectory.command:
            raise ValueError(f"Command {self.command_name} not found! Ensure that it is in the task.")
        target_quat = trajectory.command[self.command_name]
        chex.assert_shape(target_quat, (..., 4))

        # Gets the current heading vector along the relevant indices.
        indices = self.index if isinstance(self.index, tuple) else (self.index,)
        dims = tuple([cartesian_index_to_dim(index) for index in indices])
        target_heading = get_heading(target_quat)[..., dims]
        current_heading = get_heading(trajectory.qpos[..., 3:7])[..., dims]

        # Maximize the dot product between the current and target heading vectors.
        dot_product = jnp.einsum("...i,...i->...", current_heading, target_heading)
        return dot_product


@attrs.define(frozen=True, kw_only=True)
class HeadingVelocityReward(Reward):
    """Reward for moving in the heading vector direction."""

    target_velocity: float = attrs.field()
    index: CartesianIndex = attrs.field(default="x", validator=dimension_index_validator)
    flip_sign: bool = attrs.field(default=False)
    command_name: str = attrs.field(default="start_quaternion_command")

    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.command_name not in trajectory.command:
            raise ValueError(f"Command {self.command_name} not found! Ensure that it is in the task.")
        target_quat = trajectory.command[self.command_name]
        chex.assert_shape(target_quat, (..., 4))
        dim = cartesian_index_to_dim(self.index)
        heading_velocity = get_velocity_in_frame(target_quat, trajectory.qvel[..., :3])[..., dim]
        if self.flip_sign:
            heading_velocity = -heading_velocity
        return heading_velocity.clip(min=0.0, max=self.target_velocity)


@attrs.define(frozen=True, kw_only=True)
class JoystickPenalty(Reward):
    """Reward for tracking the joystick commands.

    This creates one big penalty which encourages the robot to follow the
    joystick command, in terms of it's orientation, linear velocity, and
    angular velocity.
    """

    translation_speed: float = attrs.field()
    rotation_speed: float = attrs.field()
    command_name: str = attrs.field(default="joystick_command")
    heading_reward_scale: float = attrs.field(default=0.1)
    lin_vel_penalty_scale: float = attrs.field(default=0.1)
    ang_vel_penalty_scale: float = attrs.field(default=0.1)
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.command_name not in trajectory.command:
            raise ValueError(f"Command {self.command_name} not found! Ensure that it is in the task.")
        joystick_cmd = trajectory.command[self.command_name]
        chex.assert_shape(joystick_cmd, (..., 11))

        # Splits command into one-hot encoded command and target quaternion.
        command_ohe = joystick_cmd[..., :7]
        target_quat = joystick_cmd[..., 7:]

        # Gets the current heading vector.
        target_heading = get_heading(target_quat)[..., (0, 1)]
        current_heading = get_heading(trajectory.qpos[..., 3:7])[..., (0, 1)]
        heading_reward = jnp.einsum("...i,...i->...", current_heading, target_heading) * self.heading_reward_scale

        qvel = trajectory.qvel[..., :6]
        linvel = qvel[..., :3]
        angvel = qvel[..., 3:]
        heading_vel = get_velocity_in_frame(target_quat, linvel)
        forward_vel = heading_vel[..., 0]
        left_vel = heading_vel[..., 1]
        rotation_vel = angvel[..., 2]  # Rotation about the Z axis.

        # Penalties to minimize.
        linvel = get_velocity_in_frame(target_quat, linvel)
        angvel = get_velocity_in_frame(target_quat, angvel)
        xlv = xax.get_norm(linvel[..., 0], self.norm) * self.lin_vel_penalty_scale
        ylv = xax.get_norm(linvel[..., 1], self.norm) * self.lin_vel_penalty_scale
        zlv = xax.get_norm(linvel[..., 2], self.norm) * self.lin_vel_penalty_scale
        xav = xax.get_norm(angvel[..., 0], self.norm) * self.ang_vel_penalty_scale
        yav = xax.get_norm(angvel[..., 1], self.norm) * self.ang_vel_penalty_scale
        zav = xax.get_norm(angvel[..., 2], self.norm) * self.ang_vel_penalty_scale
        alllv = xlv + ylv + zlv
        allav = xav + yav + zav

        # Computes each of the penalties.
        stand_still_penalty = alllv + allav
        walk_forward_penalty = xax.get_norm(forward_vel - self.translation_speed, self.norm) + ylv + zlv + allav
        walk_backward_penalty = xax.get_norm(forward_vel + self.translation_speed, self.norm) + ylv + zlv + allav
        turn_left_penalty = xax.get_norm(rotation_vel + self.rotation_speed, self.norm) + alllv + xlv + ylv
        turn_right_penalty = xax.get_norm(rotation_vel - self.rotation_speed, self.norm) + alllv + xlv + ylv
        strafe_left_penalty = xax.get_norm(left_vel - self.translation_speed, self.norm) + xlv + zlv + allav
        strafe_right_penalty = xax.get_norm(left_vel + self.translation_speed, self.norm) + xlv + zlv + allav

        all_penalties = jnp.stack(
            [
                stand_still_penalty - heading_reward,
                walk_forward_penalty - heading_reward,
                walk_backward_penalty - heading_reward,
                turn_left_penalty,
                turn_right_penalty,
                strafe_left_penalty - heading_reward,
                strafe_right_penalty - heading_reward,
            ],
            axis=-1,
        )

        # Weights each of the penalties by the one-hot encoded command.
        total_penalty = jnp.einsum("...i,...i->...", command_ohe, all_penalties)

        return total_penalty
