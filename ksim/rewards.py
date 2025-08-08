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
    "NaiveForwardOrientationReward",
    "AngularVelocityReward",
    "AngularVelocityPenalty",
    "XYAngularVelocityPenalty",
    "BaseHeightReward",
    "BaseHeightRangeReward",
    "ActionVelocityPenalty",
    "ActionAccelerationPenalty",
    "ActionJerkPenalty",
    "JointVelocityPenalty",
    "JointAccelerationPenalty",
    "JointJerkPenalty",
    "AvoidLimitsPenalty",
    "CtrlPenalty",
    "JointDeviationPenalty",
    "FlatBodyReward",
    "PositionTrackingReward",
    "UprightReward",
    "LinkAccelerationPenalty",
    "LinkJerkPenalty",
    "JoystickReward",
    "LinearVelocityTrackingReward",
    "AngularVelocityTrackingReward",
    "ReachabilityPenalty",
    "FeetAirTimeReward",
    "SinusoidalGaitReward",
    "EasyJoystickReward",
]

import functools
import logging
from abc import ABC, abstractmethod
from typing import Collection, Literal, Self, final

import attrs
import chex
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.commands import EasyJoystickCommandValue, JoystickCommandValue, SinusoidalGaitCommandValue
from ksim.debugging import JitLevel
from ksim.types import PhysicsModel, Trajectory
from ksim.utils.mujoco import get_body_data_idx_from_name, get_qpos_data_idxs_by_name
from ksim.utils.validators import (
    CartesianIndex,
    cartesian_index_to_dim,
    dimension_index_tuple_validator,
    norm_validator,
)
from ksim.vis import Marker

logger = logging.getLogger(__name__)

MonotonicFn = Literal["exp", "inv", "sigmoid"]


def norm_to_reward(value: Array, temp: float = 1.0, monotonic_fn: MonotonicFn = "inv") -> Array:
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
        case "inv":
            return 1.0 / (value / temp + 1.0)
        case "exp":
            return jnp.exp(-value / temp)
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


def index_to_dims(index: CartesianIndex | tuple[CartesianIndex, ...]) -> tuple[int, ...]:
    indices = index if isinstance(index, tuple) else (index,)
    return tuple(cartesian_index_to_dim(index) for index in indices)


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

    @final
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
        dims = index_to_dims(self.index)
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
class NaiveForwardReward(Reward):
    """Simple reward for moving forward in the X-direction."""

    clip_min: float | None = attrs.field(default=None)
    clip_max: float | None = attrs.field(default=None)
    in_robot_frame: bool = attrs.field(default=True)

    def get_reward(self, trajectory: Trajectory) -> Array:
        linvel = trajectory.qvel[..., :3]
        if self.in_robot_frame:
            # Same as reading from a velocimeter attached to base.
            linvel = xax.rotate_vector_by_quat(linvel, trajectory.qpos[..., 3:7], inverse=True)
        dimvel = linvel[..., 0].clip(self.clip_min, self.clip_max)
        return dimvel


@attrs.define(frozen=True, kw_only=True)
class NaiveForwardOrientationReward(NaiveForwardReward):
    """Simple reward for keeping the robot oriented in the X-direction."""

    def get_reward(self, trajectory: Trajectory) -> Array:
        quat = trajectory.qpos[..., 3:7]
        forward_vec = jnp.array([1.0, 0.0, 0.0])
        forward_vec = xax.rotate_vector_by_quat(forward_vec, quat, inverse=True)
        return forward_vec[..., 0] - jnp.linalg.norm(forward_vec[..., 1:], axis=-1)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityReward(Reward):
    """Penalty for how fast the robot is rotating in the xy-plane."""

    index: CartesianIndex | tuple[CartesianIndex, ...] = attrs.field(validator=dimension_index_tuple_validator)
    clip_min: float | None = attrs.field(default=None)
    clip_max: float | None = attrs.field(default=None)
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)
    in_robot_frame: bool = attrs.field(default=True)

    def get_reward(self, trajectory: Trajectory) -> Array:
        dims = index_to_dims(self.index)
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
class ActionVelocityPenalty(Reward):
    """Penalty for first derivative change in consecutive actions."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        actions = trajectory.action
        actions_zp = jnp.pad(actions, ((1, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done, ((1, 0),), mode="edge")[..., :-1, None]
        actions_vel = jnp.where(done, 0.0, actions_zp[..., 1:, :] - actions_zp[..., :-1, :])
        penalty = xax.get_norm(actions_vel, self.norm).mean(axis=-1)

        mask = jnp.arange(len(penalty)) >= 1
        penalty = jnp.where(mask, penalty, 0.0)
        return penalty


@attrs.define(frozen=True, kw_only=True)
class ActionAccelerationPenalty(Reward):
    """Penalty for second derivative change in consecutive actions."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        actions = trajectory.action
        actions_zp = jnp.pad(actions, ((2, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done, ((2, 0),), mode="edge")[..., :-1, None]
        actions_vel = jnp.where(done, 0.0, actions_zp[..., 1:, :] - actions_zp[..., :-1, :])
        actions_acc = jnp.where(done[..., 1:, :], 0.0, actions_vel[..., 1:, :] - actions_vel[..., :-1, :])
        penalty = xax.get_norm(actions_acc, self.norm).mean(axis=-1)

        # Mask out timesteps affected by padding to prevent artificial artifacts
        # (e.g. jump from constant velocity to non-zero acceleration)
        mask = jnp.arange(len(penalty)) >= 2
        penalty = jnp.where(mask, penalty, 0.0)
        return penalty


@attrs.define(frozen=True, kw_only=True)
class ActionJerkPenalty(Reward):
    """Penalty for third derivative change in consecutive actions."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        actions = trajectory.action
        actions_zp = jnp.pad(actions, ((3, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done, ((3, 0),), mode="edge")[..., :-1, None]
        actions_vel = jnp.where(done, 0.0, actions_zp[..., 1:, :] - actions_zp[..., :-1, :])
        actions_acc = jnp.where(done[..., 1:, :], 0.0, actions_vel[..., 1:, :] - actions_vel[..., :-1, :])
        actions_jerk = jnp.where(done[..., 2:, :], 0.0, actions_acc[..., 1:, :] - actions_acc[..., :-1, :])
        penalty = xax.get_norm(actions_jerk, self.norm).mean(axis=-1)

        mask = jnp.arange(len(penalty)) >= 3
        penalty = jnp.where(mask, penalty, 0.0)
        return penalty


@attrs.define(frozen=True, kw_only=True)
class JointVelocityPenalty(Reward):
    """Penalty for how fast the joint angular velocities are changing."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        qpos = trajectory.qpos[..., 7:]
        qpos_zp = jnp.pad(qpos, ((1, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done, ((1, 0),), mode="edge")[..., :-1, None]
        qvel = jnp.where(done, 0.0, qpos_zp[..., 1:, :] - qpos_zp[..., :-1, :])
        return xax.get_norm(qvel, self.norm).mean(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class JointAccelerationPenalty(Reward):
    """Penalty for high joint accelerations."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        qpos = trajectory.qpos[..., 7:]
        qpos_zp = jnp.pad(qpos, ((2, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done, ((2, 0),), mode="edge")[..., :-1, None]
        qvel = jnp.where(done, 0.0, qpos_zp[..., 1:, :] - qpos_zp[..., :-1, :])
        qacc = jnp.where(done[..., 1:, :], 0.0, qvel[..., 1:, :] - qvel[..., :-1, :])
        penalty = xax.get_norm(qacc, self.norm).mean(axis=-1)
        return penalty


@attrs.define(frozen=True, kw_only=True)
class JointJerkPenalty(Reward):
    """Penalty for high joint jerks."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        qpos = trajectory.qpos[..., 7:]
        qpos_zp = jnp.pad(qpos, ((3, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done, ((3, 0),), mode="edge")[..., :-1, None]
        qvel = jnp.where(done, 0.0, qpos_zp[..., 1:, :] - qpos_zp[..., :-1, :])
        qacc = jnp.where(done[..., 1:, :], 0.0, qvel[..., 1:, :] - qvel[..., :-1, :])
        qjerk = jnp.where(done[..., 2:, :], 0.0, qacc[..., 1:, :] - qacc[..., :-1, :])
        penalty = xax.get_norm(qjerk, self.norm).mean(axis=-1)
        return penalty


def joint_limits_validator(inst: "AvoidLimitsPenalty", attr: attrs.Attribute, value: xax.HashableArray) -> None:
    arr = value.array
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Joint range must have shape (n_joints, 2), got {arr.shape}")
    if not jnp.all(arr[..., 0] <= arr[..., 1]):
        raise ValueError(f"Joint range must be sorted, got {arr}")
    if not arr.dtype == jnp.float32:
        raise ValueError(f"Joint range must be a float array, got {arr.dtype}")


@attrs.define(frozen=True, kw_only=True)
class AvoidLimitsPenalty(Reward):
    """Reward for keeping the joint positions away from the joint limits."""

    joint_limits: xax.HashableArray = attrs.field(validator=joint_limits_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        joint_pos = trajectory.qpos[..., 7:]
        joint_limits = self.joint_limits.array
        penalty = -jnp.clip(joint_pos - joint_limits[..., 0], None, 0.0)
        penalty += jnp.clip(joint_pos - joint_limits[..., 1], 0.0, None)
        return penalty.mean(axis=-1)

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
class CtrlPenalty(Reward):
    """Penalty for large torque commands."""

    norm: xax.NormType = attrs.field(default="l2")
    scales: tuple[float, ...] | None = attrs.field(default=None)

    def get_reward(self, trajectory: Trajectory) -> Array:
        ctrl = trajectory.ctrl
        if self.scales is not None:
            ctrl = ctrl / jnp.array(self.scales)
        return xax.get_norm(ctrl, self.norm).mean(axis=-1)

    @classmethod
    def create(cls, model: PhysicsModel, scale: float = -1.0, scale_by_curriculum: bool = False) -> Self:
        ctrl_min = model.actuator_ctrlrange[..., 0]
        ctrl_max = model.actuator_ctrlrange[..., 1]
        ctrl_range = (ctrl_max - ctrl_min) / 2.0
        ctrl_range_list = ctrl_range.flatten().tolist()
        return cls(
            scales=tuple(ctrl_range_list),
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(frozen=True, kw_only=True)
class JointDeviationPenalty(Reward):
    """Penalty for joint deviations from target positions."""

    norm: xax.NormType = attrs.field(default="l2")
    joint_indices: tuple[int, ...] = attrs.field()
    joint_targets: tuple[float, ...] = attrs.field()
    joint_weights: tuple[float, ...] | None = attrs.field(default=None)

    def get_reward(self, trajectory: Trajectory) -> Array:
        qpos_sel = trajectory.qpos[..., jnp.array(self.joint_indices) + 7]
        target = jnp.asarray(self.joint_targets)
        diff = qpos_sel - target

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
        local_z = jnp.array([0.0, 0.0, 1.0])
        quat = trajectory.qpos[..., 3:7]
        global_z = xax.rotate_vector_by_quat(local_z, quat)
        return global_z[..., 2] - jnp.linalg.norm(global_z[..., :2], axis=-1)


@attrs.define(frozen=True, kw_only=True)
class LinkAccelerationPenalty(Reward):
    """Penalty for high link accelerations in the world frame."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        pos = trajectory.xpos[..., 1:, :]
        pos_zp = jnp.pad(pos, ((2, 0), (0, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done, ((2, 0),), mode="edge")[..., :-1, None]
        vel: Array = jnp.linalg.norm(pos_zp[..., 1:, :, :] - pos_zp[..., :-1, :, :], axis=-1)
        vel = jnp.where(done, 0.0, vel)
        acc = jnp.where(done[..., 1:, :], 0.0, vel[..., 1:, :] - vel[..., :-1, :])
        penalty = xax.get_norm(acc, self.norm).mean(axis=-1)
        return penalty


@attrs.define(frozen=True, kw_only=True)
class LinkJerkPenalty(Reward):
    """Penalty for high link jerks in the world frame."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        pos = trajectory.xpos[..., 1:, :]
        pos_zp = jnp.pad(pos, ((3, 0), (0, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done, ((3, 0),), mode="edge")[..., :-1, None]
        vel: Array = jnp.linalg.norm(pos_zp[..., 1:, :, :] - pos_zp[..., :-1, :, :], axis=-1)
        vel = jnp.where(done, 0.0, vel)
        acc = jnp.where(done[..., 1:, :], 0.0, vel[..., 1:, :] - vel[..., :-1, :])
        jerk = jnp.where(done[..., 2:, :], 0.0, acc[..., 1:, :] - acc[..., :-1, :])
        penalty = xax.get_norm(jerk, self.norm).mean(axis=-1)
        return penalty


@attrs.define(frozen=True, kw_only=True)
class SymmetryReward(Reward):
    """Rewards joints for having symmetrical positions."""

    joint_indices: tuple[int, ...] = attrs.field()
    joint_targets: tuple[float, ...] = attrs.field()

    def get_reward(self, trajectory: Trajectory) -> Array:
        qpos = trajectory.qpos[..., jnp.array(self.joint_indices) + 7] - jnp.array(self.joint_targets)
        qpos_mean = qpos.mean(axis=-2, keepdims=True)
        return (qpos * -qpos_mean).sum(axis=-1)

    @classmethod
    def create(
        cls,
        physics_model: PhysicsModel,
        joint_names: tuple[str, ...],
        joint_targets: tuple[float, ...],
        scale: float = 1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        joint_to_idx = get_qpos_data_idxs_by_name(physics_model)
        joint_indices = tuple([int(joint_to_idx[name][0]) - 7 for name in joint_names])
        return cls(
            joint_indices=joint_indices,
            joint_targets=joint_targets,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )


@attrs.define(kw_only=True)
class JoystickRewardMarker(Marker):
    radius: float = attrs.field(default=0.1)
    size: float = attrs.field(default=0.03)
    arrow_len: float = attrs.field(default=1.0)
    height: float = attrs.field(default=0.5)

    def _update_arrow(self, cmd_x: float, cmd_y: float) -> None:
        self.geom = mujoco.mjtGeom.mjGEOM_ARROW  # pyright: ignore[reportAttributeAccessIssue]
        mag = (cmd_x * cmd_x + cmd_y * cmd_y) ** 0.5
        cmd_x, cmd_y = cmd_x / mag, cmd_y / mag
        self.orientation = self.quat_from_direction((cmd_x, cmd_y, 0.0))
        self.scale = (self.size, self.size, self.arrow_len * mag)

    def update(self, trajectory: Trajectory) -> None:
        """Visualizes the joystick command target position and orientation."""
        cur_xvel, cur_yvel = trajectory.qvel[..., 0].item(), trajectory.qvel[..., 1].item()
        self.pos = (0, 0, self.height)
        self._update_arrow(cur_xvel, cur_yvel)

    @classmethod
    def get(
        cls,
        radius: float = 0.05,
        size: float = 0.03,
        arrow_len: float = 0.25,
        rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        height: float = 0.6,
    ) -> Self:
        return cls(
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_SPHERE,  # pyright: ignore[reportAttributeAccessIssue]
            scale=(radius, radius, radius),
            size=size,
            arrow_len=arrow_len,
            radius=radius,
            rgba=rgba,
            height=height,
            track_x=True,
            track_y=True,
            track_z=True,
            track_rotation=False,
        )


@attrs.define(frozen=True, kw_only=True)
class JoystickReward(Reward):
    """Reward for following the joystick command."""

    command_name: str = attrs.field(default="joystick_command")
    pos_x_scale: float = attrs.field(default=0.25)
    pos_y_scale: float = attrs.field(default=0.25)
    rot_z_scale: float = attrs.field(default=0.25)
    ang_penalty_ratio: float = attrs.field(default=2.0)

    @xax.jit(static_argnames=["self"], jit_level=JitLevel.UNROLL)
    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.command_name not in trajectory.command:
            raise ValueError(f"Command {self.command_name} not found! Ensure that it is in the task.")
        return self._get_reward_for(trajectory.command[self.command_name], trajectory)

    def _get_reward_for(self, joystick_cmd: JoystickCommandValue, trajectory: Trajectory) -> Array:
        # Gets the target X, Y, and Yaw velocities.
        tgts = joystick_cmd.vels

        # Smooths the target velocities.
        trg_xvel, trg_yvel, trg_yawvel = tgts.T

        # Gets the robot's current velocities.
        cur_xvel = trajectory.qvel[..., 0]
        cur_yvel = trajectory.qvel[..., 1]
        cur_yawvel = trajectory.qvel[..., 5]

        # Gets the robot's current yaw.
        quat = trajectory.qpos[..., 3:7]
        cur_yaw = xax.quat_to_yaw(quat)

        # Rotates the command X and Y velocities to the robot's current yaw.
        trg_xvel_rot = trg_xvel * jnp.cos(cur_yaw) - trg_yvel * jnp.sin(cur_yaw)
        trg_yvel_rot = trg_xvel * jnp.sin(cur_yaw) + trg_yvel * jnp.cos(cur_yaw)

        # Exponential kernel for the reward.
        x_rew_diff = trg_xvel_rot - cur_xvel
        y_rew_diff = trg_yvel_rot - cur_yvel
        z_rew_diff = trg_yawvel - cur_yawvel

        pos_x_rew = jnp.exp(-jnp.abs(x_rew_diff) / self.pos_x_scale)
        pos_y_rew = jnp.exp(-jnp.abs(y_rew_diff) / self.pos_y_scale)
        rot_z_rew = jnp.exp(-jnp.abs(z_rew_diff) / self.rot_z_scale)

        reward = (pos_x_rew + pos_y_rew + rot_z_rew) / 3.0

        return reward

    def get_markers(self) -> Collection[Marker]:
        return [JoystickRewardMarker.get()]


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityTrackingReward(Reward):
    """Reward for tracking the linear velocity."""

    linvel_obs_name: str = attrs.field()
    index: CartesianIndex | tuple[CartesianIndex, ...] = attrs.field(
        default=("x", "y"), validator=dimension_index_tuple_validator
    )
    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="linear_velocity_command")
    in_robot_frame: bool = attrs.field(default=True)
    norm: xax.NormType = attrs.field(default="l2")

    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.linvel_obs_name not in trajectory.obs:
            raise ValueError(f"Observation {self.linvel_obs_name} not found; add it as an observation in your task.")

        linvel = trajectory.obs[self.linvel_obs_name]
        if self.in_robot_frame:
            linvel = xax.rotate_vector_by_quat(linvel, trajectory.qpos[..., 3:7], inverse=True)

        dims = index_to_dims(self.index)

        linvel = linvel[..., dims]
        robot_vel_cmd = trajectory.command[self.command_name]
        robot_vel_cmd = robot_vel_cmd[..., dims]

        vel_error = xax.get_norm(linvel - robot_vel_cmd, self.norm).sum(axis=-1)

        return jnp.exp(-vel_error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityTrackingReward(Reward):
    """Reward for tracking the angular velocity."""

    index: CartesianIndex | tuple[CartesianIndex, ...] = attrs.field(
        default=("x", "y"), validator=dimension_index_tuple_validator
    )
    error_scale: float = attrs.field(default=0.25)
    command_name: str = attrs.field(default="angular_velocity_command")
    norm: xax.NormType = attrs.field(default="l2")

    def get_reward(self, trajectory: Trajectory) -> Array:
        angvel = trajectory.qvel[..., 3:6]

        dims = index_to_dims(self.index)

        angvel = angvel[..., dims]
        robot_angvel_cmd = trajectory.command[self.command_name]

        chex.assert_shape(robot_angvel_cmd, (..., len(dims)))

        angvel_error = xax.get_norm(angvel - robot_angvel_cmd, self.norm).sum(axis=-1)

        return jnp.exp(-angvel_error / self.error_scale)


@attrs.define(frozen=True, kw_only=True)
class ReachabilityPenalty(Reward):
    """Penalty for commands that exceed the per‑joint reachability envelope.

    Maximum joint movement in a single time step is given by:
    Δmax = vmax·Δt + ½·amax·Δt²
    """

    delta_max_j: tuple[float, ...] = attrs.field(eq=False, hash=False)
    squared: bool = True  # ‑‑ Use L2 on the excess (L1 if False).

    def get_reward(self, traj: Trajectory) -> jnp.ndarray:
        action_tj = traj.action
        q_prev_tj = traj.qpos[..., 7:]
        dm_j = jnp.asarray(self.delta_max_j)

        excess_tj = jnp.maximum(0.0, jnp.abs(action_tj - q_prev_tj) - dm_j)
        per_joint = excess_tj**2 if self.squared else excess_tj
        penalty_t = per_joint.sum(axis=-1)

        return penalty_t


@attrs.define(frozen=True, kw_only=True)
class FeetAirTimeReward(StatefulReward):
    """Reward for feet either touching or not touching the ground for some time."""

    threshold: float = attrs.field()
    ctrl_dt: float = attrs.field()
    num_feet: int = attrs.field(default=2)
    contact_obs: str = attrs.field(default="feet_contact_observation")

    def initial_carry(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(self.num_feet, dtype=jnp.int32)

    def get_reward_stateful(
        self,
        trajectory: Trajectory,
        reward_carry: Array,
    ) -> tuple[Array, Array]:
        sensor_data_tcn = trajectory.obs[self.contact_obs] > 0.5  # Values are either 0 or 1.
        sensor_data_tn = sensor_data_tcn.any(axis=-2)
        chex.assert_shape(sensor_data_tn, (..., self.num_feet))

        threshold_steps = round(self.threshold / self.ctrl_dt)

        def scan_fn(carry: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
            count_n, (contact_n, done) = carry, x
            count_n = jnp.where(done | contact_n, 0, count_n + 1)
            return count_n, count_n

        reward_carry, count_tn = xax.scan(scan_fn, reward_carry, (sensor_data_tn, trajectory.done))

        # Gradually increase reward until `threshold_steps`.
        reward_tn = count_tn.astype(jnp.float32) / threshold_steps
        reward_tn = jnp.where((count_tn > 0) & (count_tn < threshold_steps), reward_tn, 0.0)
        reward_t = reward_tn.sum(axis=-1)

        return reward_t, reward_carry


@attrs.define(kw_only=True)
class SinusoidalGaitTargetMarker(Marker):
    foot_id: int = attrs.field()
    radius: float = attrs.field(default=0.1)
    size: float = attrs.field(default=0.03)
    obs_name: str = attrs.field(default="feet_position_observation")
    cmd_name: str = attrs.field(default="sinusoidal_gait_command")

    def update(self, trajectory: Trajectory) -> None:
        """Visualizes the sinusoidal gait."""
        obs_x, obs_y, _ = trajectory.obs[self.obs_name][..., self.foot_id, :].tolist()
        cmd: SinusoidalGaitCommandValue = trajectory.command[self.cmd_name]
        cmd_h = cmd.height[..., self.foot_id].item()
        self.pos = (obs_x, obs_y, cmd_h)

    @classmethod
    def get(
        cls,
        foot_id: int,
        radius: float = 0.05,
        size: float = 0.03,
        obs_name: str = "feet_position_observation",
        cmd_name: str = "sinusoidal_gait_command",
    ) -> Self:
        return cls(
            foot_id=foot_id,
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_SPHERE,  # pyright: ignore[reportAttributeAccessIssue]
            scale=(radius, radius, radius),
            size=size,
            radius=radius,
            obs_name=obs_name,
            cmd_name=cmd_name,
            rgba=(1.0, 0.0, 0.0, 1.0),  # Red
            track_x=False,
            track_y=False,
            track_z=False,
            track_rotation=False,
        )


@attrs.define(kw_only=True)
class SinusoidalGaitPositionMarker(Marker):
    foot_id: int = attrs.field()
    radius: float = attrs.field(default=0.1)
    size: float = attrs.field(default=0.03)
    obs_name: str = attrs.field(default="feet_position_observation")

    def update(self, trajectory: Trajectory) -> None:
        """Visualizes the sinusoidal gait."""
        x, y, z = trajectory.obs[self.obs_name][..., self.foot_id, :].tolist()
        self.pos = (x, y, z)

    @classmethod
    def get(
        cls,
        foot_id: int,
        obs_name: str = "feet_position_observation",
        radius: float = 0.05,
        size: float = 0.03,
    ) -> Self:
        return cls(
            foot_id=foot_id,
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_SPHERE,  # pyright: ignore[reportAttributeAccessIssue]
            scale=(radius, radius, radius),
            size=size,
            radius=radius,
            obs_name=obs_name,
            rgba=(0.0, 1.0, 0.0, 1.0),  # Green
            track_x=False,
            track_y=False,
            track_z=False,
            track_rotation=False,
        )


@attrs.define(frozen=True, kw_only=True)
class SinusoidalGaitReward(Reward):
    """Reward for a biped following a sinusoidal gait."""

    ctrl_dt: float = attrs.field()
    pos_obs: str = attrs.field(default="feet_position_observation")
    pos_cmd: str = attrs.field(default="sinusoidal_gait_command")
    num_feet: int = attrs.field(default=2)

    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.pos_cmd not in trajectory.command:
            raise ValueError(f"Command {self.pos_cmd} not found! Ensure that it is in the task.")
        return self._get_reward_for(trajectory.command[self.pos_cmd], trajectory)

    def _get_reward_for(self, gait_cmd: SinusoidalGaitCommandValue, trajectory: Trajectory) -> Array:
        obs = trajectory.obs[self.pos_obs][..., 2]
        cmd = gait_cmd.height
        reward = jnp.exp(-xax.get_norm(obs - cmd, "l2").sum(axis=-1))
        return reward

    def get_markers(self) -> Collection[Marker]:
        return [
            marker
            for foot_id in range(self.num_feet)
            for marker in (
                SinusoidalGaitPositionMarker.get(foot_id, obs_name=self.pos_obs),
                SinusoidalGaitTargetMarker.get(foot_id, obs_name=self.pos_obs, cmd_name=self.pos_cmd),
            )
        ]


@attrs.define(kw_only=True)
class EasyJoystickGaitTargetMarker(SinusoidalGaitTargetMarker):
    cmd_name: str = attrs.field(default="easy_joystick_command")

    def update(self, trajectory: Trajectory) -> None:
        """Visualizes the sinusoidal gait."""
        obs_x, obs_y, _ = trajectory.obs[self.obs_name][..., self.foot_id, :].tolist()
        cmd: EasyJoystickCommandValue = trajectory.command[self.cmd_name]
        cmd_h = cmd.gait.height[..., self.foot_id].item()
        self.pos = (obs_x, obs_y, cmd_h)


@attrs.define(frozen=True, kw_only=True)
class EasyJoystickReward(StatefulReward):
    """Provides an easy-to-learn joystick reward.

    When training joystick control policies, there's a bunch of tricky stuff
    you need to do to get them to train well, compared with something like
    NaiveForwardReward. This reward combines a few other rewards, with the
    goal of providing a "one-shot" reward that you can put into your policy to
    give your robot the ability to follow joystick commands.
    """

    gait: SinusoidalGaitReward = attrs.field()
    joystick: JoystickReward = attrs.field()
    airtime: FeetAirTimeReward = attrs.field()
    scale: float = attrs.field(default=1.0)
    command_name: str = attrs.field(default="easy_joystick_command")

    def initial_carry(self, rng: PRNGKeyArray) -> Array:
        return self.airtime.initial_carry(rng)

    def get_reward_stateful(self, trajectory: Trajectory, reward_carry: Array) -> tuple[Array, Array]:
        if self.command_name not in trajectory.command:
            raise ValueError(f"Command {self.command_name} not found! Ensure that it is in the task.")

        cmd: EasyJoystickCommandValue = trajectory.command[self.command_name]
        joystick_reward = self.joystick._get_reward_for(cmd.joystick, trajectory) * self.joystick.scale
        gait_reward = self.gait._get_reward_for(cmd.gait, trajectory) * self.gait.scale
        airtime_reward, airtime_carry = self.airtime.get_reward_stateful(trajectory, reward_carry)

        # Mask out airtime reward when the robot is not moving.
        airtime_reward = jnp.where(cmd.joystick.command.argmax(axis=-1) == 0, 0.0, airtime_reward)

        total_reward = joystick_reward + gait_reward + airtime_reward * self.airtime.scale
        return total_reward, airtime_carry

    def get_markers(self) -> Collection[Marker]:
        return [
            marker
            for foot_id in range(self.gait.num_feet)
            for marker in (
                SinusoidalGaitPositionMarker.get(foot_id, obs_name=self.gait.pos_obs),
                EasyJoystickGaitTargetMarker.get(foot_id, obs_name=self.gait.pos_obs, cmd_name=self.command_name),
            )
        ] + [JoystickRewardMarker.get()]
