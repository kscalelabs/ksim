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
]

import functools
import logging
from abc import ABC, abstractmethod
from typing import Collection, Literal, Self

import attrs
import chex
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

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
        done = jnp.pad(trajectory.done[..., :-1], ((1, 0),), mode="edge")[..., None]
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
        done = jnp.pad(trajectory.done[..., :-1], ((2, 0),), mode="edge")[..., None]
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
        done = jnp.pad(trajectory.done[..., :-1], ((3, 0),), mode="edge")[..., None]
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
        done = trajectory.done[..., None]
        joint_vel = jnp.where(done, 0.0, trajectory.qvel[..., 6:])
        return xax.get_norm(joint_vel, self.norm).mean(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class JointAccelerationPenalty(Reward):
    """Penalty for high joint accelerations."""

    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)

    def get_reward(self, trajectory: Trajectory) -> Array:
        qpos = trajectory.qpos[..., 7:]
        qpos_zp = jnp.pad(qpos, ((2, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done[..., :-1], ((2, 0),), mode="edge")[..., None]
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
        done = jnp.pad(trajectory.done[..., :-1], ((3, 0),), mode="edge")[..., None]
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
        done = jnp.pad(trajectory.done[..., :-1], ((2, 0),), mode="edge")[..., None]
        vel = jnp.where(done, 0.0, jnp.linalg.norm(pos_zp[..., 1:, :, :] - pos_zp[..., :-1, :, :], axis=-1))
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
        done = jnp.pad(trajectory.done[..., :-1], ((3, 0),), mode="edge")[..., None]
        vel = jnp.where(done, 0.0, jnp.linalg.norm(pos_zp[..., 1:, :, :] - pos_zp[..., :-1, :, :], axis=-1))
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


@attrs.define(frozen=True, kw_only=True)
class JoystickReward(Reward):
    """Reward for tracking the joystick commands.

    This reward uses global coordinates, so the robot should always start
    facing forward in the X direction.
    """

    forward_speed: float = attrs.field()
    backward_speed: float = attrs.field()
    strafe_speed: float = attrs.field()
    rotation_speed: float = attrs.field()
    command_name: str = attrs.field(default="joystick_command")
    lin_vel_penalty_scale: float = attrs.field(default=0.01)
    ang_vel_penalty_scale: float = attrs.field(default=0.01)
    norm: xax.NormType = attrs.field(default="l2", validator=norm_validator)
    temp: float = attrs.field(default=0.1)
    monotonic_fn: MonotonicFn = attrs.field(default="inv")
    in_robot_frame: bool = attrs.field(default=False)
    stand_base_reward: float = attrs.field(default=1.0)

    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.command_name not in trajectory.command:
            raise ValueError(f"Command {self.command_name} not found! Ensure that it is in the task.")
        joystick_cmd = trajectory.command[self.command_name]
        chex.assert_shape(joystick_cmd, (..., 7))

        qvel = trajectory.qvel[..., :6]
        linvel = qvel[..., :3]
        angvel = qvel[..., 3:]

        if self.in_robot_frame:
            linvel = xax.rotate_vector_by_quat(linvel, trajectory.qpos[..., 3:7], inverse=True)

        # Penalty to discourage movement in general.
        linvel_norm = jnp.linalg.norm(linvel, axis=-1) * self.lin_vel_penalty_scale
        angvel_norm = jnp.linalg.norm(angvel, axis=-1) * self.ang_vel_penalty_scale
        vel_norm = linvel_norm + angvel_norm

        def normalize(x: Array, scale: float) -> Array:
            return x.clip(-scale, scale) / scale

        # Computes each of the penalties.
        stand_still_reward = self.stand_base_reward * jnp.ones_like(linvel[..., 0])
        walk_forward_reward = normalize(linvel[..., 0], self.forward_speed)
        walk_backward_reward = normalize(-linvel[..., 0], self.backward_speed)
        turn_left_reward = normalize(angvel[..., 2], self.rotation_speed)
        turn_right_reward = normalize(-angvel[..., 2], self.rotation_speed)
        strafe_left_reward = normalize(linvel[..., 1], self.strafe_speed)
        strafe_right_reward = normalize(-linvel[..., 1], self.strafe_speed)

        all_rewards = jnp.stack(
            [
                stand_still_reward,
                walk_forward_reward,
                walk_backward_reward,
                turn_left_reward,
                turn_right_reward,
                strafe_left_reward,
                strafe_right_reward,
            ],
            axis=-1,
        )

        # Weights each of the rewards by the one-hot encoded command.
        total_reward = jnp.einsum("...i,...i->...", joystick_cmd, all_rewards) - vel_norm

        return total_reward
