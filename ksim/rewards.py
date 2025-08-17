"""Defines a base interface for defining reward functions."""

__all__ = [
    "MonotonicFn",
    "norm_to_reward",
    "Reward",
    "StatefulReward",
    "StayAliveReward",
    "LinearVelocityPenalty",
    "AngularVelocityPenalty",
    "AngularVelocityPenalty",
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
    "ReachabilityPenalty",
    "FeetAirTimeReward",
    "FeetForcePenalty",
    "FeetTorquePenalty",
    "SinusoidalGaitReward",
    "BaseHeightTrackingReward",
]

import logging
from abc import ABC, abstractmethod
from typing import Collection, Literal, Mapping, Self, final

import attrs
import chex
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.commands import AngularVelocityCommandValue, LinearVelocityCommandValue, SinusoidalGaitCommandValue
from ksim.types import PhysicsModel, Trajectory
from ksim.utils.mujoco import get_body_data_idx_from_name, get_qpos_data_idxs_by_name
from ksim.utils.validators import (
    CartesianIndex,
    cartesian_index_to_dim,
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
    reward_name = inst.__class__.__name__
    if reward_name.lower().endswith("reward"):
        if value < 0:
            raise RuntimeError(f"Reward function {reward_name} has a negative scale {value}")
    elif reward_name.lower().endswith("penalty"):
        if value > 0:
            raise RuntimeError(f"Penalty function {reward_name} has a positive scale {value}")
    else:
        raise ValueError(f"Reward function {reward_name} does not end with 'Reward' or 'Penalty': {value}")


def index_to_dims(index: CartesianIndex | tuple[CartesianIndex, ...]) -> tuple[int, ...]:
    indices = index if isinstance(index, tuple) else (index,)
    return tuple(cartesian_index_to_dim(index) for index in indices)


@attrs.define(frozen=True, kw_only=True)
class Reward(ABC):
    """Base class for defining reward functions."""

    scale: float = attrs.field(validator=reward_scale_validator)
    scale_by_curriculum: bool = attrs.field(default=False)

    @abstractmethod
    def get_reward(self, trajectory: Trajectory) -> Array | Mapping[str, Array]:
        """Get the reward for a single trajectory.

        Args:
            trajectory: The trajectory to get the reward for.

        Returns:
            An array of shape (time) containing the reward for each timestep.
        """

    def get_markers(self, name: str) -> Collection[Marker]:
        """Get the markers for the reward, optionally overridable."""
        return []


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
    def get_reward_stateful(
        self,
        trajectory: Trajectory,
        reward_carry: PyTree,
    ) -> tuple[Array | Mapping[str, Array], PyTree]:
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

    balance: float = attrs.field(default=100.0)
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


@attrs.define(kw_only=True)
class LinearVelocityPenaltyMarker(Marker):
    size: float = attrs.field(default=0.03)
    arrow_scale: float = attrs.field(default=0.3)
    height: float = attrs.field(default=0.5)
    base_length: float = attrs.field(default=0.15)
    zero_threshold: float = attrs.field(default=1e-4)

    def update(self, trajectory: Trajectory) -> None:
        """Visualizes the sinusoidal gait."""
        linvel = trajectory.qvel[..., :3]
        linvel = xax.rotate_vector_by_quat(linvel, trajectory.qpos[..., 3:7], inverse=True)
        xy = linvel[..., :2]
        x = float(xy[..., 0])
        y = float(xy[..., 1])
        speed = float(jnp.linalg.norm(xy, axis=-1))
        direction = (x / speed, y / speed, 0.0)

        self.pos = (0.0, 0.0, self.height)

        # Always show an arrow with base_length plus scaling by speed
        self.geom = mujoco.mjtGeom.mjGEOM_ARROW  # pyright: ignore[reportAttributeAccessIssue]
        arrow_length = self.base_length + self.arrow_scale * speed
        self.scale = (self.size, self.size, arrow_length)

        # If command is near-zero, show grey arrow pointing +X.
        if speed < self.zero_threshold:
            self.orientation = self.quat_from_direction((1.0, 0.0, 0.0))
            self.rgba = (0.8, 0.8, 0.8, 0.8)
        else:
            self.orientation = self.quat_from_direction(direction)
            self.rgba = (0.2, 0.2, 0.8, 0.8)

    @classmethod
    def get(
        cls,
        *,
        arrow_scale: float = 0.3,
        height: float = 0.5,
        base_length: float = 0.15,
    ) -> Self:
        return cls(
            target_type="root",
            geom=mujoco.mjtGeom.mjGEOM_ARROW,  # pyright: ignore[reportAttributeAccessIssue]
            scale=(0.03, 0.03, base_length),
            arrow_scale=arrow_scale,
            height=height,
            base_length=base_length,
            track_rotation=True,
        )


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityPenalty(Reward):
    """Penalty for how fast the robot is moving in the z-direction."""

    cmd: str = attrs.field()
    zero_threshold: float = attrs.field(default=0.01)
    vis_height: float = attrs.field(default=0.6)

    def get_reward(self, trajectory: Trajectory) -> dict[str, Array]:
        cmd: LinearVelocityCommandValue = trajectory.command[self.cmd]

        # Gets the linear velocity in the robot's frame.
        linvel = trajectory.qvel[..., :3]
        linvel = xax.rotate_vector_by_quat(linvel, trajectory.qpos[..., 3:7], inverse=True)
        xy = linvel[..., :2]
        vel = jnp.linalg.norm(xy, axis=-1)
        x = xy[..., 0]
        y = xy[..., 1]
        yaw = jnp.arctan2(y, x)

        # Don't reward if the command is zero.
        is_zero = jnp.abs(cmd.vel) < self.zero_threshold

        return {
            "l1_vel": xax.get_norm(vel - cmd.vel, "l1"),
            "l2_vel": xax.get_norm(vel - cmd.vel, "l2"),
            "l1_yaw": jnp.where(is_zero, 0.0, xax.get_norm(yaw - cmd.yaw, "l1")),
            "l2_yaw": jnp.where(is_zero, 0.0, xax.get_norm(yaw - cmd.yaw, "l2")),
            "l1_x": xax.get_norm(x - cmd.xvel, "l1"),
            "l2_x": xax.get_norm(x - cmd.xvel, "l2"),
            "l1_y": xax.get_norm(y - cmd.yvel, "l1"),
            "l2_y": xax.get_norm(y - cmd.yvel, "l2"),
        }

    def get_markers(self, name: str) -> Collection[Marker]:
        return [LinearVelocityPenaltyMarker.get(height=self.vis_height)]


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityPenalty(Reward):
    """Penalty for how fast the robot is rotating in the xy-plane."""

    cmd: str = attrs.field()

    def get_reward(self, trajectory: Trajectory) -> dict[str, Array]:
        cmd: AngularVelocityCommandValue = trajectory.command[self.cmd]
        angvel = trajectory.qvel[..., 5]
        return {
            "l1_angvel": xax.get_norm(angvel - cmd.vel, "l1"),
            "l2_angvel": xax.get_norm(angvel - cmd.vel, "l2"),
        }


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
    contact_obs: str = attrs.field()
    position_obs: str = attrs.field()
    height: float = attrs.field()
    num_feet: int = attrs.field(default=2)
    bias: float = attrs.field(default=0.0)
    linvel_moving_threshold: float = attrs.field(default=0.05)
    angvel_moving_threshold: float = attrs.field(default=0.05)

    def initial_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(self.num_feet, dtype=jnp.int32),
            jnp.zeros(self.num_feet, dtype=jnp.float32),
        )

    def get_reward_stateful(
        self,
        trajectory: Trajectory,
        reward_carry: tuple[Array, Array],
    ) -> tuple[Array, tuple[Array, Array]]:
        not_moving_lin = jnp.linalg.norm(trajectory.qvel[..., :2], axis=-1) < self.linvel_moving_threshold
        not_moving_ang = trajectory.qvel[..., 5] < self.angvel_moving_threshold
        not_moving = not_moving_lin & not_moving_ang

        sensor_data_tcn = trajectory.obs[self.contact_obs] > 0.5  # Values are either 0 or 1.
        sensor_data_tn = sensor_data_tcn.any(axis=-2)
        chex.assert_shape(sensor_data_tn, (..., self.num_feet))

        position_tn3 = trajectory.obs[self.position_obs]
        chex.assert_shape(position_tn3, (..., self.num_feet, 3))

        threshold_steps = round(self.threshold / self.ctrl_dt)

        def scan_fn(
            carry: tuple[Array, Array],
            x: tuple[Array, Array, Array, Array],
        ) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
            (count_n, max_height_n), (contact_n, position_n3, not_moving, done) = carry, x
            reset = done | not_moving | contact_n
            count_n = jnp.where(reset, 0, count_n + 1)

            height_n = position_n3[..., 2]
            max_height_n = jnp.where(reset, 0.0, jnp.maximum(max_height_n, height_n))

            return (count_n, max_height_n), (count_n, max_height_n)

        reward_carry, (count_tn, max_height_tn) = xax.scan(
            scan_fn,
            reward_carry,
            (sensor_data_tn, position_tn3, not_moving, trajectory.done),
        )

        # Gradually increase reward until `threshold_steps`.
        reward_tn = (count_tn.astype(jnp.float32) / threshold_steps) + self.bias
        reward_tn = jnp.where((count_tn > 0) & (count_tn < threshold_steps), reward_tn, 0.0)

        # Scale the reward according to the max height.
        reward_tn = reward_tn * max_height_tn.clip(max=self.height) / self.height

        reward_t = reward_tn.sum(axis=-1)

        return reward_t, reward_carry


@attrs.define(frozen=True, kw_only=True)
class FeetForcePenalty(Reward):
    """Reward for reducing the force on the feet."""

    force_obs: str = attrs.field()
    bias: float = attrs.field(default=0.0)

    def get_reward(self, trajectory: Trajectory) -> Array:
        force_t = (jnp.linalg.norm(trajectory.obs[self.force_obs], axis=-1) - self.bias).clip(min=0.0)
        return force_t.sum(axis=-1) ** 2


@attrs.define(frozen=True, kw_only=True)
class FeetTorquePenalty(Reward):
    """Reward for reducing the force on the feet."""

    torque_obs: str = attrs.field()

    def get_reward(self, trajectory: Trajectory) -> Array:
        torque_t = jnp.linalg.norm(trajectory.obs[self.torque_obs], axis=-1)
        return torque_t.sum(axis=-1) ** 2


@attrs.define(kw_only=True)
class SinusoidalGaitTargetMarker(Marker):
    foot_id: int = attrs.field()
    obs_name: str = attrs.field()
    radius: float = attrs.field(default=0.1)
    size: float = attrs.field(default=0.03)
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
        obs_name: str,
        cmd_name: str,
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
    obs_name: str = attrs.field()
    radius: float = attrs.field(default=0.1)
    size: float = attrs.field(default=0.03)

    def update(self, trajectory: Trajectory) -> None:
        """Visualizes the sinusoidal gait."""
        x, y, z = trajectory.obs[self.obs_name][..., self.foot_id, :].tolist()
        self.pos = (x, y, z)

    @classmethod
    def get(
        cls,
        foot_id: int,
        obs_name: str,
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
    max_height: float = attrs.field()
    pos_obs: str = attrs.field()
    pos_cmd: str = attrs.field()
    num_feet: int = attrs.field(default=2)

    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.pos_cmd not in trajectory.command:
            raise ValueError(f"Command {self.pos_cmd} not found! Ensure that it is in the task.")
        return self._get_reward_for(trajectory.command[self.pos_cmd], trajectory)

    def _get_reward_for(self, gait_cmd: SinusoidalGaitCommandValue, trajectory: Trajectory) -> Array:
        obs = trajectory.obs[self.pos_obs][..., 2]
        cmd = gait_cmd.height
        reward = 1.0 - (jnp.abs(obs - cmd).sum(axis=-1)) / self.max_height
        return reward

    def get_markers(self, name: str) -> Collection[Marker]:
        return [
            marker
            for foot_id in range(self.num_feet)
            for marker in (
                SinusoidalGaitPositionMarker.get(foot_id, obs_name=self.pos_obs),
                SinusoidalGaitTargetMarker.get(foot_id, obs_name=self.pos_obs, cmd_name=self.pos_cmd),
            )
        ]


@attrs.define(frozen=True, kw_only=True)
class BaseHeightTrackingReward(Reward):
    """Penalty for deviating from the base height target."""

    command_name: str = attrs.field(default="base_height_command")

    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.command_name not in trajectory.command:
            raise ValueError(f"Command {self.command_name} not found! Ensure that it is in the task.")
        cmd = trajectory.command[self.command_name]
        trg = trajectory.qpos[..., 2]
        return -jnp.abs(cmd - trg)
