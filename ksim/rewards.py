"""Defines a base interface for defining reward functions."""

__all__ = [
    "MonotonicFn",
    "Reward",
    "StatefulReward",
    "StayAliveReward",
    "LinearVelocityReward",
    "AngularVelocityReward",
    "OffAxisVelocityReward",
    "BaseHeightReward",
    "BaseHeightRangeReward",
    "ActionVelocityPenalty",
    "ActionAccelerationPenalty",
    "ActionJerkPenalty",
    "SmallJointVelocityReward",
    "SmallJointAccelerationReward",
    "SmallJointJerkReward",
    "AvoidLimitsPenalty",
    "SmallCtrlReward",
    "JointDeviationPenalty",
    "FlatBodyReward",
    "PositionTrackingReward",
    "UprightReward",
    "LinkAccelerationPenalty",
    "LinkJerkPenalty",
    "SymmetryReward",
    "ReachabilityPenalty",
    "FeetAirTimeReward",
    "FeetGroundedAtRestReward",
    "TargetHeightReward",
    "SparseTargetHeightReward",
    "MotionlessAtRestPenalty",
    "ForcePenalty",
    "SinusoidalGaitReward",
    "JointPositionReward",
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

from ksim.commands import (
    AngularVelocityCommandValue,
    JointPositionCommandValue,
    LinearVelocityCommandValue,
    SinusoidalGaitCommandValue,
)
from ksim.types import PhysicsModel, Trajectory
from ksim.utils.mujoco import get_body_data_idx_from_name, get_joint_names_in_order, get_qpos_data_idxs_by_name
from ksim.utils.validators import CartesianIndex, cartesian_index_to_dim, norm_validator
from ksim.vis import Marker

logger = logging.getLogger(__name__)

MonotonicFn = Literal["exp", "inv", "sigmoid"]


def exp_kernel(x: Array, scale: float) -> Array:
    return jnp.exp(-jnp.square(x) / (2 * scale**2))


def exp_kernel_with_penalty(x: Array, scale: float, sq_scale: float, abs_scale: float) -> Array:
    x_abs = jnp.abs(x)
    x_sq = jnp.square(x)
    x_exp = jnp.exp(-x_sq / (2 * scale**2))
    return x_abs * -abs_scale + x_sq * -sq_scale + x_exp


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
class LinearVelocityReward(Reward):
    """Penalty for how fast the robot is moving in the z-direction."""

    cmd: str = attrs.field()
    vel_length_scale: float = attrs.field(default=0.25)
    yaw_length_scale: float = attrs.field(default=0.25)
    zero_threshold: float = attrs.field(default=0.01)
    vis_height: float = attrs.field(default=0.6)
    sq_scale: float = attrs.field(default=0.1, validator=attrs.validators.gt(0.0))
    abs_scale: float = attrs.field(default=0.1, validator=attrs.validators.gt(0.0))

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

        vel_rews = exp_kernel_with_penalty(vel - cmd.vel, self.vel_length_scale, self.sq_scale, self.abs_scale)
        yaw_rews = exp_kernel_with_penalty(yaw - cmd.yaw, self.yaw_length_scale, self.sq_scale, self.abs_scale)
        x_rews = exp_kernel_with_penalty(x - cmd.xvel, self.vel_length_scale, self.sq_scale, self.abs_scale)
        y_rews = exp_kernel_with_penalty(y - cmd.yvel, self.vel_length_scale, self.sq_scale, self.abs_scale)

        return {
            "vel": vel_rews,
            "yaw": jnp.where(is_zero, 0.0, yaw_rews),
            "x": x_rews,
            "y": y_rews,
        }

    def get_markers(self, name: str) -> Collection[Marker]:
        return [LinearVelocityPenaltyMarker.get(height=self.vis_height)]


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityReward(Reward):
    """Penalty for how fast the robot is rotating in the xy-plane."""

    cmd: str = attrs.field()
    angvel_length_scale: float = attrs.field(default=0.25)
    sq_scale: float = attrs.field(default=0.1, validator=attrs.validators.gt(0.0))
    abs_scale: float = attrs.field(default=0.1, validator=attrs.validators.gt(0.0))

    def get_reward(self, trajectory: Trajectory) -> dict[str, Array]:
        cmd: AngularVelocityCommandValue = trajectory.command[self.cmd]
        angvel = trajectory.qvel[..., 5]
        angvel_rews = exp_kernel_with_penalty(angvel - cmd.vel, self.angvel_length_scale, self.sq_scale, self.abs_scale)
        return {"angvel": angvel_rews}


@attrs.define(frozen=True, kw_only=True)
class OffAxisVelocityReward(Reward):
    """Penalizes velocities in the off-command directions."""

    lin_length_scale: float = attrs.field(default=0.25)
    ang_length_scale: float = attrs.field(default=0.25)

    def get_reward(self, trajectory: Trajectory) -> dict[str, Array]:
        linz = trajectory.qvel[..., 2]
        angx = trajectory.qvel[..., 4]
        angy = trajectory.qvel[..., 5]
        linz_rew = jnp.exp(-jnp.square(linz) / (2 * self.lin_length_scale**2))
        angx_rew = jnp.exp(-jnp.square(angx) / (2 * self.ang_length_scale**2))
        angy_rew = jnp.exp(-jnp.square(angy) / (2 * self.ang_length_scale**2))
        return {
            "linz": linz_rew,
            "angx": angx_rew,
            "angy": angy_rew,
        }


@attrs.define(frozen=True, kw_only=True)
class BaseHeightReward(Reward):
    """Penalty for deviating from the base height target."""

    height_target: float = attrs.field()
    norm: xax.NormType = attrs.field(default="l1", validator=norm_validator)
    kernel_scale: float = attrs.field(default=0.25)

    def get_reward(self, trajectory: Trajectory) -> Array:
        base_height = trajectory.qpos[..., 2]
        reward = exp_kernel(base_height - self.height_target, self.kernel_scale)
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
class SmallJointVelocityReward(Reward):
    """Penalty for how fast the joint angular velocities are changing."""

    kernel_scale: float = attrs.field(default=0.25)

    def get_reward(self, trajectory: Trajectory) -> Array:
        qpos = trajectory.qpos[..., 7:]
        qpos_zp = jnp.pad(qpos, ((1, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done, ((1, 0),), mode="edge")[..., :-1, None]
        qvel = jnp.where(done, 0.0, qpos_zp[..., 1:, :] - qpos_zp[..., :-1, :])
        reward = exp_kernel(qvel, self.kernel_scale).mean(axis=-1)
        return reward


@attrs.define(frozen=True, kw_only=True)
class SmallJointAccelerationReward(Reward):
    """Penalty for high joint accelerations."""

    kernel_scale: float = attrs.field(default=0.25)

    def get_reward(self, trajectory: Trajectory) -> Array:
        qpos = trajectory.qpos[..., 7:]
        qpos_zp = jnp.pad(qpos, ((2, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done, ((2, 0),), mode="edge")[..., :-1, None]
        qvel = jnp.where(done, 0.0, qpos_zp[..., 1:, :] - qpos_zp[..., :-1, :])
        qacc = jnp.where(done[..., 1:, :], 0.0, qvel[..., 1:, :] - qvel[..., :-1, :])
        reward = exp_kernel(qacc, self.kernel_scale).mean(axis=-1)
        return reward


@attrs.define(frozen=True, kw_only=True)
class SmallJointJerkReward(Reward):
    """Penalty for high joint jerks."""

    kernel_scale: float = attrs.field(default=0.25)

    def get_reward(self, trajectory: Trajectory) -> Array:
        qpos = trajectory.qpos[..., 7:]
        qpos_zp = jnp.pad(qpos, ((3, 0), (0, 0)), mode="edge")
        done = jnp.pad(trajectory.done, ((3, 0),), mode="edge")[..., :-1, None]
        qvel = jnp.where(done, 0.0, qpos_zp[..., 1:, :] - qpos_zp[..., :-1, :])
        qacc = jnp.where(done[..., 1:, :], 0.0, qvel[..., 1:, :] - qvel[..., :-1, :])
        qjerk = jnp.where(done[..., 2:, :], 0.0, qacc[..., 1:, :] - qacc[..., :-1, :])
        reward = exp_kernel(qjerk, self.kernel_scale).mean(axis=-1)
        return reward


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
class SmallCtrlReward(Reward):
    """Penalty for large torque commands."""

    kernel_scale: float = attrs.field(default=0.25)
    scales: tuple[float, ...] | None = attrs.field(default=None)

    def get_reward(self, trajectory: Trajectory) -> Array:
        ctrl = trajectory.ctrl
        if self.scales is not None:
            ctrl = ctrl / jnp.array(self.scales)
        return exp_kernel(ctrl, self.kernel_scale).mean(axis=-1)

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
    kernel_scale: float = attrs.field(default=0.25)

    def get_reward(self, trajectory: Trajectory) -> Array:
        body_pos = trajectory.xpos[..., self.tracked_body_idx, :]
        base_pos = trajectory.xpos[..., self.base_body_idx, :]
        target_pos = trajectory.command[self.command_name][..., :3]
        error = xax.get_norm((body_pos - base_pos) - target_pos, self.norm).sum(-1)
        reward = exp_kernel(error, self.kernel_scale)
        return reward

    @classmethod
    def create(
        cls,
        model: PhysicsModel,
        command_name: str,
        tracked_body_name: str,
        base_body_name: str,
        norm: xax.NormType = "l1",
        kernel_scale: float = 0.25,
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
            kernel_scale=kernel_scale,
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

    max_air_time: float = attrs.field()
    max_ground_time: float = attrs.field()
    ctrl_dt: float = attrs.field()
    contact_obs: str = attrs.field()
    num_feet: int = attrs.field(default=2)
    bias: float = attrs.field(default=0.0)
    linvel_moving_threshold: float = attrs.field(default=0.05)
    angvel_moving_threshold: float = attrs.field(default=0.05)

    def initial_carry(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros((self.num_feet, 2), dtype=jnp.int32)

    def get_reward_stateful(
        self,
        trajectory: Trajectory,
        reward_carry: Array,
    ) -> tuple[Array, Array]:
        not_moving_lin = jnp.linalg.norm(trajectory.qvel[..., :2], axis=-1) < self.linvel_moving_threshold
        not_moving_ang = trajectory.qvel[..., 5] < self.angvel_moving_threshold
        not_moving = not_moving_lin & not_moving_ang

        contact_tcn = trajectory.obs[self.contact_obs] > 0.5  # Values are either 0 or 1.
        contact_tn = contact_tcn.any(axis=-2)
        chex.assert_shape(contact_tn, (..., self.num_feet))

        air_steps = round(self.max_air_time / self.ctrl_dt)
        gnd_steps = round(self.max_ground_time / self.ctrl_dt)

        def scan_fn(carry: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
            air_cnt_n, gnd_cnt_n = carry[..., 0], carry[..., 1]
            contact_n, stay_still = x
            air_cnt_n = jnp.where(stay_still | contact_n, 0, air_cnt_n + 1)
            gnd_cnt_n = jnp.where(stay_still | (~contact_n), 0, gnd_cnt_n + 1)
            counts_n2 = jnp.stack([air_cnt_n, gnd_cnt_n], axis=-1)
            return counts_n2, counts_n2

        reward_carry, count_tn2 = xax.scan(
            scan_fn,
            reward_carry,
            (contact_tn, not_moving | trajectory.done),
        )

        air_cnt_tn, gnd_cnt_tn = count_tn2[..., 0], count_tn2[..., 1]

        # Gradually increase reward until `threshold_steps`.
        air_rew_tn = (air_cnt_tn.astype(jnp.float32) / air_steps) + self.bias
        air_rew_tn = jnp.where((air_cnt_tn > 0) & (air_cnt_tn < air_steps), air_rew_tn, 0.0)
        air_rew_t = air_rew_tn.max(axis=-1)

        gnd_rew_tn = (gnd_cnt_tn.astype(jnp.float32) / gnd_steps) + self.bias
        gnd_rew_tn = jnp.where((gnd_cnt_tn > 0) & (gnd_cnt_tn < gnd_steps), gnd_rew_tn, 0.0)
        gnd_rew_t = gnd_rew_tn.max(axis=-1)

        reward_t = air_rew_t + gnd_rew_t
        return reward_t, reward_carry


@attrs.define(frozen=True, kw_only=True)
class FeetGroundedAtRestReward(StatefulReward):
    max_ground_time: float = attrs.field()
    ctrl_dt: float = attrs.field()
    contact_obs: str = attrs.field()
    num_feet: int = attrs.field(default=2)
    linvel_moving_threshold: float = attrs.field(default=0.05)
    angvel_moving_threshold: float = attrs.field(default=0.05)

    def initial_carry(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(self.num_feet, dtype=jnp.int32)

    def get_reward_stateful(
        self,
        trajectory: Trajectory,
        reward_carry: Array,
    ) -> tuple[Array, Array]:
        moving_lin = jnp.linalg.norm(trajectory.qvel[..., :2], axis=-1) > self.linvel_moving_threshold
        moving_ang = trajectory.qvel[..., 5] > self.angvel_moving_threshold
        moving = moving_lin | moving_ang

        contact_tcn = trajectory.obs[self.contact_obs] > 0.5  # Values are either 0 or 1.
        contact_tn = contact_tcn.any(axis=-2)
        chex.assert_shape(contact_tn, (..., self.num_feet))

        gnd_steps = round(self.max_ground_time / self.ctrl_dt)

        def scan_fn(carry: Array, x: Array) -> tuple[Array, Array]:
            gnd_cnt_n, reset_cnt = carry, x
            gnd_cnt_n = jnp.where(reset_cnt, 0, gnd_cnt_n + 1)
            return gnd_cnt_n, gnd_cnt_n

        reward_carry, count_tn = xax.scan(
            scan_fn,
            reward_carry,
            moving[..., None] | trajectory.done[..., None] | contact_tn,
        )

        # Gradually increase reward until `threshold_steps`.
        gnd_rew_tn = count_tn.astype(jnp.float32) / gnd_steps
        gnd_rew_tn = gnd_rew_tn.clip(0.0, 1.0)
        reward_t = gnd_rew_tn.sum(axis=-1)

        return reward_t, reward_carry


@attrs.define(kw_only=True)
class BodyHeightMarker(Marker):
    foot_id: int = attrs.field()
    obs_name: str = attrs.field()
    target_height: float | None = attrs.field()
    radius: float = attrs.field(default=0.1)
    size: float = attrs.field(default=0.03)
    cmd_name: str = attrs.field(default="sinusoidal_gait_command")

    def update(self, trajectory: Trajectory) -> None:
        """Visualizes the sinusoidal gait."""
        obs_x, obs_y, obs_z = trajectory.obs[self.obs_name][..., self.foot_id, :].tolist()
        self.pos = (obs_x, obs_y, obs_z if self.target_height is None else self.target_height)

    @classmethod
    def get(
        cls,
        foot_id: int,
        obs_name: str,
        target_height: float | None = None,
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
            target_height=target_height,
            rgba=(1.0, 0.0, 0.0, 1.0) if target_height is None else (0.0, 1.0, 0.0, 1.0),
            track_x=False,
            track_y=False,
            track_z=False,
            track_rotation=False,
        )


@attrs.define(frozen=True, kw_only=True)
class TargetHeightReward(Reward):
    """Reward for having some bodies be close to a target height."""

    position_obs: str = attrs.field()
    height: float = attrs.field()
    num_feet: int = attrs.field(default=2)
    linvel_moving_threshold: float = attrs.field(default=0.05)
    angvel_moving_threshold: float = attrs.field(default=0.05)
    kernel_scale: float = attrs.field(default=0.25)
    sq_scale: float = attrs.field(default=0.1, validator=attrs.validators.gt(0.0))
    abs_scale: float = attrs.field(default=0.1, validator=attrs.validators.gt(0.0))

    def get_reward(self, trajectory: Trajectory) -> Array:
        not_moving_lin = jnp.linalg.norm(trajectory.qvel[..., :2], axis=-1) < self.linvel_moving_threshold
        not_moving_ang = trajectory.qvel[..., 5] < self.angvel_moving_threshold
        not_moving = not_moving_lin & not_moving_ang
        cur_height_tn = trajectory.obs[self.position_obs][..., 2]
        penalty_tn = (cur_height_tn - self.height).clip(min=0.0)
        reward_tn = exp_kernel_with_penalty(penalty_tn, self.kernel_scale, self.sq_scale, self.abs_scale)
        reward_t = jnp.where(not_moving, 0.0, reward_tn.max(axis=-1))
        return reward_t

    def get_markers(self, name: str) -> Collection[Marker]:
        return [
            marker
            for foot_id in range(self.num_feet)
            for marker in (
                BodyHeightMarker.get(foot_id, obs_name=self.position_obs),
                BodyHeightMarker.get(foot_id, obs_name=self.position_obs, target_height=self.height),
            )
        ]


@attrs.define(frozen=True, kw_only=True)
class SparseTargetHeightReward(StatefulReward):
    """Reward for having some bodies be close to a target height."""

    contact_obs: str = attrs.field()
    position_obs: str = attrs.field()
    height: float = attrs.field()
    num_bodies: int = attrs.field(default=2)
    linvel_moving_threshold: float = attrs.field(default=0.05)
    angvel_moving_threshold: float = attrs.field(default=0.05)

    def initial_carry(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(self.num_bodies, dtype=jnp.float32)

    def get_reward_stateful(self, trajectory: Trajectory, reward_carry: Array) -> tuple[Array, Array]:
        not_moving_lin = jnp.linalg.norm(trajectory.qvel[..., :2], axis=-1) < self.linvel_moving_threshold
        not_moving_ang = trajectory.qvel[..., 5] < self.angvel_moving_threshold
        not_moving = not_moving_lin & not_moving_ang

        contact_tcn = trajectory.obs[self.contact_obs] > 0.5  # Values are either 0 or 1.
        contact_tn = contact_tcn.any(axis=-2)
        chex.assert_shape(contact_tn, (..., self.num_bodies))

        position_tn3 = trajectory.obs[self.position_obs]
        chex.assert_shape(position_tn3, (..., self.num_bodies, 3))

        # Give a sparse reward once the foot contacts the ground, equal to the
        # maximum height of the foot since the last contact, thresholded at the
        # target height.
        def scan_fn(carry: Array, x: tuple[Array, Array, Array]) -> tuple[Array, Array]:
            max_height_n, (contact_n, position_n3, not_moving) = carry, x
            height_n = position_n3[..., 2]
            reset = not_moving | contact_n
            reward_n = jnp.where(reset, max_height_n, 0.0).clip(max=self.height)
            max_height_n = jnp.maximum(max_height_n, height_n)
            max_height_n = jnp.where(reset, 0.0, max_height_n)
            return max_height_n, reward_n

        reward_carry, reward_tn = xax.scan(
            scan_fn,
            reward_carry,
            (contact_tn, position_tn3, not_moving | trajectory.done),
        )

        reward_t = reward_tn.max(axis=-1)
        return reward_t, reward_carry

    def get_markers(self, name: str) -> Collection[Marker]:
        return [
            marker
            for foot_id in range(self.num_bodies)
            for marker in (
                BodyHeightMarker.get(foot_id, obs_name=self.position_obs),
                BodyHeightMarker.get(foot_id, obs_name=self.position_obs, target_height=self.height),
            )
        ]


@attrs.define(frozen=True, kw_only=True)
class MotionlessAtRestPenalty(Reward):
    """Reward for feet either touching or not touching the ground for some time."""

    linvel_moving_threshold: float = attrs.field(default=0.05)
    angvel_moving_threshold: float = attrs.field(default=0.05)

    def get_reward(self, trajectory: Trajectory) -> Array:
        not_moving_lin = jnp.linalg.norm(trajectory.qvel[..., :2], axis=-1) < self.linvel_moving_threshold
        not_moving_ang = trajectory.qvel[..., 5] < self.angvel_moving_threshold
        not_moving = not_moving_lin & not_moving_ang

        joint_vel = trajectory.qvel[..., 6:]
        joint_vel_norm = jnp.linalg.norm(joint_vel, axis=-1)
        penalty = jnp.where(not_moving, joint_vel_norm, 0.0)
        return penalty


@attrs.define(frozen=True, kw_only=True)
class ForcePenalty(StatefulReward):
    """Reward for reducing the force on some body.

    This is modeled with a low-pass filter to simulate compliance, since when
    using stiff contacts the force can sometimes be very high.
    """

    force_obs: str = attrs.field()
    ctrl_dt: float = attrs.field()
    ema_time: float = attrs.field(default=0.03)
    ema_scale: float = attrs.field(default=0.001)
    num_feet: int = attrs.field(default=2)
    bias: float = attrs.field(default=0.0)

    def initial_carry(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(self.num_feet, dtype=jnp.float32)

    def get_reward_stateful(
        self,
        trajectory: Trajectory,
        reward_carry: Array,
    ) -> tuple[Array, Array]:
        alpha = jnp.exp(-self.ctrl_dt / self.ema_time)
        obs = (jnp.linalg.norm(trajectory.obs[self.force_obs], axis=-1) - self.bias).clip(min=0)

        def scan_fn(carry: Array, x: Array) -> tuple[Array, Array]:
            ema_n, obs_n = carry, x
            ema_n = alpha * ema_n + (1 - alpha) * obs_n
            return ema_n, ema_n

        ema_fn, ema_acc = xax.scan(scan_fn, reward_carry, obs)
        penalty = jnp.log1p(self.ema_scale * ema_acc).sum(axis=-1)
        return penalty, ema_fn


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
class JointPositionReward(Reward):
    """Reward for tracking the joint positions."""

    command_name: str = attrs.field()
    joint_indices: tuple[int, ...] = attrs.field()
    length_scale: float = attrs.field(default=0.25)
    sq_scale: float = attrs.field(default=0.1, validator=attrs.validators.gt(0.0))
    abs_scale: float = attrs.field(default=0.1, validator=attrs.validators.gt(0.0))

    def get_reward(self, trajectory: Trajectory) -> Array:
        if self.command_name not in trajectory.command:
            raise ValueError(f"Command {self.command_name} not found! Ensure that it is in the task.")
        cmd: JointPositionCommandValue = trajectory.command[self.command_name]
        trg = trajectory.qpos[..., self.joint_indices]
        diff = cmd.current_position - trg
        return exp_kernel_with_penalty(diff, self.length_scale, self.sq_scale, self.abs_scale).mean(axis=-1)

    @classmethod
    def create(
        cls,
        physics_model: PhysicsModel,
        joint_names: Collection[str],
        command_name: str,
        length_scale: float = 0.25,
        sq_scale: float = 0.1,
        abs_scale: float = 0.1,
        scale: float = 1.0,
        scale_by_curriculum: bool = False,
    ) -> Self:
        all_names = get_joint_names_in_order(physics_model)[1:]  # Remove floating base.
        for joint_name in joint_names:
            if joint_name not in all_names:
                raise ValueError(f"Joint {joint_name} not found in the model! Options are: {all_names}")

        joint_name_to_indices = {name: idx for idx, name in enumerate(all_names, start=7)}
        joint_indices = tuple(joint_name_to_indices[name] for name in joint_names)

        return cls(
            joint_indices=joint_indices,
            command_name=command_name,
            length_scale=length_scale,
            sq_scale=sq_scale,
            abs_scale=abs_scale,
            scale=scale,
            scale_by_curriculum=scale_by_curriculum,
        )
