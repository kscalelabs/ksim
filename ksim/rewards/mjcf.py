"""Defines reward functions to use with MJCF environments."""

from typing import Generic, TypeVar

import jax.numpy as jnp

from ksim.rewards.base import Reward
from ksim.state.base import State

Tstate = TypeVar("Tstate", bound=State)


class LinearVelocityZPenalty(Reward[Tstate], Generic[Tstate,]):
    """Penalty for how fast the robot is moving in the z-direction."""

    def __call__(self, state: Tstate) -> jnp.ndarray:
        return jnp.square(state.base_lin_vel_n3[..., 2])


class ActionRatePenalty(Reward[Tstate], Generic[Tstate]):
    """Penalty for how fast the robot is changing its actions.

    This discourages the robot from changing it's actions too rapidly, making
    the learned policy less jerky.
    """

    scale: float

    def __call__(self, state: Tstate) -> jnp.ndarray:
        return jnp.square(state.actions_nj - state.last_actions_nj)


class GaitSymmetryReward(Reward[Tstate], Generic[Tstate]):
    """Reward function for how symmetrical the robot's gait is."""

    left_hip_pitch_index: int
    right_hip_pitch_index: int
    left_knee_index: int
    right_knee_index: int

    def __init__(
        self,
        scale: float,
        left_hip_index: int,
        right_hip_index: int,
        left_knee_index: int,
        right_knee_index: int,
    ) -> None:
        super().__init__(scale)
        self.left_hip_pitch_index = left_hip_index
        self.right_hip_pitch_index = right_hip_index
        self.left_knee_index = left_knee_index
        self.right_knee_index = right_knee_index

    def __call__(self, state: Tstate) -> jnp.ndarray:
        left_hip = state.dof_pos_nj[:, self.left_hip_pitch_index]
        right_hip = state.dof_pos_nj[:, self.right_hip_pitch_index]
        left_knee = state.dof_pos_nj[:, self.left_knee_index]
        right_knee = state.dof_pos_nj[:, self.right_knee_index]
        hip_symmetry = jnp.abs(left_hip - right_hip)
        knee_symmetry = jnp.abs(left_knee - right_knee)
        return jnp.exp(-(hip_symmetry + knee_symmetry))


class SimilarityToDefaultPenalty(Reward[Tstate], Generic[Tstate]):
    """Penalty for how similar the robot's current state is to the default state."""

    def __call__(self, state: Tstate) -> jnp.ndarray:
        return jnp.sum(jnp.abs(state.dof_pos_nj - state.default_dof_pos_nj), axis=1)


class BaseHeightPenalty(Reward[Tstate], Generic[Tstate]):
    """Penalty for how high the robot's base is."""

    base_height_target: float

    def __init__(self, scale: float, base_height_target: float) -> None:
        super().__init__(scale)
        self.base_height_target = base_height_target

    def __call__(self, state: Tstate) -> jnp.ndarray:
        return jnp.square(state.base_pos_n3[..., 2] - self.base_height_target)


class TrackingLinearVelocityXYReward(Reward[Tstate], Generic[Tstate]):
    """Reward function for how well the robot is tracking the commanded linear velocity."""

    tracking_sigma: float

    def __init__(self, scale: float, tracking_sigma: float) -> None:
        super().__init__(scale)
        self.tracking_sigma = tracking_sigma

    def __call__(self, state: Tstate) -> jnp.ndarray:
        lin_vel_error = jnp.sum(jnp.square(state.commands_n3[..., :2] - state.base_lin_vel_n3[..., :2]), axis=1)
        return jnp.exp(-lin_vel_error / self.tracking_sigma)


class TrackingAngularVelocityReward(Reward[Tstate], Generic[Tstate]):
    """Reward function for how well the robot is tracking the commanded angular velocity."""

    tracking_sigma: float

    def __init__(self, scale: float, tracking_sigma: float) -> None:
        super().__init__(scale)
        self.tracking_sigma = tracking_sigma

    def __call__(self, state: Tstate) -> jnp.ndarray:
        ang_vel_error = jnp.sum(jnp.square(state.commands_n3[..., 2] - state.base_ang_vel_n3[..., 2]), axis=1)
        return jnp.exp(-ang_vel_error / self.tracking_sigma)
