"""Defines walking-related reward functions."""

from typing import Generic, TypeVar

import jax.numpy as jnp

from ksim.rewards.base import Reward
from ksim.state.base import State

Tstate = TypeVar("Tstate", bound=State)


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
