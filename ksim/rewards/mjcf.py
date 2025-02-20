"""Defines reward functions to use with MJCF environments."""

import logging
from typing import Collection

import jax.numpy as jnp
from brax.base import State
from brax.envs.base import State as BraxState
from brax.mjx.base import State as MjxState

from ksim.rewards.base import Reward, RewardBuilder
from ksim.utils.data import BuilderData
from ksim.utils.mujoco import lookup_in_dict

logger = logging.getLogger(__name__)


class LinearVelocityZPenalty(Reward):
    """Penalty for how fast the robot is moving in the z-direction."""

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        lin_vel_z = state.xd.vel[..., 0, 2]
        return jnp.square(lin_vel_z)


class AngularVelocityXYPenalty(Reward):
    """Penalty for how fast the robot is rotating in the xy-plane."""

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        breakpoint()
        ang_vel_xy = state.xd.vel[..., 0, :2]
        return jnp.square(ang_vel_xy).sum(axis=-1)


class TrackLinearVelocityXYReward(Reward):
    """Reward for how well the robot is tracking the linear velocity command."""

    cmd_name: str

    def __init__(self, scale: float, cmd_name: str = "linear_velocity_command") -> None:
        super().__init__(scale)

        self.cmd_name = cmd_name

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        breakpoint()
        lin_vel_cmd_2 = state.info["commands"][self.cmd_name]
        lin_vel_xy = state.xd.vel[..., 0, :2]
        return jnp.sum(lin_vel_xy * lin_vel_cmd_2, axis=-1)


class TrackAngularVelocityZReward(Reward):
    """Reward for how well the robot is tracking the angular velocity command."""

    cmd_name: str

    def __init__(self, scale: float, cmd_name: str = "angular_velocity_command") -> None:
        super().__init__(scale)

        self.cmd_name = cmd_name

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        breakpoint()
        ang_vel_cmd_1 = state.info["commands"][self.cmd_name]
        ang_vel_z = state.xd.vel[..., 0, 1]
        return jnp.sum(ang_vel_z * ang_vel_cmd_1, axis=-1)


class FootSlipPenalty(Reward):
    """Penalty for how much the robot's foot is slipping."""

    foot_ids: jnp.ndarray

    def __init__(self, scale: float, foot_ids: Collection[int]) -> None:
        super().__init__(scale)

        self.foot_ids = jnp.array(sorted(foot_ids))

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        breakpoint()
        foot_vel_xy = state.xd.vel[..., 0, :2]
        return jnp.sum(foot_vel_xy, axis=-1)


class FootSlipPenaltyBuilder(RewardBuilder[FootSlipPenalty]):
    def __init__(self, scale: float, foot_names: Collection[str]) -> None:
        super().__init__()

        self.foot_names = foot_names
        self.scale = scale

    def __call__(self, data: BuilderData) -> FootSlipPenalty:
        foot_ids = lookup_in_dict(self.foot_names, data.body_name_to_idx, "Foot")
        return FootSlipPenalty(self.scale, foot_ids)


class ActionSmoothnessPenalty(Reward):
    """Penalty for how smooth the robot's action is."""

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        breakpoint()
        return jnp.sum(jnp.square(action[..., 1:] - action[..., :-1]), axis=-1)


class FootContactPenalty(Reward):
    """Penalty for how much the robot's foot is in contact with the ground.

    This penalty pushes the robot to keep at most `max_allowed_contact` feet
    on the ground at any given time.

    If `wait_steps` is greater than 0, the penalty will only be applied after
    more than `max_allowed_contact` feet have been in contact with the ground
    for `wait_steps` steps.
    """

    foot_ids: jnp.ndarray
    max_allowed_contact: int
    wait_steps: int

    def __init__(
        self,
        foot_ids: Collection[int],
        scale: float,
        max_allowed_contact: int | None = None,
        wait_steps: int = 0,
    ) -> None:
        super().__init__(scale)

        if max_allowed_contact is None:
            max_allowed_contact = len(foot_ids) - 1

        self.foot_ids = jnp.array(sorted(foot_ids))
        self.max_allowed_contact = max_allowed_contact
        self.wait_steps = wait_steps

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        if state.contact is None:
            return jnp.zeros_like(state.q[0])

        contact = state.contact

        if isinstance(state, MjxState):
            has_contact = jnp.any(contact.geom[:, :, None] == self.foot_ids[None, None, :], axis=(1, 2))
            return jnp.where(has_contact, contact.dist, 1e4).min() <= self.contact_eps

        else:
            raise NotImplementedError(f"IllegalContactTermination is not implemented for {type(state)}")

    def post_accumulate(self, reward: jnp.ndarray) -> jnp.ndarray:
        return reward


class FootContactPenaltyBuilder(RewardBuilder[FootContactPenalty]):
    def __init__(
        self,
        foot_names: Collection[str],
        scale: float,
        max_allowed_contact: int | None = None,
        wait_seconds: float = 0.0,
    ) -> None:
        super().__init__()

        self.foot_names = foot_names
        self.scale = scale
        self.max_allowed_contact = max_allowed_contact
        self.wait_seconds = wait_seconds

    def __call__(self, data: BuilderData) -> FootContactPenalty:
        foot_ids = lookup_in_dict(self.foot_names, data.body_name_to_idx, "Foot")
        wait_steps = int(self.wait_seconds / data.ctrl_dt)
        return FootContactPenalty(foot_ids, self.scale, self.max_allowed_contact, wait_steps)
