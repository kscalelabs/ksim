"""Defines reward functions to use with MJCF environments."""

import logging
from typing import Collection

import jax.numpy as jnp
from brax.base import State
from brax.envs.base import State as BraxState
from brax.mjx.base import State as MjxState

from ksim.rewards.base import Reward, RewardBuilder
from ksim.utils.data import BuilderData
from ksim.utils.mujoco import link_names_to_ids

logger = logging.getLogger(__name__)


class LinearVelocityZPenalty(Reward):
    """Penalty for how fast the robot is moving in the z-direction."""

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        lin_vel_z = state.xd.vel[..., 0, 2]
        return jnp.square(lin_vel_z)


class FootContactPenalty(Reward):
    """Penalty for how much the robot's foot is in contact with the ground.

    This penalty pushes the robot to keep at most `max_allowed_contact` feet
    on the ground at any given time.

    If `wait_steps` is greater than 0, the penalty will only be applied after
    more than `max_allowed_contact` feet have been in contact with the ground
    for `wait_steps` steps.
    """

    link_ids: jnp.ndarray
    max_allowed_contact: int
    wait_steps: int

    def __init__(
        self,
        link_ids: Collection[int],
        scale: float,
        max_allowed_contact: int | None = None,
        wait_steps: int = 0,
    ) -> None:
        super().__init__(scale)

        if max_allowed_contact is None:
            max_allowed_contact = len(link_ids) - 1

        self.link_ids = jnp.array(sorted(link_ids))
        self.max_allowed_contact = max_allowed_contact
        self.wait_steps = wait_steps

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        if state.contact is None:
            return jnp.zeros_like(state.q[0])

        contact = state.contact

        if isinstance(state, MjxState):
            has_contact = jnp.any(contact.geom[:, :, None] == self.link_ids[None, None, :], axis=(1, 2))
            return jnp.where(has_contact, contact.dist, 1e4).min() <= self.contact_eps

        else:
            raise NotImplementedError(f"IllegalContactTermination is not implemented for {type(state)}")

    def post_accumulate(self, reward: jnp.ndarray) -> jnp.ndarray:
        return reward


class FootContactPenaltyBuilder(RewardBuilder[FootContactPenalty]):
    def __init__(
        self,
        link_names: Collection[str],
        scale: float,
        max_allowed_contact: int | None = None,
        wait_seconds: float = 0.0,
    ) -> None:
        super().__init__()

        self.link_names = link_names
        self.scale = scale
        self.max_allowed_contact = max_allowed_contact
        self.wait_seconds = wait_seconds

    def __call__(self, data: BuilderData) -> FootContactPenalty:
        link_ids = link_names_to_ids(self.link_names, data.model)
        wait_steps = int(self.wait_seconds / data.ctrl_dt)
        return FootContactPenalty(link_ids, self.scale, self.max_allowed_contact, wait_steps)
