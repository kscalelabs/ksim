"""Defines a base interface for defining reward functions."""

import functools
import logging
from abc import ABC, abstractmethod
from typing import Collection, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
import xax
from brax.base import State
from brax.envs.base import State as BraxState
from brax.mjx.base import State as MjxState

from ksim.utils.data import BuilderData
from ksim.utils.mujoco import lookup_in_dict

logger = logging.getLogger(__name__)


class Reward(eqx.Module, ABC):
    """Base class for defining reward functions."""

    scale: float

    def __init__(self, scale: float) -> None:
        self.scale = scale

        # Reward functions should end with either "Reward" or "Penalty", which
        # we use here to check if the scale is positive or negative.
        name = self.reward_name
        if name.lower().endswith("reward"):
            if self.scale < 0:
                logger.warning("Reward function %s has a negative scale: %f", name, self.scale)
        elif name.lower().endswith("penalty"):
            if self.scale > 0:
                logger.warning("Penalty function %s has a positive scale: %f", name, self.scale)
        else:
            logger.warning("Reward function %s does not end with 'Reward' or 'Penalty': %f", name, self.scale)

    def post_accumulate(self, reward: jnp.ndarray) -> jnp.ndarray:
        """Runs a post-epoch accumulation step.

        This function is called after the reward has been accumulated over the
        entire epoch. It can be used to normalize the reward, or apply some
        accumulation function - for example, you might might want to only
        start providing the reward or penalty after a certain number of steps
        have passed.

        Args:
            reward: The accumulated reward over the epoch.
        """
        return reward

    @abstractmethod
    def __call__(self, prev_state: BraxState, action: jnp.ndarray, next_state: State) -> jnp.ndarray: ...

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def reward_name(self) -> str:
        return self.get_name()


T = TypeVar("T", bound=Reward)


class RewardBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, data: BuilderData) -> T:
        """Builds a reward from a MuJoCo model.

        Args:
            data: The data to build the reward from.

        Returns:
            A reward that can be applied to a state.
        """


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

    cmd_name: str = eqx.field(static=True)

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

    cmd_name: str = eqx.field(static=True)

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

    foot_ids: jnp.ndarray = eqx.field(static=True)

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

    foot_ids: jnp.ndarray = eqx.field(static=True)
    max_allowed_contact: int = eqx.field(static=True)
    wait_steps: int = eqx.field(static=True)

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
