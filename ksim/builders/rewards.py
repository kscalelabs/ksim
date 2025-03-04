"""Defines a base interface for defining reward functions."""

import functools
import logging
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

import attrs
import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
import xax
from flax.core import FrozenDict
from jaxtyping import Array

from ksim.utils.data import BuilderData

logger = logging.getLogger(__name__)

NormType = Literal["l1", "l2"]


def get_norm(x: Array, norm: NormType) -> Array:
    match norm:
        case "l1":
            return jnp.abs(x)
        case "l2":
            return jnp.square(x)
        case _:
            raise ValueError(f"Invalid norm: {norm}")


@attrs.define(frozen=True, kw_only=True)
class Reward(ABC):
    """Base class for defining reward functions."""

    scale: float

    def __post_init__(self) -> None:
        # Reward function classes should end with either "Reward" or "Penalty",
        # which we use here to check if the scale is positive or negative.
        name = self.__class__.__name__.lower()
        if name.lower().endswith("reward"):
            if self.scale < 0:
                logger.warning("Reward function %s has a negative scale: %f", name, self.scale)
        elif name.lower().endswith("penalty"):
            if self.scale > 0:
                logger.warning("Penalty function %s has a positive scale: %f", name, self.scale)
        else:
            logger.warning(
                "Reward function %s does not end with 'Reward' or 'Penalty': %f", name, self.scale
            )

    def post_accumulate(self, reward: Array) -> Array:
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
    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array: ...

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


@attrs.define(frozen=True, kw_only=True)
class HeightReward(Reward):
    """Reward for how high the robot is."""

    height_target: float = attrs.field(default=1.4)

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        height = mjx_data_t_plus_1.qpos[2]
        reward = jnp.exp(-jnp.abs(height - self.height_target) * 50)
        # jax.debug.print("height: {height}", height=height)
        # jax.debug.print("reward: {reward}", reward=reward)
        return reward


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityZPenalty(Reward):
    """Penalty for how fast the robot is moving in the z-direction."""

    norm: NormType = attrs.field(default="l2")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        lin_vel_z = mjx_data_t_plus_1.qvel[2]
        return get_norm(lin_vel_z, self.norm)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityXYPenalty(Reward):
    """Penalty for how fast the robot is rotating in the xy-plane."""

    norm: NormType = attrs.field(default="l2")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        ang_vel_xy = mjx_data_t_plus_1.qvel[3:5]
        return get_norm(ang_vel_xy, self.norm).sum(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class TrackAngularVelocityZReward(Reward):
    """Reward for how well the robot is tracking the angular velocity command."""

    cmd_name: str = attrs.field(default="angular_velocity_command")
    norm: NormType = attrs.field(default="l2")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        ang_vel_cmd_1 = command_t[self.cmd_name][0]
        ang_vel_z = mjx_data_t_plus_1.qvel[5]
        return get_norm(ang_vel_z * ang_vel_cmd_1, self.norm)


@attrs.define(frozen=True, kw_only=True)
class TrackLinearVelocityXYReward(Reward):
    """Reward for how well the robot is tracking the linear velocity command."""

    cmd_name: str = attrs.field(default="linear_velocity_command")
    norm: NormType = attrs.field(default="l2")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        lin_vel_cmd_2 = command_t[self.cmd_name]
        lin_vel_xy_2 = mjx_data_t_plus_1.qvel[:2]
        return get_norm(lin_vel_xy_2 * lin_vel_cmd_2, self.norm).sum(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class ActionSmoothnessPenalty(Reward):
    """Penalty for how smooth the robot's action is."""

    norm: NormType = attrs.field(default="l2")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        normed_action_smoothness = get_norm(action_t - action_t_minus_1, self.norm).sum(axis=-1)
        return jax.lax.select(
            action_t_minus_1 is None,
            jnp.zeros_like(normed_action_smoothness),
            normed_action_smoothness,
        )


@attrs.define(frozen=True, kw_only=True)
class FootContactPenalty(Reward):
    """Penalty for how much the robot's foot is in contact with the ground.

    If the robot's foot is on the ground for more than `allowed_contact_prct`
    percent of the time, the penalty will be applied.

    We additionally specify a list of commands which, if set to zero, will
    cause the penalty to be ignored. This is to avoid penalizing foot contact
    if the robot is being commanded to stay still.
    """

    illegal_geom_idxs: Array
    allowed_contact_prct: float
    contact_eps: float = attrs.field(default=1e-2)
    skip_if_zero_command: list[str] = attrs.field(factory=list)
    eps: float = attrs.field(default=1e-6)

    def __post_init__(self) -> None:
        assert 0 <= self.allowed_contact_prct <= 1
        if len(self.skip_if_zero_command) == 0 and self.skip_if_zero_command is not None:
            assert False, "skip_if_zero_command should be None or non-empty"

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        has_contact_1 = jnp.isin(mjx_data_t_plus_1.contact.geom1, self.illegal_geom_idxs)
        has_contact_2 = jnp.isin(mjx_data_t_plus_1.contact.geom2, self.illegal_geom_idxs)
        has_contact = jnp.logical_or(has_contact_1, has_contact_2)

        # Handle case where there might be no contacts or no matches
        distances_where_contact = jnp.where(has_contact, mjx_data_t_plus_1.contact.dist, 1e4)
        min_distance = jnp.min(distances_where_contact, initial=1e4)
        penalty = (min_distance <= self.contact_eps).astype(jnp.float32)

        if self.skip_if_zero_command:
            commands_are_zero = jnp.stack(
                [(command_t[cmd] < self.eps).all() for cmd in self.skip_if_zero_command],
                axis=0,
            )
            penalty = jnp.where(commands_are_zero, 0.0, penalty).sum()

        return penalty

    def post_accumulate(self, reward: Array) -> Array:
        # We only want to apply the penalty if the total contact time is more
        # than ``self.allowed_contact_prct`` percent of the time. Since the
        # reward tensor will be an array of zeros and ones, we can adjust by
        # the difference between the mean and ``self.allowed_contact_prct``.
        mean_contact = reward.mean()
        multiplier = 1 - (mean_contact - self.allowed_contact_prct).clip(min=0)
        return reward * multiplier

    def get_name(self) -> str:
        return super().get_name()


@attrs.define(frozen=True, kw_only=True)
class FootContactPenaltyBuilder(RewardBuilder[FootContactPenalty]):
    scale: float
    foot_body_names: list[str]
    allowed_contact_prct: float
    contact_eps: float = attrs.field(default=1e-2)
    skip_if_zero_command: list[str] | None = attrs.field(default=None)

    def __call__(self, data: BuilderData) -> FootContactPenalty:
        illegal_geom_idxs = []
        for geom_idx, body_name in data.mujoco_mappings.geom_idx_to_body_name.items():
            if body_name in self.foot_body_names:
                illegal_geom_idxs.append(geom_idx)

        illegal_geom_idxs = jnp.array(illegal_geom_idxs)

        return FootContactPenalty(
            scale=self.scale,
            illegal_geom_idxs=illegal_geom_idxs,
            allowed_contact_prct=self.allowed_contact_prct,
            contact_eps=self.contact_eps,
            skip_if_zero_command=self.skip_if_zero_command if self.skip_if_zero_command else [],
        )
