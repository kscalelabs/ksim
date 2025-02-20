"""Defines a base interface for defining reward functions."""

import functools
import logging
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

import equinox as eqx
import jax.numpy as jnp
import xax
from brax.base import State
from brax.envs.base import State as BraxState
from brax.mjx.base import State as MjxState

from ksim.utils.data import BuilderData
from ksim.utils.mujoco import lookup_in_dict

logger = logging.getLogger(__name__)

NormType = Literal["l1", "l2"]


def get_norm(x: jnp.ndarray, norm: NormType) -> jnp.ndarray:
    match norm:
        case "l1":
            return jnp.abs(x)
        case "l2":
            return jnp.square(x)
        case _:
            raise ValueError(f"Invalid norm: {norm}")


class Reward(eqx.Module, ABC):
    """Base class for defining reward functions."""

    scale: float = eqx.field(static=True)

    def __init__(self, *, scale: float) -> None:
        self.scale = scale

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

    norm: NormType = eqx.field(default="l2", static=True)

    def __init__(
        self,
        *,
        scale: float,
        norm: NormType = "l2",
    ) -> None:
        super().__init__(scale)

        self.norm = norm

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        lin_vel_z = state.xd.vel[..., 0, 2]
        return get_norm(lin_vel_z, self.norm)


class AngularVelocityXYPenalty(Reward):
    """Penalty for how fast the robot is rotating in the xy-plane."""

    norm: NormType = eqx.field(default="l2", static=True)

    def __init__(
        self,
        *,
        scale: float,
        norm: NormType = "l2",
    ) -> None:
        super().__init__(scale)

        self.norm = norm

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        ang_vel_xy = state.xd.ang[..., 0, :2]
        return get_norm(ang_vel_xy, self.norm).sum(axis=-1)


class TrackAngularVelocityZReward(Reward):
    """Reward for how well the robot is tracking the angular velocity command."""

    cmd_name: str = eqx.field(static=True)
    norm: NormType = eqx.field(default="l2", static=True)

    def __init__(
        self,
        *,
        scale: float,
        cmd_name: str = "angular_velocity_command",
        norm: NormType = "l2",
    ) -> None:
        super().__init__(scale)

        self.cmd_name = cmd_name
        self.norm = norm

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        ang_vel_cmd_1 = prev_state.info["commands"][self.cmd_name][..., 0]
        ang_vel_z = state.xd.vel[..., 0, 1]
        return get_norm(ang_vel_z * ang_vel_cmd_1, self.norm)


class TrackLinearVelocityXYReward(Reward):
    """Reward for how well the robot is tracking the linear velocity command."""

    cmd_name: str = eqx.field(default="linear_velocity_command", static=True)
    norm: NormType = eqx.field(default="l2", static=True)

    def __init__(
        self,
        *,
        scale: float,
        cmd_name: str = "linear_velocity_command",
        norm: NormType = "l2",
    ) -> None:
        super().__init__(scale)

        self.cmd_name = cmd_name
        self.norm = norm

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        lin_vel_cmd_2 = prev_state.info["commands"][self.cmd_name]
        lin_vel_xy_2 = state.xd.vel[..., 0, :2]
        return get_norm(lin_vel_xy_2 * lin_vel_cmd_2, self.norm).sum(axis=-1)


class ActionSmoothnessPenalty(Reward):
    """Penalty for how smooth the robot's action is."""

    norm: NormType = eqx.field(default="l2", static=True)

    def __init__(
        self,
        *,
        scale: float,
        norm: NormType = "l2",
    ) -> None:
        super().__init__(scale)

        self.norm = norm

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        last_action = prev_state.info["last_action"]
        return get_norm(action - last_action, self.norm).sum(axis=-1)


class FootContactPenalty(Reward):
    """Penalty for how much the robot's foot is in contact with the ground.

    If the robot's foot is on the ground for more than `allowed_contact_prct`
    percent of the time, the penalty will be applied.

    We additionally specify a list of commands which, if set to zero, will
    cause the penalty to be ignored. This is to avoid penalizing foot contact
    if the robot is being commanded to stay still.
    """

    foot_id: int = eqx.field(static=True)
    allowed_contact_prct: float = eqx.field(static=True)
    contact_eps: float = eqx.field(default=1e-2, static=True)
    foot_name: str | None = eqx.field(default=None, static=True)
    skip_if_zero_command: list[str] = eqx.field(default_factory=lambda: [], static=True)
    eps: float = eqx.field(default=1e-6, static=True)

    def __init__(
        self,
        *,
        scale: float,
        foot_id: int,
        allowed_contact_prct: float,
        contact_eps: float = 1e-2,
        foot_name: str | None = None,
        skip_if_zero_command: list[str] | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(scale)

        assert 0 <= allowed_contact_prct <= 1
        assert skip_if_zero_command is None or len(skip_if_zero_command) > 0

        self.foot_id = foot_id
        self.allowed_contact_prct = allowed_contact_prct
        self.contact_eps = contact_eps
        self.foot_name = foot_name
        self.skip_if_zero_command = skip_if_zero_command
        self.eps = eps

    def __call__(self, prev_state: BraxState, action: jnp.ndarray, state: State) -> jnp.ndarray:
        if state.contact is None:
            return jnp.zeros_like(state.q[0])

        contact = state.contact

        if isinstance(state, MjxState):
            has_contact = (contact.geom == self.foot_id).any(axis=-1)
            penalty = jnp.where(has_contact, contact.dist, 1e4).min() <= self.contact_eps
            if self.skip_if_zero_command is not None:
                commands_are_zero = jnp.all(
                    jnp.stack(
                        [(prev_state.info["commands"][cmd] < self.eps).all() for cmd in self.skip_if_zero_command],
                        axis=-1,
                    ),
                    axis=-1,
                )
                penalty = jnp.where(commands_are_zero, 0, penalty)

            return penalty

        else:
            raise NotImplementedError(f"IllegalContactTermination is not implemented for {type(state)}")

    def post_accumulate(self, reward: jnp.ndarray) -> jnp.ndarray:
        # We only want to apply the penalty if the total contact time is more
        # than ``self.allowed_contact_prct`` percent of the time. Since the
        # reward tensor will be an array of zeros and ones, we can adjust by
        # the difference between the mean and ``self.allowed_contact_prct``.
        mean_contact = reward.mean()
        return (reward - (mean_contact + self.allowed_contact_prct)).clip(min=0)

    def get_name(self) -> str:
        base_name = super().get_name()
        return base_name if self.foot_name is None else f"{base_name}_{self.foot_name}"


class FootContactPenaltyBuilder(RewardBuilder[FootContactPenalty]):
    def __init__(
        self,
        *,
        scale: float,
        foot_name: str,
        allowed_contact_prct: float,
        contact_eps: float = 1e-2,
        skip_if_zero_command: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.scale = scale
        self.foot_name = foot_name
        self.allowed_contact_prct = allowed_contact_prct
        self.contact_eps = contact_eps
        self.skip_if_zero_command = skip_if_zero_command

    def __call__(self, data: BuilderData) -> FootContactPenalty:
        foot_id = lookup_in_dict([self.foot_name], data.body_name_to_idx, "Foot")[0]
        return FootContactPenalty(
            scale=self.scale,
            foot_id=foot_id,
            allowed_contact_prct=self.allowed_contact_prct,
            contact_eps=self.contact_eps,
            foot_name=self.foot_name,
            skip_if_zero_command=self.skip_if_zero_command,
        )
