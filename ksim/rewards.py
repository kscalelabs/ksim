"""Defines a base interface for defining reward functions."""

import functools
import logging
from abc import ABC, abstractmethod
from typing import Callable, Generic, TypeVar

import attrs
import jax
import jax.numpy as jnp
import xax
from flax.core import FrozenDict
from jaxtyping import Array
from mujoco import mjx

from ksim.utils.data import BuilderData
from ksim.utils.mujoco import geoms_colliding

logger = logging.getLogger(__name__)


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

    # TODO: Use post_accumulate after unrolling
    def post_accumulate(self, reward: Array, done: Array) -> Array:
        """Runs a post-epoch accumulation step.

        This function is called after the reward has been accumulated over the
        entire epoch. It can be used to normalize the reward, or apply some
        accumulation function - for example, you might want to only
        start providing the reward or penalty after a certain number of steps
        have passed.

        Args:
            reward: The accumulated reward over the epoch.
            done: Array of episode termination flags.
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
        done_t: Array,
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
class TerminationPenalty(Reward):
    """Penalty for terminating the episode."""

    scale: float = attrs.field(default=-1.0)

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        return jnp.sum(done_t.any())


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
        done_t: Array,
    ) -> Array:
        height = mjx_data_t_plus_1.qpos[2]
        reward = jnp.exp(-jnp.abs(height - self.height_target) * 50)
        return reward


@attrs.define(frozen=True, kw_only=True)
class OrientationPenalty(Reward):
    """Penalty for how well the robot is oriented."""

    norm: xax.NormType = attrs.field(default="l2")
    target_orientation: list = attrs.field(default=[0.073, 0.0, 1.0])

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        return jnp.sum(
            xax.get_norm(
                xax.quat_to_euler(mjx_data_t_plus_1.qpos[3:7]) - jnp.array(self.target_orientation),
                self.norm,
            )
        )


@attrs.define(frozen=True, kw_only=True)
class TorquePenalty(Reward):
    """Penalty for high torques."""

    norm: xax.NormType = attrs.field(default="l1")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        return jnp.sum(xax.get_norm(mjx_data_t_plus_1.actuator_force, self.norm))


@attrs.define(frozen=True, kw_only=True)
class EnergyPenalty(Reward):
    """Penalty for high energies."""

    norm: xax.NormType = attrs.field(default="l1")

    # NOTE: I think this is actually penalizing power (?). Rename if needed
    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        return jnp.sum(
            xax.get_norm(mjx_data_t_plus_1.qvel[6:], self.norm)
            * xax.get_norm(mjx_data_t_plus_1.actuator_force, self.norm)
        )


@attrs.define(frozen=True, kw_only=True)
class JointAccelerationPenalty(Reward):
    """Penalty for high joint accelerations."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        return jnp.sum(xax.get_norm(mjx_data_t_plus_1.qacc[6:], self.norm))


@attrs.define(frozen=True, kw_only=True)
class LinearVelocityZPenalty(Reward):
    """Penalty for how fast the robot is moving in the z-direction."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        lin_vel_z = mjx_data_t_plus_1.qvel[2]
        return xax.get_norm(lin_vel_z, self.norm)


@attrs.define(frozen=True, kw_only=True)
class AngularVelocityXYPenalty(Reward):
    """Penalty for how fast the robot is rotating in the xy-plane."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        ang_vel_xy = mjx_data_t_plus_1.qvel[3:5]
        return xax.get_norm(ang_vel_xy, self.norm).sum(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class TrackAngularVelocityZReward(Reward):
    """Reward for how well the robot is tracking the angular velocity command."""

    cmd_name: str = attrs.field(default="angular_velocity_command_vector")
    norm: xax.NormType = attrs.field(default="l2")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        ang_vel_cmd_1 = command_t[self.cmd_name][0]
        ang_vel_z = mjx_data_t_plus_1.qvel[5]
        return xax.get_norm(ang_vel_z * ang_vel_cmd_1, self.norm)


@attrs.define(frozen=True, kw_only=True)
class TrackLinearVelocityXYReward(Reward):
    """Reward for how well the robot is tracking the linear velocity command."""

    cmd_name: str = attrs.field(default="linear_velocity_command_vector")
    sensitivity: float = attrs.field(default=1.0)

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        cmd_vel_xy = command_t[self.cmd_name]
        actual_vel_xy = mjx_data_t_plus_1.qvel[:2]

        # Compute tracking error as L2 distance between commanded and actual velocity
        tracking_error = jnp.linalg.norm(cmd_vel_xy - actual_vel_xy)

        # Convert error to reward in [0,1] range using exponential decay
        # sensitivity controls how quickly reward decays with error
        # Higher sensitivity = sharper decay = more demanding tracking
        tracking_reward = jnp.exp(-self.sensitivity * tracking_error)

        return tracking_reward


@attrs.define(frozen=True, kw_only=True)
class ActionSmoothnessPenalty(Reward):
    """Penalty for how smooth the robot's action is."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        # During tracing, both branches of jax.lax.cond are evaluated, so
        # we need to handle the case where action_t_minus_1 is None.
        # This only works if action_t_minus_1 is statically None or not None.
        if action_t_minus_1 is None:
            return jnp.zeros_like(xax.get_norm(action_t, self.norm).sum(axis=-1))
        return xax.get_norm(action_t - action_t_minus_1, self.norm).sum(axis=-1)


@attrs.define(frozen=True, kw_only=True)
class FootSlipPenalty(Reward):
    """Penalty for horizontal movement while feet are contacting the floor."""

    foot_geom_idxs: Array
    floor_idx: int

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        contacts = jnp.array(
            [
                geoms_colliding(mjx_data_t_plus_1, geom_idx, self.floor_idx)
                for geom_idx in self.foot_geom_idxs
            ]
        )

        # Get x and y velocities
        body_vel = mjx_data_t_plus_1.qvel[:2]

        return jnp.sum(jnp.linalg.norm(body_vel, axis=-1) * contacts)


@attrs.define(frozen=True, kw_only=True)
class FootSlipPenaltyBuilder(RewardBuilder[FootSlipPenalty]):
    scale: float
    foot_geom_names: list[str]

    def __call__(self, data: BuilderData) -> FootSlipPenalty:
        illegal_geom_idxs = []
        for geom_name in self.foot_geom_names:
            illegal_geom_idxs.append(data.mujoco_mappings.geom_name_to_idx[geom_name])

        illegal_geom_idxs = jnp.array(illegal_geom_idxs)
        floor_idx = data.mujoco_mappings.floor_geom_idx
        if floor_idx is None:
            raise ValueError("No floor geom found in model")

        return FootSlipPenalty(
            scale=self.scale,
            foot_geom_idxs=illegal_geom_idxs,
            floor_idx=floor_idx,
        )


# TODO: Look into using bodies instead of geoms where appropriate
@attrs.define(frozen=True, kw_only=True)
class FeetClearancePenalty(Reward):
    """Penalty for deviation from desired feet clearance."""

    foot_geom_idxs: Array
    max_foot_height: float
    norm: xax.NormType = attrs.field(default="l1")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        feet_heights = mjx_data_t_plus_1.geom_xpos[self.foot_geom_idxs][:, 2]

        # TODO: Look into adding linear feet velocity norm to scale the foot delta

        return jnp.sum(xax.get_norm(feet_heights - self.max_foot_height, self.norm))


@attrs.define(frozen=True, kw_only=True)
class FeetClearancePenaltyBuilder(RewardBuilder[FeetClearancePenalty]):
    scale: float
    foot_geom_names: list[str]
    max_foot_height: float

    def __call__(self, data: BuilderData) -> FeetClearancePenalty:
        illegal_geom_idxs = []
        for geom_name in self.foot_geom_names:
            illegal_geom_idxs.append(data.mujoco_mappings.geom_name_to_idx[geom_name])
        illegal_geom_idxs = jnp.array(illegal_geom_idxs)

        return FeetClearancePenalty(
            scale=self.scale,
            foot_geom_idxs=illegal_geom_idxs,
            max_foot_height=self.max_foot_height,
        )


@attrs.define(frozen=True, kw_only=True)
class FeetAirTimeReward(Reward):
    """Reward for how much the robot's feet are in the air.

    Rewards proper walking gait by encouraging exactly one foot
    to be in contact with the ground at any time.
    """

    left_foot_geom_idxs: Array
    right_foot_geom_idxs: Array
    contact_eps: float = attrs.field(default=1e-2)
    skip_if_zero_command: tuple[str, ...] = attrs.field(factory=tuple)
    eps: float = attrs.field(default=1e-6)
    single_contact_reward: float = attrs.field(default=1.0)
    no_contact_penalty: float = attrs.field(default=0.5)
    all_contact_penalty: float = attrs.field(default=0.8)

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        # Check if any left foot geom is in contact
        left_foot_in_contact = jnp.zeros((), dtype=jnp.bool_)
        for foot_idx in self.left_foot_geom_idxs:
            has_contact_1 = mjx_data_t_plus_1.contact.geom1 == foot_idx
            has_contact_2 = mjx_data_t_plus_1.contact.geom2 == foot_idx
            has_foot_contact = jnp.logical_or(has_contact_1, has_contact_2)

            # Check if any contact for this foot is close enough
            distances_where_contact = jnp.where(
                has_foot_contact, mjx_data_t_plus_1.contact.dist, 1e4
            )
            min_distance = jnp.min(distances_where_contact, initial=1e4)
            this_geom_in_contact = min_distance <= self.contact_eps
            left_foot_in_contact = jnp.logical_or(left_foot_in_contact, this_geom_in_contact)

        right_foot_in_contact = jnp.zeros((), dtype=jnp.bool_)
        for foot_idx in self.right_foot_geom_idxs:
            has_contact_1 = mjx_data_t_plus_1.contact.geom1 == foot_idx
            has_contact_2 = mjx_data_t_plus_1.contact.geom2 == foot_idx
            has_foot_contact = jnp.logical_or(has_contact_1, has_contact_2)

            distances_where_contact = jnp.where(
                has_foot_contact, mjx_data_t_plus_1.contact.dist, 1e4
            )
            min_distance = jnp.min(distances_where_contact, initial=1e4)
            this_geom_in_contact = min_distance <= self.contact_eps
            right_foot_in_contact = jnp.logical_or(right_foot_in_contact, this_geom_in_contact)

        # Count how many feet are in contact
        num_feet_in_contact = (left_foot_in_contact + right_foot_in_contact).astype(jnp.float32)

        # Determine reward based on contact pattern:
        # - Reward for exactly one foot in contact
        # - Penalty for no feet in contact
        # - Penalty for all feet in contact
        reward = jnp.where(
            num_feet_in_contact == 1,
            self.single_contact_reward,  # One foot in contact
            jnp.where(
                num_feet_in_contact == 0,
                -self.no_contact_penalty,  # No feet in contact - revisit as reward
                -self.all_contact_penalty,  # Multiple feet in contact
            ),
        )

        # Skip if commanded to stand still
        if self.skip_if_zero_command:
            commands_are_zero = jnp.stack(
                [(command_t[cmd] < self.eps).all() for cmd in self.skip_if_zero_command],
                axis=0,
            )
            reward = jnp.where(jnp.any(commands_are_zero), 0.0, reward)

        return reward

    def post_accumulate(self, reward: Array, done: Array) -> Array:
        # Identify consecutive positive rewards (correct foot contacts)
        # We're looking at the full sequence of rewards from the episode

        # Create a mask for correct contacts (where reward > 0)
        correct_contacts = reward > 0

        # Calculate streak lengths at each timestep, reset on episode boundaries (done=True)
        def count_streak(
            carry: Array, inputs: tuple[Array, Array]
        ) -> tuple[Array, tuple[Array, Array]]:
            streak, x, is_done = carry, inputs[0], inputs[1]
            # Reset streak to 0 when done is True
            streak = jnp.where(is_done, 0, streak)
            # Increment streak only for correct contacts
            streak = jnp.where(x, streak + 1, 0)
            return streak, (streak, is_done)

        # Create inputs for scan: (correct_contacts, done)
        scan_inputs = (correct_contacts, done)

        # Initialize streak counter at 0
        _, (streaks, _) = jax.lax.scan(count_streak, jnp.array(0), scan_inputs)
        # Apply exponential scaling to the base reward based on streak length
        # Start with a moderate multiplier that grows with streak length
        streak_multiplier = jnp.minimum(1.0 + 0.1 * streaks, 2.0)  # Cap at 2x reward

        # Apply the multiplier to positive rewards only
        scaled_reward = jnp.where(reward > 0, reward * streak_multiplier, reward)

        return scaled_reward

    def __hash__(self) -> int:
        return hash(
            (
                self.left_foot_geom_idxs,
                self.right_foot_geom_idxs,
                self.contact_eps,
                self.skip_if_zero_command,
                self.eps,
                self.single_contact_reward,
                self.no_contact_penalty,
                self.all_contact_penalty,
            )
        )

    def get_name(self) -> str:
        return super().get_name()


@attrs.define(frozen=True, kw_only=True)
class FeetAirTimeRewardBuilder(RewardBuilder[FeetAirTimeReward]):
    scale: float
    left_foot_geom_names: list[str]
    right_foot_geom_names: list[str]
    contact_eps: float = attrs.field(default=1e-2)
    skip_if_zero_command: tuple[str, ...] | None = attrs.field(default=None)
    single_contact_reward: float = attrs.field(default=1.0)
    no_contact_penalty: float = attrs.field(default=0.5)
    all_contact_penalty: float = attrs.field(default=0.8)

    def __call__(self, data: BuilderData) -> FeetAirTimeReward:
        left_foot_geom_idxs = []
        for geom_name in self.left_foot_geom_names:
            try:
                left_foot_geom_idxs.append(data.mujoco_mappings.geom_name_to_idx[geom_name])
            except KeyError:
                raise ValueError(
                    f"Geom '{geom_name}' not found in model. "
                    f"Available geoms: {data.mujoco_mappings.geom_name_to_idx.keys()}"
                )

        left_foot_geom_idxs = jnp.array(left_foot_geom_idxs)
        right_foot_geom_idxs = []
        for geom_name in self.right_foot_geom_names:
            try:
                right_foot_geom_idxs.append(data.mujoco_mappings.geom_name_to_idx[geom_name])
            except KeyError:
                raise ValueError(
                    f"Geom '{geom_name}' not found in model. "
                    f"Available geoms: {data.mujoco_mappings.geom_name_to_idx.keys()}"
                )

        right_foot_geom_idxs = jnp.array(right_foot_geom_idxs)

        return FeetAirTimeReward(
            scale=self.scale,
            left_foot_geom_idxs=left_foot_geom_idxs,
            right_foot_geom_idxs=right_foot_geom_idxs,
            contact_eps=self.contact_eps,
            skip_if_zero_command=self.skip_if_zero_command if self.skip_if_zero_command else (),
            single_contact_reward=self.single_contact_reward,
            no_contact_penalty=self.no_contact_penalty,
            all_contact_penalty=self.all_contact_penalty,
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
    skip_if_zero_command: tuple[str, ...] = attrs.field(factory=tuple)
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
        done_t: Array,
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

    def post_accumulate(self, reward: Array, done: Array) -> Array:
        # We only want to apply the penalty if the total contact time is more
        # than ``self.allowed_contact_prct`` percent of the time. Since the
        # reward tensor will be an array of zeros and ones, we can adjust by
        # the difference between the mean and ``self.allowed_contact_prct``.
        mean_contact = reward.mean()
        multiplier = 1 - (mean_contact - self.allowed_contact_prct).clip(min=0)
        return reward * multiplier

    def __hash__(self) -> int:
        return hash(
            (
                self.allowed_contact_prct,
                self.contact_eps,
                self.skip_if_zero_command,
                self.eps,
            )
        )

    def get_name(self) -> str:
        return super().get_name()


@attrs.define(frozen=True, kw_only=True)
class FootContactPenaltyBuilder(RewardBuilder[FootContactPenalty]):
    scale: float
    foot_geom_names: list[str]
    allowed_contact_prct: float
    contact_eps: float = attrs.field(default=1e-2)
    skip_if_zero_command: tuple[str, ...] | None = attrs.field(default=None)

    def __call__(self, data: BuilderData) -> FootContactPenalty:
        illegal_geom_idxs = []
        for geom_name in self.foot_geom_names:
            illegal_geom_idxs.append(data.mujoco_mappings.geom_name_to_idx[geom_name])

        illegal_geom_idxs = jnp.array(illegal_geom_idxs)

        return FootContactPenalty(
            scale=self.scale,
            illegal_geom_idxs=illegal_geom_idxs,
            allowed_contact_prct=self.allowed_contact_prct,
            contact_eps=self.contact_eps,
            skip_if_zero_command=self.skip_if_zero_command if self.skip_if_zero_command else (),
        )


@attrs.define(frozen=True, kw_only=True)
class DefaultPoseDeviationPenalty(Reward):
    """Penalty for deviating from a default/reference pose.

    Penalizes joint positions that deviate from specified default values.
    This helps maintain proper posture during movement.
    """

    joint_indices: Array
    default_positions: Array
    joint_deviation_weights: Array
    norm: xax.NormType = attrs.field(default="l2")
    exclude_base_pose: bool = attrs.field(default=True)

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        current_positions = mjx_data_t_plus_1.qpos[self.joint_indices]
        deviations = current_positions - self.default_positions
        weighted_deviations = deviations * self.joint_deviation_weights
        return jnp.sum(xax.get_norm(weighted_deviations, self.norm))


@attrs.define(frozen=True, kw_only=True)
class DefaultPoseDeviationPenaltyBuilder(RewardBuilder[DefaultPoseDeviationPenalty]):
    scale: float
    default_positions: dict[str, float]
    deviation_weights: dict[str, float]
    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, data: BuilderData) -> DefaultPoseDeviationPenalty:
        # Convert joint names to indices
        joint_indices = []
        default_positions_list = []
        joint_deviation_weights = []

        for joint_name, position in self.default_positions.items():
            try:
                idx_range = data.mujoco_mappings.qpos_name_to_idx_range[joint_name]
                start_idx = idx_range[0]
                joint_indices.append(start_idx)
                default_positions_list.append(position)
                joint_deviation_weights.append(self.deviation_weights[joint_name])
            except KeyError:
                raise ValueError(f"Joint '{joint_name}' not found in model")

        joint_indices_array = jnp.array(joint_indices)
        default_positions_array = jnp.array(default_positions_list)
        joint_deviation_weights_array = jnp.array(joint_deviation_weights)

        return DefaultPoseDeviationPenalty(
            scale=self.scale,
            joint_indices=joint_indices_array,
            default_positions=default_positions_array,
            joint_deviation_weights=joint_deviation_weights_array,
            norm=self.norm,
        )


@attrs.define(frozen=True, kw_only=True)
class JointPosLimitPenalty(Reward):
    """Penalty for joint positions exceeding soft limits.

    Penalizes joint positions that exceed specified soft lower and upper bounds.
    This encourages the robot to stay within a safe range of motion and avoid
    hitting hard joint limits which can cause instability.
    """

    joint_indices: Array
    soft_lower_limits: Array
    soft_upper_limits: Array

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        # Get current joint positions
        joint_positions = mjx_data_t_plus_1.qpos[self.joint_indices]

        # Calculate violations of soft limits
        lower_violations = -jnp.clip(joint_positions - self.soft_lower_limits, None, 0.0)
        upper_violations = jnp.clip(joint_positions - self.soft_upper_limits, 0.0, None)

        # Combine violations
        total_violations = lower_violations + upper_violations

        return jnp.sum(total_violations)


@attrs.define(frozen=True, kw_only=True)
class JointPosLimitPenaltyBuilder(RewardBuilder[JointPosLimitPenalty]):
    scale: float
    joint_limits: dict[str, tuple[float, float]]

    def __call__(self, data: BuilderData) -> JointPosLimitPenalty:
        # Convert joint names to indices
        joint_indices = []
        soft_lowers = []
        soft_uppers = []

        for joint_name, (lower, upper) in self.joint_limits.items():
            if joint_name in data.mujoco_mappings.qpos_name_to_idx_range:
                idx_range = data.mujoco_mappings.qpos_name_to_idx_range[joint_name]
                start_idx = idx_range[0]
                joint_indices.append(start_idx)
                soft_lowers.append(lower)
                soft_uppers.append(upper)
            else:
                raise ValueError(f"Joint '{joint_name}' not found in model")

        if not joint_indices:
            raise ValueError("No valid joints specified for JointPosLimitPenalty")

        return JointPosLimitPenalty(
            scale=self.scale,
            joint_indices=jnp.array(joint_indices),
            soft_lower_limits=jnp.array(soft_lowers),
            soft_upper_limits=jnp.array(soft_uppers),
        )


@attrs.define(frozen=True, kw_only=True)
class SinusoidalFeetHeightReward(Reward):
    """Reward for matching the feet to a sinusoidal pattern."""

    left_foot_geom_idx: int
    right_foot_geom_idx: int
    sinusoidal_feet_height: Callable[[Array], Array]
    vertical_offset: float
    norm: xax.NormType = attrs.field(default="l1")
    sensitivity: float = attrs.field(default=2.0)
    penalty_scale: float = attrs.field(default=0.2)
    error_clamp: float = attrs.field(default=0.5)

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        # Get the height of the feet
        left_foot_height = mjx_data_t_plus_1.geom_xpos[self.left_foot_geom_idx][2]
        right_foot_height = mjx_data_t_plus_1.geom_xpos[self.right_foot_geom_idx][2]
        # Calculate the sinusoidal pattern
        sin_pos = self.sinusoidal_feet_height(mjx_data_t_plus_1.time)

        sin_pos_left_mask = jnp.maximum(sin_pos, 0.0)
        sin_pos_right_mask = jnp.maximum(-sin_pos, 0.0)

        # Compute targets for left and right feet
        left_foot_target = jnp.sum(sin_pos_left_mask) + self.vertical_offset
        right_foot_target = jnp.sum(sin_pos_right_mask) + self.vertical_offset

        # Compute error
        left_foot_error = xax.get_norm(left_foot_height - left_foot_target, self.norm)
        right_foot_error = xax.get_norm(right_foot_height - right_foot_target, self.norm)

        # Total error
        total_error = left_foot_error + right_foot_error

        # Calculate reward using both exponential reward and linear penalty components
        # Mujoco playground uses just exponential reward, but isaac gym uses this setup
        exp_reward = jnp.exp(-self.sensitivity * total_error)
        linear_penalty = self.penalty_scale * jnp.minimum(total_error, self.error_clamp)

        reward = exp_reward - linear_penalty

        return reward


@attrs.define(frozen=True, kw_only=True)
class SinusoidalFeetHeightRewardBuilder(RewardBuilder[SinusoidalFeetHeightReward]):
    left_foot_geom_name: str
    right_foot_geom_name: str
    amplitude: float  # radians
    vertical_offset: float = attrs.field(default=0.0)
    period: float  # seconds
    scale: float
    sensitivity: float = attrs.field(default=2.0)
    penalty_scale: float = attrs.field(default=0.2)
    error_clamp: float = attrs.field(default=0.5)

    def __call__(self, data: BuilderData) -> SinusoidalFeetHeightReward:
        try:
            left_foot_geom_idx = data.mujoco_mappings.geom_name_to_idx[self.left_foot_geom_name]
            right_foot_geom_idx = data.mujoco_mappings.geom_name_to_idx[self.right_foot_geom_name]
        except KeyError:
            raise ValueError(
                f"Foot geom '{self.left_foot_geom_name}' or "
                f"'{self.right_foot_geom_name}' not found in model. "
                f"Available geoms: {data.mujoco_mappings.geom_name_to_idx.keys()}"
            )

        def sinusoidal_feet_height(time: Array) -> Array:
            return self.amplitude * jnp.sin(2 * jnp.pi * time / self.period)

        return SinusoidalFeetHeightReward(
            scale=self.scale,
            left_foot_geom_idx=left_foot_geom_idx,
            right_foot_geom_idx=right_foot_geom_idx,
            sinusoidal_feet_height=sinusoidal_feet_height,
            vertical_offset=self.vertical_offset,
            sensitivity=self.sensitivity,
            penalty_scale=self.penalty_scale,
            error_clamp=self.error_clamp,
        )


@attrs.define(frozen=True, kw_only=True)
class DHForwardReward(Reward):
    """Legacy default humanoid forward reward that linearly scales velocity."""

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        # Take just the x velocity component
        x_delta = -mjx_data_t_plus_1.qvel[1]
        return x_delta


@attrs.define(frozen=True, kw_only=True)
class XPosReward(Reward):
    """Reward for how far the robot has moved in the x direction."""

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        x_pos = mjx_data_t_plus_1.qpos[0]
        return x_pos


@attrs.define(frozen=True, kw_only=True)
class DHHealthyReward(Reward):
    """Legacy default humanoid healthy reward that gives binary reward based on height."""

    healthy_z_lower: float = attrs.field(default=0.5)
    healthy_z_upper: float = attrs.field(default=1.5)

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        return jnp.array(1.0)


@attrs.define(frozen=True, kw_only=True)
class DHTerminationPenalty(Reward):
    """Penalty for terminating the episode."""

    healthy_z_lower: float = attrs.field(default=0.5)
    healthy_z_upper: float = attrs.field(default=1.5)

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        height = mjx_data_t.qpos[2]
        is_unhealthy = jnp.where(height < self.healthy_z_lower, 1.0, 0.0)
        is_unhealthy = jnp.where(height > self.healthy_z_upper, 1.0, is_unhealthy)
        return is_unhealthy


@attrs.define(frozen=True, kw_only=True)
class DHControlPenalty(Reward):
    """Legacy default humanoid control cost that penalizes squared action magnitude."""

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
        done_t: Array,
    ) -> Array:
        return jnp.sum(jnp.square(action_t))
