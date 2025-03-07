"""Defines a base interface for defining reward functions."""

import functools
import logging
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

import attrs
import jax
import jax.numpy as jnp
import xax
from flax.core import FrozenDict
from jaxtyping import Array
from mujoco import mjx

from ksim.utils.data import BuilderData
from ksim.utils.mujoco import geoms_colliding
from ksim.utils.transforms import quat_to_euler

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
        accumulation function - for example, you might want to only
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
        return reward


# TODO: Check that this is correct
@attrs.define(frozen=True, kw_only=True)
class OrientationPenalty(Reward):
    """Penalty for how well the robot is oriented."""

    norm: NormType = attrs.field(default="l2")
    target_orientation: jax.Array = attrs.field(default=jnp.array([0.073, 0.0, 1.0]))

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        return get_norm(
            quat_to_euler(mjx_data_t_plus_1.qpos[3:7]) - self.target_orientation, self.norm
        )


@attrs.define(frozen=True, kw_only=True)
class TorquePenalty(Reward):
    """Penalty for high torques."""

    norm: NormType = attrs.field(default="l1")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        return get_norm(mjx_data_t_plus_1.actuator_force, self.norm)


@attrs.define(frozen=True, kw_only=True)
class EnergyPenalty(Reward):
    """Penalty for high energies."""

    norm: NormType = attrs.field(default="l1")

    # NOTE: I think this is actually penalizing power (?). Rename if needed
    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        return get_norm(mjx_data_t_plus_1.qvel[6:], self.norm) * get_norm(
            mjx_data_t_plus_1.actuator_force, self.norm
        )


@attrs.define(frozen=True, kw_only=True)
class JointAccelerationPenalty(Reward):
    """Penalty for high joint accelerations."""

    norm: NormType = attrs.field(default="l2")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        return get_norm(mjx_data_t_plus_1.qacc[6:], self.norm)


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
    sensitivity: float = attrs.field(default=1.0)

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
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

    norm: NormType = attrs.field(default="l2")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        # During tracing, both branches of jax.lax.cond are evaluated, so
        # we need to handle the case where action_t_minus_1 is None.
        # This only works if action_t_minus_1 is statically None or not None.
        if action_t_minus_1 is None:
            return jnp.zeros_like(get_norm(action_t, self.norm).sum(axis=-1))
        return get_norm(action_t - action_t_minus_1, self.norm).sum(axis=-1)


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
    ) -> Array:
        contacts = jnp.array(
            [
                geoms_colliding(mjx_data_t_plus_1, geom_idx, self.floor_idx)
                for geom_idx in self.foot_geom_idxs
            ]
        )

        # Get x and y velocities
        body_vel = mjx_data_t_plus_1.qvel[:2]

        return jnp.linalg.norm(body_vel, axis=-1) * contacts


@attrs.define(frozen=True, kw_only=True)
class FootSlipPenaltyBuilder(RewardBuilder[FootSlipPenalty]):
    scale: float
    foot_body_names: list[str]

    def __call__(self, data: BuilderData) -> FootSlipPenalty:
        illegal_geom_idxs = []
        for geom_idx, body_name in data.mujoco_mappings.geom_idx_to_body_name.items():
            if body_name in self.foot_body_names:
                illegal_geom_idxs.append(geom_idx)

        illegal_geom_idxs = jnp.array(illegal_geom_idxs)

        floor_idx = data.mujoco_mappings.floor_geom_idx

        if floor_idx is None:
            raise ValueError("No floor geom found in model")

        return FootSlipPenalty(
            scale=self.scale,
            foot_geom_idxs=illegal_geom_idxs,
            floor_idx=floor_idx,
        )


@attrs.define(frozen=True, kw_only=True)
class FeetClearancePenalty(Reward):
    """Penalty for deviation from desired feet clearance."""

    foot_geom_idxs: Array
    max_foot_height: float
    norm: NormType = attrs.field(default="l1")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        feet_heights = mjx_data_t_plus_1.geom_xpos[self.foot_geom_idxs][:, 2]

        # TODO: Look into adding linear feet velocity norm to scale the foot delta

        return get_norm(feet_heights - self.max_foot_height, self.norm)


@attrs.define(frozen=True, kw_only=True)
class FeetClearancePenaltyBuilder(RewardBuilder[FeetClearancePenalty]):
    scale: float
    foot_body_names: list[str]
    max_foot_height: float

    def __call__(self, data: BuilderData) -> FeetClearancePenalty:
        illegal_geom_idxs = []
        for geom_idx, body_name in data.mujoco_mappings.geom_idx_to_body_name.items():
            if body_name in self.foot_body_names:
                illegal_geom_idxs.append(geom_idx)

        illegal_geom_idxs = jnp.array(illegal_geom_idxs)

        return FeetClearancePenalty(
            scale=self.scale,
            foot_geom_idxs=illegal_geom_idxs,
            max_foot_height=self.max_foot_height,
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


@attrs.define(frozen=True, kw_only=True)
class FeetAirTimeReward(Reward):
    """Reward for keeping the robot's feet in the air.

    If the robot's feet are in the air for more than `required_air_time_prct`
    percent of the time, the reward will be applied proportionally.

    We additionally specify a list of commands which, if set to zero, will
    cause the reward to be ignored. This is to avoid rewarding air time
    if the robot is being commanded to stay still.
    """

    foot_geom_idxs: Array
    floor_idx: int
    required_air_time_prct: float
    skip_if_zero_command: list[str] = attrs.field(factory=list)
    eps: float = attrs.field(default=1e-6)

    def __post_init__(self) -> None:
        super().__post_init__()
        assert 0 <= self.required_air_time_prct <= 1
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
        # Check if each foot is in contact with the floor
        contacts = jnp.array(
            [
                geoms_colliding(mjx_data_t_plus_1, geom_idx, self.floor_idx)
                for geom_idx in self.foot_geom_idxs
            ]
        )

        # Count how many feet are off the ground (not in contact)
        feet_in_air = (~contacts).sum().astype(jnp.float32)

        # Skip the reward if specified commands are zero
        if self.skip_if_zero_command:
            commands_are_zero = jnp.stack(
                [(command_t[cmd] < self.eps).all() for cmd in self.skip_if_zero_command],
                axis=0,
            )
            feet_in_air = jnp.where(commands_are_zero.any(), 0.0, feet_in_air)

        return feet_in_air

    def post_accumulate(self, reward: Array) -> Array:
        # Calculate the average number of feet in the air across timesteps
        mean_feet_in_air = reward.mean()

        # Calculate the maximum possible feet in air (length of foot_geom_idxs)
        max_feet = len(self.foot_geom_idxs)

        # Calculate the minimum required average feet in air
        min_required_feet = self.required_air_time_prct * max_feet

        # Only apply reward when average feet in air exceeds the required amount,
        # and scale it by how much it exceeds the requirement
        excess_feet = (mean_feet_in_air - min_required_feet).clip(min=0)

        # Normalize by the maximum possible excess
        max_possible_excess = max_feet - min_required_feet
        normalized_multiplier = excess_feet / max_possible_excess.clip(min=1e-6)

        return reward * normalized_multiplier


@attrs.define(frozen=True, kw_only=True)
class FeetAirTimeRewardBuilder(RewardBuilder[FeetAirTimeReward]):
    scale: float
    foot_body_names: list[str]
    required_air_time_prct: float
    skip_if_zero_command: list[str] | None = attrs.field(default=None)

    def __call__(self, data: BuilderData) -> FeetAirTimeReward:
        foot_geom_idxs = []
        for geom_idx, body_name in data.mujoco_mappings.geom_idx_to_body_name.items():
            if body_name in self.foot_body_names:
                foot_geom_idxs.append(geom_idx)

        foot_geom_idxs = jnp.array(foot_geom_idxs)

        floor_idx = data.mujoco_mappings.floor_geom_idx
        if floor_idx is None:
            raise ValueError("No floor geom found in model")

        return FeetAirTimeReward(
            scale=self.scale,
            foot_geom_idxs=foot_geom_idxs,
            floor_idx=floor_idx,
            required_air_time_prct=self.required_air_time_prct,
            skip_if_zero_command=self.skip_if_zero_command if self.skip_if_zero_command else [],
        )


# TODO: Make sure this penalty behaves as expected
@attrs.define(frozen=True, kw_only=True)
class ContactForcePenalty(Reward):
    """Penalty for excessive contact forces on specific body parts.

    Penalizes contact forces that exceed a specified maximum threshold.
    This encourages smoother, more controlled movements without harsh impacts.
    """

    foot_geom_idxs: Array
    floor_idx: int
    max_contact_force: float
    norm: NormType = attrs.field(default="l1")

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        total_penalty = jnp.zeros_like(mjx_data_t_plus_1.contact.force)

        for foot_geom_idx in self.foot_geom_idxs:
            # If colliding, get the force information from contact data
            # Find all contacts involving this foot and the floor
            foot_floor_contacts = jnp.logical_or(
                jnp.logical_and(
                    mjx_data_t_plus_1.contact.geom1 == foot_geom_idx,
                    mjx_data_t_plus_1.contact.geom2 == self.floor_idx,
                ),
                jnp.logical_and(
                    mjx_data_t_plus_1.contact.geom1 == self.floor_idx,
                    mjx_data_t_plus_1.contact.geom2 == foot_geom_idx,
                ),
            )

            # Extract contact forces for matches
            # In MJX, contact forces are stored in a 3D array per contact point
            contact_forces = mjx_data_t_plus_1.contact.force

            # Calculate force magnitude (norm of the force vector)
            force_magnitudes = jnp.linalg.norm(contact_forces, axis=1)

            # Apply mask to get only forces for this foot-floor contact
            masked_forces = jnp.where(foot_floor_contacts, force_magnitudes, 0.0)

            # Find the maximum force for this foot-floor contact
            max_force = jnp.max(masked_forces)

            # Penalize only forces exceeding the threshold
            force_penalty = jnp.clip(max_force - self.max_contact_force, min=0.0)

            # Apply the norm and add to total penalty
            foot_penalty = get_norm(force_penalty, self.norm)
            total_penalty += foot_penalty

        return total_penalty


@attrs.define(frozen=True, kw_only=True)
class ContactForcePenaltyBuilder(RewardBuilder[ContactForcePenalty]):
    scale: float
    foot_body_names: list[str]
    max_contact_force: float
    norm: NormType = attrs.field(default="l1")

    def __call__(self, data: BuilderData) -> ContactForcePenalty:
        foot_geom_idxs = []
        for geom_idx, body_name in data.mujoco_mappings.geom_idx_to_body_name.items():
            if body_name in self.foot_body_names:
                foot_geom_idxs.append(geom_idx)

        foot_geom_idxs = jnp.array(foot_geom_idxs)

        floor_idx = data.mujoco_mappings.floor_geom_idx
        if floor_idx is None:
            raise ValueError("No floor geom found in model")

        return ContactForcePenalty(
            scale=self.scale,
            foot_geom_idxs=foot_geom_idxs,
            floor_idx=floor_idx,
            max_contact_force=self.max_contact_force,
            norm=self.norm,
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
    norm: NormType = attrs.field(default="l2")
    exclude_base_pose: bool = attrs.field(default=True)

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        # Get current joint positions
        current_positions = mjx_data_t_plus_1.qpos[self.joint_indices]

        # Calculate deviation from default pose
        deviations = current_positions - self.default_positions

        # Apply weights to deviations
        weighted_deviations = deviations * self.joint_deviation_weights

        return get_norm(weighted_deviations, self.norm)


@attrs.define(frozen=True, kw_only=True)
class DefaultPoseDeviationPenaltyBuilder(RewardBuilder[DefaultPoseDeviationPenalty]):
    scale: float
    default_positions: dict[str, float]
    deviation_weights: dict[str, float]
    norm: NormType = attrs.field(default="l2")

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
    ) -> Array:
        # Get current joint positions
        joint_positions = mjx_data_t_plus_1.qpos[self.joint_indices]

        # Calculate violations of soft limits
        lower_violations = -jnp.clip(joint_positions - self.soft_lower_limits, None, 0.0)
        upper_violations = jnp.clip(joint_positions - self.soft_upper_limits, 0.0, None)

        # Combine violations
        total_violations = lower_violations + upper_violations

        return total_violations


@attrs.define(frozen=True, kw_only=True)
class JointPosLimitPenaltyBuilder(RewardBuilder[JointPosLimitPenalty]):
    scale: float
    joint_limits: dict[str, tuple[float, float]]

    def __call__(self, data: BuilderData) -> JointPosLimitPenalty:
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
            norm=self.norm,
        )
