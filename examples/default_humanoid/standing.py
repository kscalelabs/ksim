# mypy: disable-error-code="override"
"""Defines simple task for training a jumping policy for the default humanoid using an GRU actor."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import attrs
import xax
from jaxtyping import Array

import ksim

from .walking import HumanoidWalkingTask, HumanoidWalkingTaskConfig


@attrs.define(frozen=True, kw_only=True)
class StationaryPenalty(ksim.Reward):
    """Incentives staying in place laterally."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return xax.get_norm(trajectory.qvel[..., :2], self.norm).sum(axis=-1)


@dataclass
class HumanoidJumpingTaskConfig(HumanoidWalkingTaskConfig):
    pass


Config = TypeVar("Config", bound=HumanoidJumpingTaskConfig)


class HumanoidJumpingTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            StationaryPenalty(scale=-0.1),
            ksim.BaseHeightRangePenalty(z_lower=1.1, z_upper=1.5, scale=-1.0),
            ksim.ActuatorForcePenalty(scale=-0.01),
            ksim.LinearVelocityZPenalty(scale=-0.01),
            ksim.AngularVelocityXYPenalty(scale=-0.01),
        ]


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.default_humanoid.standing
    # To visualize the environment, use the following command:
    #   python -m examples.default_humanoid.standing run_environment=True
    HumanoidJumpingTask.launch(
        HumanoidJumpingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=10,
            rollout_length_seconds=10.0,
            # Logging parameters.
            log_full_trajectory_every_n_seconds=60,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
        ),
    )
