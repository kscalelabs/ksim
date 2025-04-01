# mypy: disable-error-code="override"
"""Defines simple task for training a jumping policy for the default humanoid using an GRU actor."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import ksim

from .walking import HumanoidWalkingTask, HumanoidWalkingTaskConfig


@dataclass
class HumanoidStandingTaskConfig(HumanoidWalkingTaskConfig):
    pass


Config = TypeVar("Config", bound=HumanoidStandingTaskConfig)


class HumanoidStandingTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.BaseHeightRangeReward(z_lower=1.1, z_upper=1.5, scale=1.0),
            ksim.StayAliveReward(scale=1.0),
            ksim.ActionSmoothnessPenalty(scale=-0.001),
        ]

    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.Randomization]:
        return []  # Turn off randomization.


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.default_humanoid.standing
    # To visualize the environment, use the following command:
    #   python -m examples.default_humanoid.standing run_environment=True
    HumanoidStandingTask.launch(
        HumanoidStandingTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=32,
            epochs_per_log_step=10,
            rollout_length_seconds=10.0,
            # Logging parameters.
            # log_full_trajectory_every_n_seconds=60,
            log_full_trajectory_on_first_step=True,
            # Simulation parameters.
            dt=0.0025,
            ctrl_dt=0.01,
            max_action_latency=0.0,
            min_action_latency=0.0,
        ),
    )
