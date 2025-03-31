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
            ksim.StationaryPenalty(ctrl_dt=self.config.ctrl_dt, scale=-0.1),
            ksim.BaseHeightRangePenalty(z_lower=1.3, z_upper=1.5, scale=-1.0),
            ksim.TerminationPenalty(scale=-1.0),
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
            num_envs=256,
            batch_size=32,
            num_passes=4,
            epochs_per_log_step=10,
            rollout_length_seconds=3.0,
            # Logging parameters.
            # log_full_trajectory_every_n_seconds=60,
            # Simulation parameters.
            dt=0.0025,
            ctrl_dt=0.01,
            max_action_latency=0.0,
            min_action_latency=0.0,
        ),
    )
