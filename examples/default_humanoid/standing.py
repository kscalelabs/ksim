# mypy: disable-error-code="override"
"""Defines simple task for training a jumping policy for the default humanoid using an GRU actor."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import ksim
import xax

from .walking import HumanoidWalkingTask, HumanoidWalkingTaskConfig


@dataclass
class HumanoidStandingTaskConfig(HumanoidWalkingTaskConfig):
    step_phase: float = xax.field(value=0.25)


Config = TypeVar("Config", bound=HumanoidStandingTaskConfig)


class HumanoidStandingTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.BaseHeightRangeReward(z_lower=1.1, z_upper=1.5, scale=1.0),
            ksim.StayAliveReward(scale=1.0),
            ksim.FeetNoContactReward(window_size=round(self.config.step_phase / self.config.ctrl_dt), scale=0.1),
        ]


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
            epochs_per_log_step=1,
            rollout_length_seconds=4.0,
            # Logging parameters.
            # log_full_trajectory_every_n_seconds=60,
            log_full_trajectory_on_first_step=True,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            # PPO parameters.
            entropy_coef=0.001,
            value_loss_coef=0.5,
            gamma=0.97,
            lam=0.98,
        ),
    )
