# mypy: disable-error-code="override"
"""Defines simple task for training a walking policy for the default humanoid with delta position control."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from kscale.web.gen.api import JointMetadataOutput

import ksim

from .walking_rnn import (
    HumanoidWalkingRNNTask,
    HumanoidWalkingRNNTaskConfig,
)


@dataclass
class HumanoidWalkingDeltaTaskConfig(HumanoidWalkingRNNTaskConfig):
    pass


Config = TypeVar("Config", bound=HumanoidWalkingDeltaTaskConfig)


class HumanoidWalkingDeltaTask(HumanoidWalkingRNNTask[Config], Generic[Config]):
    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.MITDeltaPositionActuators(
            physics_model=physics_model,
            joint_name_to_metadata=metadata,
        )

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards = super().get_rewards(physics_model)
        rewards.append(ksim.FeetFlatReward(scale=1.0))
        return rewards


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.walking_delta
    # To visualize the environment, use the following command:
    #   python -m examples.walking_delta run_model_viewer=True
    HumanoidWalkingDeltaTask.launch(
        HumanoidWalkingDeltaTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=2,
            epochs_per_log_step=1,
            rollout_length_seconds=8.0,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=3,
            ls_iterations=5,
            max_action_latency=0.01,
        ),
    )
