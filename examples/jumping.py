# mypy: disable-error-code="override"
"""Defines simple task for training a jumping policy."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import attrs
import jax.numpy as jnp
from jaxtyping import Array

import ksim

from .walking_rnn import HumanoidWalkingRNNTask, HumanoidWalkingRNNTaskConfig


@attrs.define(frozen=True, kw_only=True)
class UpwardReward(ksim.Reward):
    """Incentives forward movement."""

    velocity_clip: float = attrs.field(default=10.0)

    def get_reward(self, trajectory: ksim.Trajectory) -> Array:
        # Just try to maximize the velocity in the Z direction.
        z_delta = jnp.clip(trajectory.qvel[..., 2], 0, self.velocity_clip)
        return z_delta


@dataclass
class HumanoidJumpingTaskConfig(HumanoidWalkingRNNTaskConfig):
    pass


Config = TypeVar("Config", bound=HumanoidJumpingTaskConfig)


class HumanoidJumpingTask(HumanoidWalkingRNNTask[Config], Generic[Config]):
    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            UpwardReward(scale=0.5),
            ksim.LinearVelocityPenalty(index=("z",), scale=-0.01),
            ksim.XYAngularVelocityPenalty(scale=-0.01),
        ]


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.jumping
    # To visualize the environment, use the following command:
    #   python -m examples.jumping run_mode=view
    HumanoidJumpingTask.launch(
        HumanoidJumpingTaskConfig(
            num_envs=2048,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=10.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            iterations=3,
            ls_iterations=5,
            max_action_latency=0.01,
        ),
    )
