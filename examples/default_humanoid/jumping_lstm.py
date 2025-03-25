"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Array

import ksim

from .walking import DHControlPenalty, DHHealthyReward, HumanoidWalkingTask
from .walking_lstm import HumanoidWalkingLSTMTask, HumanoidWalkingLSTMTaskConfig


@attrs.define(frozen=True, kw_only=True)
class UpwardReward(ksim.Reward):
    """Incentives forward movement."""

    velocity_clip: float = attrs.field(default=10.0)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        # Just try to maximize the velocity in the Z direction.
        z_delta = jnp.clip(trajectory.qvel[..., 2], 0, self.velocity_clip)
        return z_delta


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


@dataclass
class HumanoidJumpingLSTMTaskConfig(HumanoidWalkingLSTMTaskConfig):
    pass


class HumanoidJumpingLSTMTask(HumanoidWalkingTask[HumanoidJumpingLSTMTaskConfig]):

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            UpwardReward(scale=0.5),
            DHControlPenalty(scale=-0.01),
            DHHealthyReward(scale=0.5, healthy_z_upper=5.0),
        ]


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.default_humanoid.walking_lstm
    # To visualize the environment, use the following command:
    #   python -m examples.default_humanoid.walking_lstm run_environment=True
    HumanoidWalkingLSTMTask.launch(
        HumanoidWalkingLSTMTaskConfig(
            num_envs=2048,
            num_batches=64,
            num_passes=8,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=10.0,  # This needs to be shorter because of memory constraints.
            eval_rollout_length_seconds=4.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            use_mit_actuators=False,
        ),
    )
