# mypy: disable-error-code="override"
"""Defines simple task for training a jumping policy for the default humanoid using an GRU actor."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array

import ksim

from .walking import HumanoidWalkingTask
from .walking_gru import HumanoidWalkingGRUTaskConfig


@attrs.define(frozen=True, kw_only=True)
class UpwardReward(ksim.Reward):
    """Incentives forward movement."""

    velocity_clip: float = attrs.field(default=10.0)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        # Just try to maximize the velocity in the Z direction.
        z_delta = jnp.clip(trajectory.qvel[..., 2], 0, self.velocity_clip)
        return z_delta


@attrs.define(frozen=True, kw_only=True)
class StationaryPenalty(ksim.Reward):
    """Incentives staying in place laterally."""

    norm: xax.NormType = attrs.field(default="l2")

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return xax.get_norm(trajectory.qvel[..., :2], self.norm).sum(axis=-1)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


@dataclass
class HumanoidJumpingGRUTaskConfig(HumanoidWalkingGRUTaskConfig):
    pass


Config = TypeVar("Config", bound=HumanoidJumpingGRUTaskConfig)


class HumanoidJumpingGRUTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            UpwardReward(scale=0.5),
            StationaryPenalty(scale=-0.1),
            ksim.ActuatorForcePenalty(scale=-0.01),
            ksim.LinearVelocityZPenalty(scale=-0.01),
            ksim.AngularVelocityXYPenalty(scale=-0.01),
        ]


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.default_humanoid.walking_gru
    # To visualize the environment, use the following command:
    #   python -m examples.default_humanoid.walking_gru run_environment=True
    HumanoidJumpingGRUTask.launch(
        HumanoidJumpingGRUTaskConfig(
            num_envs=2048,
            batch_size=256,
            num_passes=8,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=10.0,  # This needs to be shorter because of memory constraints.
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
