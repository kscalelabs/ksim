# mypy: disable-error-code="override"
"""Example walking task using Adversarial Motion Priors."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, TypeVar

import equinox as eqx
import jax
import numpy as np
import xax
from jaxtyping import Array, PRNGKeyArray

import ksim
from ksim.utils.priors import ReferenceMapping

from .walking import (
    NUM_JOINTS,
    HumanoidWalkingTask,
    HumanoidWalkingTaskConfig,
)


class DefaultHumanoidDiscriminator(eqx.Module):
    """Discriminator for the walking task, returns logit."""

    layers: list[Callable[[Array], Array]]

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
        num_frames: int,
    ) -> None:
        num_inputs = NUM_JOINTS

        layers: list[Callable[[Array], Array]] = []
        for _ in range(depth):
            key, subkey = jax.random.split(key)
            layers += [
                eqx.nn.Conv1d(
                    in_channels=num_inputs,
                    out_channels=hidden_size,
                    kernel_size=num_frames,
                    # We left-pad the input so that the discriminator output
                    # at time T can be reasonably interpretted as the logit
                    # for the motion from time T - N to T. In other words, we
                    # this means that all the reward for time T will be based
                    # on the motion from the previous N frames.
                    padding=(num_frames - 1, 0),
                    key=subkey,
                ),
                jax.nn.relu,
            ]
            num_inputs = hidden_size

        layers += [
            eqx.nn.Conv1d(
                in_channels=num_inputs,
                out_channels=1,
                kernel_size=1,
                key=key,
            )
        ]

        self.layers = layers

    def forward(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class HumanoidWalkingAMPTaskConfig(HumanoidWalkingTaskConfig, ksim.AMPConfig):
    bvh_path: str = xax.field(
        value=str(Path(__file__).parent / "data" / "walk-relaxed_actorcore.bvh"),
        help="The path to the BVH file.",
    )
    rotate_bvh_euler: tuple[float, float, float] = xax.field(
        value=(0, 0, 0),
        help="Optional rotation to ensure the BVH tree matches the Mujoco model.",
    )
    bvh_scaling_factor: float = xax.field(
        value=1.0,
        help="Scaling factor to ensure the BVH tree matches the Mujoco model.",
    )
    bvh_offset: tuple[float, float, float] = xax.field(
        value=(0.0, 0.0, 0.0),
        help="Offset to ensure the BVH tree matches the Mujoco model.",
    )
    mj_base_name: str = xax.field(
        value="pelvis",
        help="The Mujoco body name of the base of the humanoid",
    )
    constrained_joint_ids: tuple[int, ...] = xax.field(
        value=(0, 1, 2, 3, 4, 5, 6),
        help="The indices of the joints to constrain. By default, freejoints.",
    )
    reference_base_name: str = xax.field(
        value="CC_Base_Pelvis",
        help="The BVH joint name of the base of the humanoid",
    )
    visualize_reference_points: bool = xax.field(
        value=False,
        help="Whether to visualize the reference points.",
    )
    visualize_reference_motion: bool = xax.field(
        value=False,
        help="Whether to visualize the reference motion after running IK.",
    )
    discriminator_num_frames: int = xax.field(
        value=10,
        help="The number of frames to use for the discriminator.",
    )
    num_discriminator_passes: int = xax.field(
        value=1,
        help="The number of times to pass the discriminator.",
    )
    w_gp: float = xax.field(
        value=10.0,
        help="Gradient penalty coefficient for discriminator (WGAN-GP style).",
    )


HUMANOID_REFERENCE_MAPPINGS = (
    ReferenceMapping("CC_Base_L_ThighTwist01", "thigh_left"),  # hip
    ReferenceMapping("CC_Base_L_CalfTwist01", "shin_left"),  # knee
    ReferenceMapping("CC_Base_L_Foot", "foot_left"),  # foot
    ReferenceMapping("CC_Base_L_UpperarmTwist01", "upper_arm_left"),  # shoulder
    ReferenceMapping("CC_Base_L_ForearmTwist01", "lower_arm_left"),  # elbow
    ReferenceMapping("CC_Base_L_Hand", "hand_left"),  # hand
    ReferenceMapping("CC_Base_R_ThighTwist01", "thigh_right"),  # hip
    ReferenceMapping("CC_Base_R_CalfTwist01", "shin_right"),  # knee
    ReferenceMapping("CC_Base_R_Foot", "foot_right"),  # foot
    ReferenceMapping("CC_Base_R_UpperarmTwist01", "upper_arm_right"),  # shoulder
    ReferenceMapping("CC_Base_R_ForearmTwist01", "lower_arm_right"),  # elbow
    ReferenceMapping("CC_Base_R_Hand", "hand_right"),  # hand
)


Config = TypeVar("Config", bound=HumanoidWalkingAMPTaskConfig)


class HumanoidWalkingAMPTask(HumanoidWalkingTask[Config], ksim.AMPTask[Config], Generic[Config]):
    """Adversarial Motion Prior task."""


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.walking_reference_motion
    # To visualize the environment, use the following command:
    #   python -m examples.walking_reference_motion run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m examples.walking_reference_motion num_envs=8 num_batches=2
    config = HumanoidWalkingAMPTaskConfig()
    task = HumanoidWalkingAMPTask(config)

    HumanoidWalkingAMPTask.launch(
        HumanoidWalkingAMPTaskConfig(
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            valid_every_n_steps=10,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            rollout_length_seconds=5.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            # Gait matching parameters.
            bvh_path=str(Path(__file__).parent / "data" / "walk_normal_dh.bvh"),
            rotate_bvh_euler=(0, np.pi / 2, 0),
            bvh_scaling_factor=1 / 100,
            mj_base_name="pelvis",
            reference_base_name="CC_Base_Pelvis",
        ),
    )
