"""Walking default humanoid task with amp."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import attrs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xax
from jaxtyping import Array, PRNGKeyArray

import ksim
from ksim.examples.default_humanoid.walking import (
    AuxOutputs,
    DefaultHumanoidActor,
    DefaultHumanoidCritic,
    HumanoidWalkingTask,
    HumanoidWalkingTaskConfig,
)


class NaiveVelocityReward(ksim.Reward):
    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        return trajectory.qvel[..., 0].clip(max=5.0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AmpAuxOutputs(AuxOutputs):
    tracked_pos: xax.FrozenDict[int, Array]


@attrs.define(frozen=True, kw_only=True)
class AmpReward(ksim.Reward):
    discriminator: AmpDiscriminatorMLP
    amp_reward_coef: float

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        # Concatenate state observations
        state = trajectory.obs
        next_state = trajectory.next_obs
        # d = self.discriminator(state, next_state)
        # this will live in the update step
        reward = self.amp_reward_coef * jnp.clip(1 - (1 / 4) * jnp.square(d - 1), min=0)
        return reward


@dataclass
class HumanoidWalkingAmpTaskConfig(HumanoidWalkingTaskConfig):
    reference_base_name: str = xax.field(
        value="CC_Base_Pelvis",
        help="The BVH joint name of the base of the humanoid",
    )
    visualize_reference_gait: bool = xax.field(
        value=False,
        help="Whether to visualize the reference gait.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingAmpTaskConfig)


class DefaultHumanoidDiscriminator(eqx.Module):
    """Discriminator for the walking task."""

    mlp: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_inputs = NUM_INPUTS
        num_outputs = 1

        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=num_outputs,
            width_size=hidden_size,
            depth=depth,
            key=key,
            activation=jax.nn.relu,
        )

    def forward(
        self,
        state: Array,
        expert_state: Array,
    ) -> Array:
        x_n = jnp.concatenate(
            [state, expert_state],
            axis=-1,
        )
        return self.mlp(x_n)


class DefaultHumanoidModel(eqx.Module):
    actor: DefaultHumanoidActor
    critic: DefaultHumanoidCritic
    discriminator: DefaultHumanoidDiscriminator

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
        num_mixtures: int,
    ) -> None:
        self.actor = DefaultHumanoidActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=0.5,
            hidden_size=hidden_size,
            depth=depth,
            num_mixtures=num_mixtures,
        )
        self.critic = DefaultHumanoidCritic(
            key,
            hidden_size=hidden_size,
            depth=depth,
        )
        self.discriminator = DefaultHumanoidDiscriminator(
            key,
            hidden_size=hidden_size,
            depth=depth,
        )


class HumanoidWalkingAmpTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        rewards = [
            ksim.BaseHeightRangeReward(z_lower=0.8, z_upper=1.5, dropoff=10.0, scale=0.5),
            ksim.LinearVelocityPenalty(index="z", scale=-0.01),
            ksim.AngularVelocityPenalty(index="x", scale=-0.01),
            ksim.AngularVelocityPenalty(index="y", scale=-0.01),
            NaiveVelocityReward(scale=0.1),
            AmpReward(reference_gait=self.reference_gait, ctrl_dt=self.config.ctrl_dt, scale=0.1),
        ]

        return rewards

    def get_on_policy_variables(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> ksim.PPOVariables:
        # Use cached log probabilities from training.
        if not isinstance(trajectories.aux_outputs, AuxOutputs):
            raise ValueError("No aux outputs found in trajectories")

        # Compute the values online.
        par_critic_fn = jax.vmap(self._run_critic, in_axes=(None, 0, 0))
        values_t1 = par_critic_fn(model.critic, trajectories.obs, trajectories.command)

        par_discriminator_fn = jax.vmap(self._run_discriminator, in_axes=(None, 0, 0))
        discriminator_t1 = par_discriminator_fn(model.discriminator, trajectories.obs, trajectories.command)

        return ksim.PPOVariables(
            log_probs_tn=trajectories.aux_outputs.log_probs,
            values_t=values_t1.squeeze(-1),
            discriminator_t=discriminator_t1.squeeze(-1),
        )


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.default_humanoid.walking
    # To visualize the environment, use the following command:
    #   python -m examples.default_humanoid.walking run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m examples.default_humanoid.walking num_envs=8 num_batches=2
    HumanoidWalkingAmpTask.launch(
        HumanoidWalkingAmpTaskConfig(
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            valid_every_n_steps=10,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=5.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            # Gait matching parameters.
            bvh_path=str(Path(__file__).parent / "data" / "walk-relaxed_actorcore.bvh"),
            rotate_bvh_euler=(0, np.pi / 2, 0),
            bvh_scaling_factor=1 / 100,
            mj_base_name="pelvis",
            reference_base_name="CC_Base_Pelvis",
        ),
    )
