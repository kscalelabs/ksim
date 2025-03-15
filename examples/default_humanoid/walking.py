"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass
from typing import Collection

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx

from ksim.actuators import MITPositionActuators
from ksim.commands import Command, LinearVelocityCommand
from ksim.env.data import PhysicsModel
from ksim.env.mjx_engine import MjxEngine
from ksim.model.base import ActorCriticAgent, KSimModule
from ksim.model.distributions import TanhGaussianDistribution
from ksim.model.types import ModelCarry
from ksim.normalization import Normalizer, PassThrough, Standardize
from ksim.observation import (
    ActuatorForceObservation,
    CenterOfMassInertiaObservation,
    CenterOfMassVelocityObservation,
    LegacyPositionObservation,
    LegacyVelocityObservation,
    Observation,
)
from ksim.resets import RandomizeJointPositions, RandomizeJointVelocities
from ksim.rewards import DHForwardReward, HeightReward, Reward
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.terminations import Termination, UnhealthyTermination

NUM_OUTPUTS = 21


class DefaultHumanoidActor(eqx.Module, KSimModule):
    """Actor for the walking task."""

    mlp: eqx.nn.MLP
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=338,  # TODO: use similar pattern when dummy data gets passed in to populate
            out_size=NUM_OUTPUTS * 2,
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, obs: FrozenDict[str, Array], command: FrozenDict[str, Array], carry: ModelCarry | None) -> Array:
        obs_vec = jnp.concatenate([v for v in obs.values()], axis=-1)
        command_vec = jnp.concatenate([v for v in command.values()], axis=-1)
        x = jnp.concatenate([obs_vec, command_vec], axis=-1)
        prediction = self.mlp(x)

        mean = prediction[..., :NUM_OUTPUTS]
        std = prediction[..., NUM_OUTPUTS:]

        # softplus and clipping for stability
        std = (jax.nn.softplus(std) + self.min_std) * self.var_scale
        std = jnp.clip(std, self.min_std, self.max_std)

        # concat because Gaussian-like distributions expect the parameters
        # to be mean concat std
        parametrization = jnp.concatenate([mean, std], axis=-1)

        return parametrization

    # TODO: we should move all this to RL and away from the model definition
    def initial_carry(self) -> ModelCarry:
        """No carry for now, but we could use this to initialize recurrence or action history."""
        return None

    def forward_accross_episode(self, obs: FrozenDict[str, Array], command: FrozenDict[str, Array]) -> Array:
        """Forward pass across the episode (time, ...). No env dimension.

        By default, we vmap the forward pass for efficiency. If you implement
        recurrence, you should override this with an appropriate scan.
        """
        vmapped_forward = jax.vmap(self.forward, in_axes=(0, 0, None))
        prediction, _ = vmapped_forward(obs, command, None)
        return prediction


class DefaultHumanoidCritic(eqx.Module, KSimModule):
    """Critic for the walking task."""

    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=338,  # TODO: is there a nice way of inferring this?
            out_size=1,
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )

    def forward(self, obs: FrozenDict[str, Array], command: FrozenDict[str, Array], carry: ModelCarry | None) -> Array:
        obs_vec = jnp.concatenate([v for v in obs.values()], axis=-1)
        command_vec = jnp.concatenate([v for v in command.values()], axis=-1)
        x = jnp.concatenate([obs_vec, command_vec], axis=-1)
        return self.mlp(x)

    def initial_carry(self) -> ModelCarry:
        """No carry for now, but we could use this to initialize recurrence."""
        return None

    def forward_accross_episode(self, obs: FrozenDict[str, Array], command: FrozenDict[str, Array]) -> Array:
        """Forward pass across the episode (time, ...). No env dimension.

        By default, we vmap the forward pass for efficiency. If you implement
        recurrence, you should override this with an appropriate scan.
        """
        vmapped_forward = jax.vmap(self.forward, in_axes=(0, 0, None))
        prediction, _ = vmapped_forward(obs, command, None)
        return prediction


class HumanoidWalkingTask(PPOTask):

    def get_model_and_metadata(self) -> tuple[PhysicsModel, dict[str, JointMetadataOutput]]:
        mj_model = mujoco.MjModel.from_xml_path("examples/default_humanoid/scene.mjcf")
        metadata = get_joint_metadata(mj_model)  # TODO: implement this function properly
        mjx_model = mjx.Model(mj_model)
        return mjx_model, metadata

    def get_engine(self, physics_model: PhysicsModel, metadata: dict[str, JointMetadataOutput]) -> MjxEngine:
        return MjxEngine(
            default_physics_model=physics_model,
            resetters=[
                RandomizeJointPositions(scale=0.01),
                RandomizeJointVelocities(scale=0.01),
            ],
            actuators=MITPositionActuators(physics_model, metadata),
            # TODO: add randomizers
            dt=self.config.dt,
            ctrl_dt=self.config.ctrl_dt,
            solver=mjx.SolverType.CG,  # TODO: experiment with euler
            iterations=6,
            ls_iterations=6,
            disableflags=mjx.DisableBit.EULERDAMP,
            min_action_latency_step=0,
            max_action_latency_step=0,
        )

    def get_model(self, key: PRNGKeyArray) -> ActorCriticAgent:
        return ActorCriticAgent(
            critic_model=DefaultHumanoidCritic(key),
            actor_model=DefaultHumanoidActor(key, min_std=0.01, max_std=1.0, var_scale=1.0),
            action_distribution=TanhGaussianDistribution(action_dim=NUM_OUTPUTS),
        )

    def get_obs_normalizer(self, dummy_obs: FrozenDict[str, Array]) -> Normalizer:
        return Standardize(dummy_obs, alpha=1.0)

    def get_cmd_normalizer(self, dummy_cmd: FrozenDict[str, Array]) -> Normalizer:
        return PassThrough()

    # from ML: I haven't made up my mind on this API, but I generally think we should move away
    # from the hidden builder pattern. Giving the data directly will help with this.
    # In fact, we might even want to make this return a pure function.
    def get_obs_generators(self, physics_model: PhysicsModel) -> Collection[Observation]:
        return [
            LegacyPositionObservation(exclude_xy=True),
            LegacyVelocityObservation(),
            CenterOfMassInertiaObservation(),
            CenterOfMassVelocityObservation(),
            ActuatorForceObservation(),
        ]

    def get_command_generators(self) -> Collection[Command]:
        return [LinearVelocityCommand(x_scale=0.0, y_scale=0.0, switch_prob=0.02, zero_prob=0.3)]

    def get_reward_generators(self, physics_model: PhysicsModel) -> Collection[Reward]:
        return [
            HeightReward(scale=1.0, height_target=0.7),
            DHForwardReward(scale=0.2),
        ]

    def get_termination_generators(self, physics_model: PhysicsModel) -> Collection[Termination]:
        return [UnhealthyTermination(unhealthy_z_lower=0.8, unhealthy_z_upper=2.0)]


if __name__ == "__main__":
    # python -m examples.default_humanoid.walking
    HumanoidWalkingTask.launch(
        PPOConfig(
            num_learning_epochs=8,
            num_env_states_per_minibatch=8192,
            num_minibatches=32,
            num_envs=2048,
            dt=0.005,
            ctrl_dt=0.02,
            learning_rate=1e-5,
            save_every_n_steps=25,
            only_save_most_recent=False,
            reward_scaling_alpha=0.0,
            obs_norm_alpha=0.0,
            # ksim-legacy original setup was dt=0.003 and ctrl_dt=0.012 ~ 83.33 hz
            scale_rewards=False,
            gamma=0.97,
            lam=0.95,
            normalize_advantage=True,
            normalize_advantage_in_minibatch=True,
            entropy_coef=0.001,
            clip_param=0.3,
            use_clipped_value_loss=False,
            max_grad_norm=1.0,
            max_action_latency=0.0,
            min_action_latency=0.0,
            eval_rollout_length=1000,
        ),
    )
