"""Defines simple task for training a walking policy for K-Bot."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx

from ksim.actuators import TorqueActuators
from ksim.commands import Command, LinearVelocityCommand
from ksim.env.data import PhysicsModel
from ksim.env.mjx_engine import MjxEngine
from ksim.observation import ActuatorForceObservation, Observation
from ksim.resets import RandomizeJointPositions, RandomizeJointVelocities
from ksim.rewards import DHForwardReward, HeightReward, Reward
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.terminations import Termination, UnhealthyTermination
from ksim.utils.named_access import get_joint_metadata

NUM_INPUTS = 56
NUM_OUTPUTS = 21


class DefaultHumanoidActor(eqx.Module):
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
            in_size=NUM_INPUTS,
            out_size=NUM_OUTPUTS * 2,
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(
        self,
        obs: FrozenDict[str, Array],
        command: FrozenDict[str, Array],
        carry: None,
    ) -> tuple[Array, None]:
        obs_vec = jnp.concatenate([v for v in obs.values()], axis=-1)
        command_vec = jnp.concatenate([v for v in command.values()], axis=-1)
        assert obs_vec.ndim == command_vec.ndim
        x = jnp.concatenate([obs_vec, command_vec], axis=-1)

        # Split the output into mean and standard deviation.
        prediction = self.mlp(x)
        mean = prediction[..., :NUM_OUTPUTS]
        std = prediction[..., NUM_OUTPUTS:]

        # Softplus and clip to ensure positive standard deviations.
        std = (jax.nn.softplus(std) + self.min_std) * self.var_scale
        std = jnp.clip(std, self.min_std, self.max_std)

        # Concatenate the Gaussian parameters into a single array.
        parametrization = jnp.concatenate([mean, std], axis=-1)

        return parametrization, None

    def initial_carry(self) -> None:
        return None


class DefaultHumanoidCritic(eqx.Module):
    """Critic for the walking task."""

    mlp: eqx.nn.MLP

    def __init__(self, key: PRNGKeyArray) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=NUM_INPUTS,
            out_size=1,  # Always output a single critic value.
            width_size=64,
            depth=5,
            key=key,
            activation=jax.nn.relu,
        )

    def forward(
        self,
        obs: FrozenDict[str, Array],
        command: FrozenDict[str, Array],
        carry: None,
    ) -> tuple[Array, None]:
        obs_vec = jnp.concatenate([v for v in obs.values()], axis=-1)
        command_vec = jnp.concatenate([v for v in command.values()], axis=-1)
        x = jnp.concatenate([obs_vec, command_vec], axis=-1)
        return self.mlp(x), None

    def batched_forward_across_time(self, obs: FrozenDict[str, Array], command: FrozenDict[str, Array]) -> Array:
        """Forward pass across the episode (time, ...). No env dimension.

        By default, we vmap the forward pass for efficiency. If you implement
        recurrence, you should override this with an appropriate scan.
        """
        vmapped_forward = jax.vmap(self.forward, in_axes=(0, 0, None))
        prediction, _ = vmapped_forward(obs, command, None)
        return prediction


class DefaultHumanoidModel(eqx.Module):
    actor: DefaultHumanoidActor
    critic: DefaultHumanoidCritic

    def __init__(self, key: PRNGKeyArray) -> None:
        self.actor = DefaultHumanoidActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=1.0,
        )
        self.critic = DefaultHumanoidCritic(key)


class HumanoidWalkingTaskConfig(PPOConfig):
    """Config for the humanoid walking task."""

    pass


class HumanoidWalkingTask(PPOTask[HumanoidWalkingTaskConfig]):
    def get_mujoco_model_and_metadata(self) -> tuple[mujoco.MjModel, dict[str, JointMetadataOutput]]:
        mjcf_path = (Path(__file__).parent / "scene.mjcf").resolve().as_posix()
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
        metadata = get_joint_metadata(mj_model)

        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG

        return mj_model, metadata

    def get_engine(self, physics_model: PhysicsModel, metadata: dict[str, JointMetadataOutput]) -> MjxEngine:
        return MjxEngine(
            default_physics_model=physics_model,
            resetters=[
                RandomizeJointPositions(scale=0.01),
                RandomizeJointVelocities(scale=0.01),
            ],
            # actuators=MITPositionActuators(physics_model, metadata), # TODO:bring it back
            actuators=TorqueActuators(),
            # TODO: add randomizers
            dt=self.config.dt,
            ctrl_dt=self.config.ctrl_dt,
            min_action_latency_step=0,
            max_action_latency_step=0,
        )

    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidModel:
        return DefaultHumanoidModel(key)

    def get_observations(self, physics_model: PhysicsModel) -> list[Observation]:
        return [
            ActuatorForceObservation(),
        ]

    def get_commands(self, physics_model: PhysicsModel) -> list[Command]:
        return [
            LinearVelocityCommand(x_scale=0.0, y_scale=0.0, switch_prob=0.02, zero_prob=0.3),
        ]

    def get_rewards(self, physics_model: PhysicsModel) -> list[Reward]:
        return [
            HeightReward(scale=1.0, height_target=0.7),
            DHForwardReward(scale=0.2),
        ]

    def get_terminations(self, physics_model: PhysicsModel) -> list[Termination]:
        return [
            UnhealthyTermination(unhealthy_z_lower=0.8, unhealthy_z_upper=2.0),
        ]


if __name__ == "__main__":
    # python -m examples.default_humanoid.walking
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            compile_unroll=False,
            num_learning_epochs=8,
            num_env_states_per_minibatch=20,
            num_minibatches=5,
            num_envs=10,
            dt=0.005,
            ctrl_dt=0.02,
            learning_rate=1e-5,
            save_every_n_steps=25,
            only_save_most_recent=False,
            reward_scaling_alpha=0.0,
            obs_norm_alpha=0.0,
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
