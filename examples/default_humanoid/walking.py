"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass
from pathlib import Path

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx

from ksim.actuators import Actuators, MITPositionActuators, TorqueActuators
from ksim.commands import Command, LinearVelocityCommand
from ksim.env.data import PhysicsModel
from ksim.observation import ActuatorForceObservation, Observation
from ksim.randomization import (
    Randomization,
    WeightRandomization,
)
from ksim.resets import RandomJointPositionReset, RandomJointVelocityReset, Reset
from ksim.rewards import DHForwardReward, HeightReward, Reward
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.terminations import Termination, UnhealthyTermination
from ksim.utils.mujoco import get_joint_metadata

NUM_INPUTS = 29
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

    def __call__(
        self,
        act_frc_obs_n: Array,
        lin_vel_cmd_n: Array,
    ) -> distrax.Distribution:
        x_n = jnp.concatenate([act_frc_obs_n, lin_vel_cmd_n], axis=-1)  # (NUM_INPUTS)

        # Split the output into mean and standard deviation.
        prediction_n = self.mlp(x_n)
        mean_n = prediction_n[..., :NUM_OUTPUTS]
        std_n = prediction_n[..., NUM_OUTPUTS:]

        # Softplus and clip to ensure positive standard deviations.
        std_n = (jax.nn.softplus(std_n) + self.min_std) * self.var_scale
        std_n = jnp.clip(std_n, self.min_std, self.max_std)

        # return distrax.Transformed(distrax.Normal(mean_n, std_n), distrax.Tanh())
        return distrax.Normal(mean_n, std_n)


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

    def __call__(
        self,
        act_frc_obs_n: Array,
        lin_vel_cmd_n: Array,
    ) -> Array:
        x_n = jnp.concatenate([act_frc_obs_n, lin_vel_cmd_n], axis=-1)  # (NUM_INPUTS)
        return self.mlp(x_n)


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


@dataclass
class HumanoidWalkingTaskConfig(PPOConfig):
    """Config for the humanoid walking task."""

    use_mit_actuators: bool = xax.field(
        value=False,
        help="Whether to use the MIT actuator model, where the actions are position commands",
    )
    render_track_body_id: int | None = xax.field(
        value=0,
        help="The body id to track with the render camera.",
    )
    kp: float = xax.field(
        value=1.0,
        help="The Kp for the actuators",
    )
    kd: float = xax.field(
        value=0.1,
        help="The Kd for the actuators",
    )
    armature: float = xax.field(
        value=1e-2,
        help="A value representing the effective inertia of the actuator armature",
    )
    friction: float = xax.field(
        value=1e-6,
        help="The dynamic friction loss for the actuator",
    )


class HumanoidWalkingTask(PPOTask[HumanoidWalkingTaskConfig]):
    def get_mujoco_model(self) -> tuple[mujoco.MjModel, dict[str, JointMetadataOutput]]:
        mjcf_path = (Path(__file__).parent / "scene.mjcf").resolve().as_posix()
        mj_model = mujoco.MjModel.from_xml_path(mjcf_path)

        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG

        return mj_model

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> dict[str, JointMetadataOutput]:
        return get_joint_metadata(
            mj_model,
            kp=self.config.kp,
            kd=self.config.kd,
            armature=self.cfonfig.armature,
            friction=self.config.friction,
        )

    def get_actuators(self, physics_model: PhysicsModel, metadata: dict[str, JointMetadataOutput]) -> Actuators:
        if self.config.use_mit_actuators:
            return MITPositionActuators(physics_model, metadata)
        else:
            return TorqueActuators()

    def get_randomization(self, physics_model: PhysicsModel) -> list[Randomization]:
        return [
            WeightRandomization(scale=0.01),
        ]

    def get_resets(self, physics_model: PhysicsModel) -> list[Reset]:
        return [
            RandomJointPositionReset(scale=0.01),
            RandomJointVelocityReset(scale=0.01),
        ]

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

    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidModel:
        return DefaultHumanoidModel(key)

    def get_initial_carry(self) -> None:
        return None

    def sample_action(
        self,
        model: DefaultHumanoidModel,
        carry: None,
        physics_model: PhysicsModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, None]:
        act_frc_obs_n = observations["actuator_force_observation"]
        lin_vel_cmd_n = commands["linear_velocity_command"]
        action_dist_n = model.actor(act_frc_obs_n, lin_vel_cmd_n)
        action_n = action_dist_n.sample(seed=rng)
        return action_n, None


if __name__ == "__main__":
    # python -m examples.default_humanoid.walking run_environment=True
    HumanoidWalkingTask.launch(
        HumanoidWalkingTaskConfig(
            num_envs=8,
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
            rollout_length_seconds=20.0,
            eval_rollout_length_seconds=5.0,
        ),
    )
