"""defines simple task for training a walking policy for zbot2."""

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

from ksim.builders.commands import AngularVelocityCommand, LinearVelocityCommand
from ksim.builders.observation import (
    ActuatorForceObservation,
    BaseAngularVelocityObservation,
    BaseLinearVelocityObservation,
    BaseOrientationObservation,
    CenterOfMassInertiaObservation,
    CenterOfMassVelocityObservation,
    JointPositionObservation,
    JointVelocityObservation,
    SensorObservationBuilder,
)
from ksim.builders.resets import XYPositionResetBuilder
from ksim.builders.rewards import (
    AngularVelocityXYPenalty,
    DefaultPoseDeviationPenaltyBuilder,
    DHForwardReward,
    DHHealthyReward,
    HeightReward,
    OrientationPenalty,
    TrackAngularVelocityZReward,
    TrackLinearVelocityXYReward,
)
from ksim.builders.terminations import PitchTooGreatTermination, RollTooGreatTermination
from ksim.env.mjx.mjx_env import MjxEnv, MjxEnvConfig
from ksim.model.distributions import TanhGaussianDistribution
from ksim.model.formulations import ActorCriticAgent, ActorModel
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask

######################
# Static Definitions #
######################

NUM_OUTPUTS = 18


@dataclass
class ZBot2WalkingConfig(PPOConfig, MjxEnvConfig):
    """Combining configs for the ZBot2 walking task and fixing params."""

    robot_model_name: str = "examples/zbot2/"


#####################
# Model Definitions #
#####################


class ZBot2LearnedStdActorModel(ActorModel):
    network: MLP
    min_std: float = 0.01
    max_std: float = 1.0
    var_scale: float = 1.0

    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        """Forward pass of the actor model."""
        x = jnp.concatenate(list(obs.values()), axis=-1)
        predictions = self.network(x)

        mean = predictions[..., :NUM_OUTPUTS]
        std = predictions[..., NUM_OUTPUTS:]

        # need to do this for stability
        std = (jax.nn.softplus(std) + self.min_std) * self.var_scale
        std = jnp.clip(std, self.min_std, self.max_std)

        # concat because Gaussian-like distributions expect the parameters
        # to be mean concat std
        actions_n = jnp.concatenate([mean, std], axis=-1)

        return actions_n


class ZBot2CriticModel(nn.Module):
    network: MLP

    @nn.compact
    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> jax.Array:
        """Forward pass of the critic model."""
        x = jnp.concatenate(list(obs.values()), axis=-1)
        value_estimate = self.network(x)

        return value_estimate


####################
# Task Definitions #
####################


class ZBot2WalkingTask(PPOTask[ZBot2WalkingConfig]):
    def get_environment(self) -> MjxEnv:
        """Get the environment."""
        return MjxEnv(
            self.config,
            terminations=[
                RollTooGreatTermination(max_roll=1.04),
                PitchTooGreatTermination(max_pitch=1.04),
            ],
            resets=[
                XYPositionResetBuilder(),
            ],
            rewards=[
                # AngularVelocityXYPenalty(scale=-0.15),
                # TrackLinearVelocityXYReward(scale=1.0),
                # HeightReward(scale=1.0, height_target=0.42),
                # TrackAngularVelocityZReward(scale=1.0),
                # OrientationPenalty(scale=-0.5, target_orientation=[0.0, 0.0, 0.0]),
                # DefaultPoseDeviationPenaltyBuilder(
                #     scale=-0.1,
                #     default_positions={
                #         "left_shoulder_pitch": 0.0,
                #         "left_shoulder_yaw": 0.0,
                #         "left_elbow": 0.0,
                #         "right_shoulder_pitch": 0.0,
                #         "right_shoulder_yaw": 0.0,
                #         "right_elbow": 0.0,
                #         "left_hip_pitch": 0.0,
                #         "left_hip_roll": 0.0,
                #         "left_hip_yaw": 0.0,
                #         "left_knee": 0.0,
                #         "left_ankle": 0.0,
                #         "right_hip_pitch": 0.0,
                #         "right_hip_roll": 0.0,
                #         "right_hip_yaw": 0.0,
                #         "right_knee": 0.0,
                #         "right_ankle": 0.0,
                #     },
                #     deviation_weights={
                #         "left_shoulder_pitch": 1.0,
                #         "left_shoulder_yaw": 1.0,
                #         "left_elbow": 1.0,
                #         "right_shoulder_pitch": 1.0,
                #         "right_shoulder_yaw": 1.0,
                #         "right_elbow": 1.0,
                #         "left_hip_pitch": 2.0,
                #         "left_hip_roll": 2.0,
                #         "left_hip_yaw": 2.0,
                #         "left_knee": 1.0,
                #         "left_ankle": 1.0,
                #         "right_hip_pitch": 2.0,
                #         "right_hip_roll": 2.0,
                #         "right_hip_yaw": 2.0,
                #         "right_knee": 1.0,
                #         "right_ankle": 1.0,
                #     },
                # ),
                DHHealthyReward(
                    scale=0.5,
                ),
                DHForwardReward(scale=0.2),
            ],
            observations=[
                BaseOrientationObservation(noise_type="gaussian"),
                BaseLinearVelocityObservation(noise_type="gaussian"),
                BaseAngularVelocityObservation(noise_type="gaussian"),
                JointPositionObservation(noise_type="gaussian"),
                JointVelocityObservation(noise_type="gaussian"),
                CenterOfMassInertiaObservation(noise_type="gaussian"),
                CenterOfMassVelocityObservation(noise_type="gaussian"),
                ActuatorForceObservation(noise_type="gaussian"),
                SensorObservationBuilder(sensor_name="IMU_acc"),  # Sensor has noise already.
                SensorObservationBuilder(sensor_name="IMU_gyro"),  # Sensor has noise already.
            ],
            commands=[
                LinearVelocityCommand(
                    x_scale=0.0,
                    y_scale=0.0,
                    switch_prob=0.02,
                    zero_prob=0.3,
                ),
                AngularVelocityCommand(
                    scale=0.0,
                    switch_prob=0.02,
                    zero_prob=0.8,
                ),
            ],
        )

    def get_model(self, key: PRNGKeyArray) -> ActorCriticAgent:
        """Get the model."""
        return ActorCriticAgent(
            actor_module=ZBot2LearnedStdActorModel(
                MLP(
                    out_dim=NUM_OUTPUTS * 2,  # 2x for std prediction
                    hidden_dims=(64,) * 5,
                    activation=nn.relu,
                    bias_init=nn.initializers.zeros,
                ),
            ),
            critic_module=ZBot2CriticModel(
                MLP(
                    out_dim=1,
                    hidden_dims=(64,) * 5,
                    activation=nn.relu,
                    bias_init=nn.initializers.zeros,
                ),
            ),
            distribution=TanhGaussianDistribution(action_dim=NUM_OUTPUTS),
        )

    def get_init_actor_carry(self) -> jnp.ndarray | None:
        """Get the initial actor carry."""
        return None

    def get_init_critic_carry(self) -> None:
        """Get the initial critic carry."""
        return None


if __name__ == "__main__":
    # python -m examples.zbot2.walking
    ZBot2WalkingTask.launch(
        ZBot2WalkingConfig(
            num_learning_epochs=8,
            num_env_states_per_minibatch=8192,
            num_minibatches=32,
            num_envs=2048,
            dt=0.001,
            ctrl_dt=0.005,
            learning_rate=0.00005,
            save_every_n_steps=50,
            only_save_most_recent=False,
            reward_scaling_alpha=0.0,
            obs_norm_alpha=0.0,
            solver_iterations=6,
            solver_ls_iterations=6,
            actuator_type="mit",
            scale_rewards=False,
            gamma=0.97,
            lam=0.95,
            normalize_advantage=True,
            normalize_advantage_in_minibatch=True,
            entropy_coef=0.001,
            clip_param=0.2,
            use_clipped_value_loss=False,
            max_grad_norm=1.0,
        ),
    )
