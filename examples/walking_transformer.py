# mypy: disable-error-code="override"
"""Defines simple task for training a walking policy for the default humanoid using an GRU actor."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

import ksim

from .walking import (
    NUM_JOINTS,
    AuxOutputs,
    DefaultHumanoidActor,
    HumanoidWalkingTask,
    HumanoidWalkingTaskConfig,
)

HIDDEN_SIZE = 256
NUM_HEADS = HIDDEN_SIZE // 64
TRANSFORMER_DEPTH = 2
DEPTH = 3


class TransformerBlock(eqx.Module):
    """A single transformer block with self-attention and feed-forward network."""

    self_attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm

    def __init__(self, key: PRNGKeyArray, *, hidden_size: int, num_heads: int) -> None:
        key1, key2 = jax.random.split(key)
        self.self_attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            key_size=hidden_size,
            value_size=hidden_size,
            output_size=hidden_size,
            key=key1,
        )
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size,
            width_size=hidden_size,  # Deviation from typical Transformer architecture.
            depth=2,
            key=key2,
            activation=jax.nn.gelu,
        )
        self.norm1 = eqx.nn.LayerNorm(hidden_size)
        self.norm2 = eqx.nn.LayerNorm(hidden_size)

    def __call__(self, x: Array) -> Array:
        # Self-attention
        attn_out = self.self_attn(x, x, x)
        x = x + attn_out
        x = jax.vmap(self.norm1, in_axes=0)(x)

        # Feed-forward
        mlp_out = jax.vmap(self.mlp, in_axes=0)(x)
        x = x + mlp_out
        x = jax.vmap(self.norm2, in_axes=0)(x)
        return x


class DefaultHumanoidTransformerCritic(eqx.Module):
    """Transformer-based critic for the walking task."""

    input_proj: eqx.nn.Linear
    transformer_blocks: tuple[TransformerBlock, ...]
    output_proj: eqx.nn.MLP
    hidden_size: int = eqx.static_field()

    def __init__(self, key: PRNGKeyArray, *, hidden_size: int) -> None:
        num_inputs = NUM_JOINTS + NUM_JOINTS + 160 + 96 + 3 + 3 + NUM_JOINTS + 3 + 4 + 3 + 3 + 1 + 1 + 1 + 1
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create transformer blocks
        transformer_blocks = []
        for _ in range(TRANSFORMER_DEPTH):
            key, transformer_block_key = jax.random.split(key)
            transformer_blocks.append(
                TransformerBlock(
                    key=transformer_block_key,
                    hidden_size=hidden_size,
                    num_heads=NUM_HEADS,
                )
            )
        self.transformer_blocks = tuple(transformer_blocks)

        # Project to output
        self.output_proj = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=num_outputs,
            width_size=hidden_size,
            depth=DEPTH,
            key=key,
            activation=jax.nn.gelu,
        )

        self.hidden_size = hidden_size

    def forward(
        self,
        dh_joint_pos_tj: Array,
        dh_joint_vel_tj: Array,
        com_inertia_tn: Array,
        com_vel_tn: Array,
        imu_acc_t3: Array,
        imu_gyro_t3: Array,
        act_frc_obs_tn: Array,
        base_pos_t3: Array,
        base_quat_t4: Array,
        lin_vel_obs_t3: Array,
        ang_vel_obs_t3: Array,
        lin_vel_cmd_x_t1: Array,
        lin_vel_cmd_y_t1: Array,
        ang_vel_cmd_z_t1: Array,
        done_t: Array,
    ) -> Array:
        obs_tn = jnp.concatenate(
            [
                dh_joint_pos_tj,  # NUM_JOINTS
                dh_joint_vel_tj,  # NUM_JOINTS
                com_inertia_tn,  # 160
                com_vel_tn,  # 96
                imu_acc_t3,  # 3
                imu_gyro_t3,  # 3
                act_frc_obs_tn,  # NUM_JOINTS
                base_pos_t3,  # 3
                base_quat_t4,  # 4
                lin_vel_obs_t3,  # 3
                ang_vel_obs_t3,  # 3
                lin_vel_cmd_x_t1,  # 1
                lin_vel_cmd_y_t1,  # 1
                ang_vel_cmd_z_t1,  # 1
                done_t.reshape(-1, 1),  # 1
            ],
            axis=-1,
        )

        # Project input to hidden size
        x = jax.vmap(self.input_proj, in_axes=0)(obs_tn)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Project to output
        out = jax.vmap(self.output_proj, in_axes=0)(x)

        # Return output and dummy hidden states (unused for transformer)
        return out


class DefaultHumanoidModel(eqx.Module):
    actor: DefaultHumanoidActor
    critic: DefaultHumanoidTransformerCritic

    def __init__(self, key: PRNGKeyArray) -> None:
        self.actor = DefaultHumanoidActor(key, min_std=0.01, max_std=1.0, var_scale=1.0)
        self.critic = DefaultHumanoidTransformerCritic(key, hidden_size=HIDDEN_SIZE)


@dataclass
class HumanoidWalkingTransformerTaskConfig(HumanoidWalkingTaskConfig):
    pass


Config = TypeVar("Config", bound=HumanoidWalkingTransformerTaskConfig)


class HumanoidWalkingTransformerTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidModel:
        return DefaultHumanoidModel(key)

    def _run_critic(
        self,
        model: DefaultHumanoidTransformerCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        done: Array,
    ) -> Array:
        dh_joint_pos_tj = observations["joint_position_observation"]
        dh_joint_vel_tj = observations["joint_velocity_observation"]
        com_inertia_tn = observations["center_of_mass_inertia_observation"]
        com_vel_tn = observations["center_of_mass_velocity_observation"]
        imu_acc_t3 = observations["sensor_observation_imu_acc"]
        imu_gyro_t3 = observations["sensor_observation_imu_gyro"]
        act_frc_obs_tn = observations["actuator_force_observation"]
        base_pos_t3 = observations["base_position_observation"]
        base_quat_t4 = observations["base_orientation_observation"]
        lin_vel_obs_t3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_t3 = observations["base_angular_velocity_observation"]
        lin_vel_cmd_x_t1 = commands["linear_velocity_command_x"]
        lin_vel_cmd_y_t1 = commands["linear_velocity_command_y"]
        ang_vel_cmd_z_t1 = commands["angular_velocity_command_z"]

        return model.forward(
            dh_joint_pos_tj=dh_joint_pos_tj,
            dh_joint_vel_tj=dh_joint_vel_tj / 10.0,
            com_inertia_tn=com_inertia_tn,
            com_vel_tn=com_vel_tn,
            imu_acc_t3=imu_acc_t3 / 50.0,
            imu_gyro_t3=imu_gyro_t3 / 3.0,
            act_frc_obs_tn=act_frc_obs_tn / 100.0,
            base_pos_t3=base_pos_t3,
            base_quat_t4=base_quat_t4,
            lin_vel_obs_t3=lin_vel_obs_t3,
            ang_vel_obs_t3=ang_vel_obs_t3,
            lin_vel_cmd_x_t1=lin_vel_cmd_x_t1,
            lin_vel_cmd_y_t1=lin_vel_cmd_y_t1,
            ang_vel_cmd_z_t1=ang_vel_cmd_z_t1,
            done_t=done,
        )

    def get_on_policy_variables(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> ksim.PPOVariables:
        # Use cached log probabilities from rollout.
        if not isinstance(trajectories.aux_outputs, AuxOutputs):
            raise ValueError("No aux outputs found in trajectories")

        # Gets the value by calling the critic.
        values_t1 = self._run_critic(
            model=model.critic,
            observations=trajectories.obs,
            commands=trajectories.command,
            done=trajectories.done,
        )

        return ksim.PPOVariables(
            log_probs_tn=trajectories.aux_outputs.log_probs,
            values_t=values_t1.squeeze(-1),
        )

    def get_off_policy_variables(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> ksim.PPOVariables:
        # Vectorize over the time dimensions.
        par_actor_fn = jax.vmap(self._run_actor, in_axes=(None, 0, 0))
        action_dist_tj = par_actor_fn(model.actor, trajectories.obs, trajectories.command)
        log_probs_tj = action_dist_tj.log_prob(trajectories.action)

        # Gets the value by calling the critic.
        values_t1 = self._run_critic(
            model=model.critic,
            observations=trajectories.obs,
            commands=trajectories.command,
            done=trajectories.done,
        )

        return ksim.PPOVariables(
            log_probs_tn=log_probs_tj,
            values_t=values_t1.squeeze(-1),
        )

    def sample_action(
        self,
        model: DefaultHumanoidModel,
        carry: None,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, None, AuxOutputs]:
        # Runs the actor model to get the action distribution.
        action_dist_tj = self._run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
        )

        action_tj = action_dist_tj.sample(seed=rng)
        action_log_prob_tj = action_dist_tj.log_prob(action_tj)

        aux_outputs = AuxOutputs(
            log_probs=action_log_prob_tj,
        )

        return action_tj, None, aux_outputs


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.default_humanoid.walking_gru
    # To visualize the environment, use the following command:
    #   python -m examples.default_humanoid.walking_gru run_environment=True
    HumanoidWalkingTransformerTask.launch(
        HumanoidWalkingTransformerTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=32,
            epochs_per_log_step=1,
            rollout_length_seconds=2.0,
            num_rollout_levels=3,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
        ),
    )
