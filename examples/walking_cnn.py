# mypy: disable-error-code="override"
"""Defines simple task for training a walking policy for the default humanoid using an GRU actor."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray
from omegaconf import MISSING

import ksim

from .walking import (
    NUM_INPUTS,
    NUM_JOINTS,
    AuxOutputs,
    HumanoidWalkingTask,
    HumanoidWalkingTaskConfig,
)


class DefaultHumanoidCNNActor(eqx.Module):
    """CNN-based actor for the walking task."""

    input_proj: eqx.nn.Linear
    cnn_blocks: tuple[eqx.nn.Conv1d, ...]
    output_proj: eqx.nn.MLP
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
        hidden_size: int,
        depth: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        num_inputs = NUM_INPUTS
        num_outputs = NUM_JOINTS

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create convolution blocks.
        cnn_blocks = []
        for _ in range(depth):
            key, cnn_key = jax.random.split(key)
            cnn_blocks.append(
                eqx.nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=[((kernel_size - 1) * dilation, 0)],  # Left-padding to keep the length constant.
                    key=cnn_key,
                )
            )
        self.cnn_blocks = tuple(cnn_blocks)

        # Project to output
        self.output_proj = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=num_outputs * 2,
            width_size=hidden_size,
            depth=depth,
            key=key,
            activation=jax.nn.gelu,
        )

        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(
        self,
        timestep_1: Array,
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
        carry_tn: Array | None = None,
    ) -> tuple[distrax.Distribution, Array]:
        obs_tn = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
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
            ],
            axis=-1,
        )

        if carry_tn is not None:
            obs_tn = jnp.concatenate([carry_tn, obs_tn], axis=-2)

        # Project input to hidden size
        x_tn = jax.vmap(self.input_proj, in_axes=0)(obs_tn)
        x_nt = x_tn.transpose(1, 0)

        # Apply transformer blocks
        for block in self.cnn_blocks:
            x_nt = block(x_nt)
            x_nt = jax.nn.relu(x_nt)

        # Project to output
        x_tn = x_nt.transpose(1, 0)
        out_tn = jax.vmap(self.output_proj, in_axes=0)(x_tn)

        # Remove the part of the output that was the carry.
        if carry_tn is not None:
            out_tn = out_tn[..., carry_tn.shape[-2] :, :]

        # Converts the output to a distribution.
        mean_tn = out_tn[..., :NUM_JOINTS]
        std_tn = out_tn[..., NUM_JOINTS:]

        # Softplus and clip to ensure positive standard deviations.
        std_tn = jnp.clip((jax.nn.softplus(std_tn) + self.min_std) * self.var_scale, max=self.max_std)

        dist_tn = distrax.Normal(mean_tn, std_tn)
        return dist_tn, obs_tn


class DefaultHumanoidCNNCritic(eqx.Module):
    """CNN-based critic for the walking task."""

    input_proj: eqx.nn.Linear
    cnn_blocks: tuple[eqx.nn.Conv1d, ...]
    output_proj: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        num_inputs = NUM_INPUTS
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create convolution blocks.
        cnn_blocks = []
        for _ in range(depth):
            key, cnn_key = jax.random.split(key)
            cnn_blocks.append(
                eqx.nn.Conv1d(
                    in_channels=hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=[((kernel_size - 1) * dilation, 0)],  # Left-padding to keep the length constant.
                    key=cnn_key,
                )
            )
        self.cnn_blocks = tuple(cnn_blocks)

        # Project to output
        self.output_proj = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=num_outputs,
            width_size=hidden_size,
            depth=depth,
            key=key,
            activation=jax.nn.gelu,
        )

    def forward(
        self,
        timestep_1: Array,
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
    ) -> Array:
        obs_tn = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
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
            ],
            axis=-1,
        )

        # Project input to hidden size
        x_tn = jax.vmap(self.input_proj, in_axes=0)(obs_tn)
        x_nt = x_tn.transpose(1, 0)

        # Apply transformer blocks
        for block in self.cnn_blocks:
            x_nt = block(x_nt)

        # Project to output
        x_tn = x_nt.transpose(1, 0)
        out_tn = jax.vmap(self.output_proj, in_axes=0)(x_tn)
        return out_tn


class DefaultHumanoidCNNModel(eqx.Module):
    actor: DefaultHumanoidCNNActor
    critic: DefaultHumanoidCNNCritic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        self.actor = DefaultHumanoidCNNActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=0.5,
            hidden_size=hidden_size,
            depth=depth,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.critic = DefaultHumanoidCNNCritic(
            key,
            hidden_size=hidden_size,
            depth=depth,
            kernel_size=kernel_size,
            dilation=dilation,
        )


@dataclass
class HumanoidWalkingCNNTaskConfig(HumanoidWalkingTaskConfig):
    # Model parameters.
    kernel_size: int = xax.field(
        value=MISSING,
        help="The kernel size for the CNN.",
    )
    dilation: int = xax.field(
        value=1,
        help="The dilation for the CNN.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingCNNTaskConfig)


class HumanoidWalkingCNNTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidCNNModel:
        return DefaultHumanoidCNNModel(
            key,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            kernel_size=self.config.kernel_size,
            dilation=self.config.dilation,
        )

    def _run_actor(
        self,
        model: DefaultHumanoidCNNActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry_tn: Array | None = None,
    ) -> tuple[distrax.Distribution, Array]:
        timestep_1 = observations["timestep_observation"]
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
            timestep_1=timestep_1,
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
            carry_tn=carry_tn,
        )

    def _run_critic(
        self,
        model: DefaultHumanoidCNNCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        timestep_1 = observations["timestep_observation"]
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
            timestep_1=timestep_1,
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
        )

    def get_on_policy_variables(
        self,
        model: DefaultHumanoidCNNModel,
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
        )

        return ksim.PPOVariables(
            log_probs_tn=trajectories.aux_outputs.log_probs,
            values_t=values_t1.squeeze(-1),
        )

    def get_off_policy_variables(
        self,
        model: DefaultHumanoidCNNModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> ksim.PPOVariables:
        # Vectorize over the time dimensions.
        action_dist_tj, _ = self._run_actor(model.actor, trajectories.obs, trajectories.command)
        log_probs_tj = action_dist_tj.log_prob(trajectories.action)

        # Gets the value by calling the critic.
        values_t1 = self._run_critic(
            model=model.critic,
            observations=trajectories.obs,
            commands=trajectories.command,
        )

        return ksim.PPOVariables(
            log_probs_tn=log_probs_tj,
            values_t=values_t1.squeeze(-1),
        )

    def get_initial_carry(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(shape=((self.config.kernel_size - 1) * self.config.dilation, NUM_INPUTS))

    def sample_action(
        self,
        model: DefaultHumanoidCNNModel,
        carry: Array,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array, AuxOutputs]:
        (observations, commands) = jax.tree.map(lambda x: x[None], (observations, commands))

        # Runs the actor model to get the action distribution.
        action_dist_tj, obs_tn = self._run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry_tn=carry,
        )

        # Get the true observation and the next carry.
        carry = obs_tn[..., 1:, :]

        action_tj = action_dist_tj.sample(seed=rng)
        action_log_prob_tj = action_dist_tj.log_prob(action_tj)

        # Remove time dimension.
        action_j = action_tj.squeeze(-2)
        action_log_prob_j = action_log_prob_tj.squeeze(-2)

        aux_outputs = AuxOutputs(
            log_probs=action_log_prob_j,
        )

        return action_j, carry, aux_outputs


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.walking_cnn
    # To visualize the environment, use the following command:
    #   python -m examples.walking_cnn run_environment=True
    HumanoidWalkingCNNTask.launch(
        HumanoidWalkingCNNTaskConfig(
            # Training parameters.
            num_envs=4096,
            batch_size=256,
            num_passes=4,
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
