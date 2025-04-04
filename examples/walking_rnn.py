# mypy: disable-error-code="override"
"""Defines simple task for training a walking policy for the default humanoid using an RNN actor."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

import ksim

from .walking import (
    NUM_INPUTS,
    NUM_JOINTS,
    HumanoidWalkingTask,
    HumanoidWalkingTaskConfig,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array
    actor_carry: Array
    critic_carry: Array


class DefaultHumanoidRNNActor(eqx.Module):
    """RNN-based actor for the walking task."""

    input_proj: eqx.nn.Linear
    rnn: eqx.nn.GRUCell
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

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnn = eqx.nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
            key=rnn_key,
        )

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
        dh_joint_pos_j: Array,
        dh_joint_vel_j: Array,
        com_inertia_n: Array,
        com_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        act_frc_obs_n: Array,
        base_pos_3: Array,
        base_quat_4: Array,
        lin_vel_obs_3: Array,
        ang_vel_obs_3: Array,
        lin_vel_cmd_x_1: Array,
        lin_vel_cmd_y_1: Array,
        ang_vel_cmd_z_1: Array,
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                act_frc_obs_n,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
                lin_vel_cmd_x_1,  # 1
                lin_vel_cmd_y_1,  # 1
                ang_vel_cmd_z_1,  # 1
            ],
            axis=-1,
        )

        x_n = self.input_proj(obs_n)
        x_n = self.rnn(x_n, carry)
        out_n = self.output_proj(x_n)

        # Converts the output to a distribution.
        mean_n = out_n[..., :NUM_JOINTS]
        std_n = out_n[..., NUM_JOINTS:]

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        dist_n = distrax.Normal(mean_n, std_n)
        return dist_n, x_n


class DefaultHumanoidRNNCritic(eqx.Module):
    """RNN-based critic for the walking task."""

    input_proj: eqx.nn.Linear
    rnn: eqx.nn.GRUCell
    output_proj: eqx.nn.MLP

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
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

        # Create RNN layer
        key, rnn_key = jax.random.split(key)
        self.rnn = eqx.nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
            key=rnn_key,
        )

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
        dh_joint_pos_j: Array,
        dh_joint_vel_j: Array,
        com_inertia_n: Array,
        com_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        act_frc_obs_n: Array,
        base_pos_3: Array,
        base_quat_4: Array,
        lin_vel_obs_3: Array,
        ang_vel_obs_3: Array,
        lin_vel_cmd_x_1: Array,
        lin_vel_cmd_y_1: Array,
        ang_vel_cmd_z_1: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                act_frc_obs_n,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
                lin_vel_cmd_x_1,  # 1
                lin_vel_cmd_y_1,  # 1
                ang_vel_cmd_z_1,  # 1
            ],
            axis=-1,
        )

        x_n = self.input_proj(obs_n)
        x_n = self.rnn(x_n, carry)
        out_n = self.output_proj(x_n)

        return out_n, x_n


class DefaultHumanoidRNNModel(eqx.Module):
    actor: DefaultHumanoidRNNActor
    critic: DefaultHumanoidRNNCritic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        hidden_size: int,
        depth: int,
    ) -> None:
        self.actor = DefaultHumanoidRNNActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=0.5,
            hidden_size=hidden_size,
            depth=depth,
        )
        self.critic = DefaultHumanoidRNNCritic(
            key,
            hidden_size=hidden_size,
            depth=depth,
        )


@dataclass
class HumanoidWalkingRNNTaskConfig(HumanoidWalkingTaskConfig):
    # Model parameters.
    hidden_size: int = xax.field(
        value=256,
        help="The hidden size for the RNN.",
    )
    depth: int = xax.field(
        value=5,
        help="The depth for the MLPs.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingRNNTaskConfig)


class HumanoidWalkingRNNTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidRNNModel:
        return DefaultHumanoidRNNModel(
            key,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
        )

    def _run_actor(
        self,
        model: DefaultHumanoidRNNActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]
        lin_vel_cmd_x_1 = commands["linear_velocity_command_x"]
        lin_vel_cmd_y_1 = commands["linear_velocity_command_y"]
        ang_vel_cmd_z_1 = commands["angular_velocity_command_z"]

        return model.forward(
            timestep_1=timestep_1,
            dh_joint_pos_j=dh_joint_pos_j,
            dh_joint_vel_j=dh_joint_vel_j / 10.0,
            com_inertia_n=com_inertia_n,
            com_vel_n=com_vel_n,
            imu_acc_3=imu_acc_3 / 50.0,
            imu_gyro_3=imu_gyro_3 / 3.0,
            act_frc_obs_n=act_frc_obs_n / 100.0,
            base_pos_3=base_pos_3,
            base_quat_4=base_quat_4,
            lin_vel_obs_3=lin_vel_obs_3,
            ang_vel_obs_3=ang_vel_obs_3,
            lin_vel_cmd_x_1=lin_vel_cmd_x_1,
            lin_vel_cmd_y_1=lin_vel_cmd_y_1,
            ang_vel_cmd_z_1=ang_vel_cmd_z_1,
            carry=carry,
        )

    def _run_critic(
        self,
        model: DefaultHumanoidRNNCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]
        lin_vel_cmd_x_1 = commands["linear_velocity_command_x"]
        lin_vel_cmd_y_1 = commands["linear_velocity_command_y"]
        ang_vel_cmd_z_1 = commands["angular_velocity_command_z"]

        return model.forward(
            timestep_1=timestep_1,
            dh_joint_pos_j=dh_joint_pos_j,
            dh_joint_vel_j=dh_joint_vel_j / 10.0,
            com_inertia_n=com_inertia_n,
            com_vel_n=com_vel_n,
            imu_acc_3=imu_acc_3 / 50.0,
            imu_gyro_3=imu_gyro_3 / 3.0,
            act_frc_obs_n=act_frc_obs_n / 100.0,
            base_pos_3=base_pos_3,
            base_quat_4=base_quat_4,
            lin_vel_obs_3=lin_vel_obs_3,
            ang_vel_obs_3=ang_vel_obs_3,
            lin_vel_cmd_x_1=lin_vel_cmd_x_1,
            lin_vel_cmd_y_1=lin_vel_cmd_y_1,
            ang_vel_cmd_z_1=ang_vel_cmd_z_1,
            carry=carry,
        )

    def get_on_policy_variables(
        self,
        model: DefaultHumanoidRNNModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> ksim.PPOVariables:
        # Use cached values from rollout.
        if not isinstance(trajectories.aux_outputs, AuxOutputs):
            raise ValueError("No aux outputs found in trajectories")

        return ksim.PPOVariables(
            log_probs_tn=trajectories.aux_outputs.log_probs,
            values_t=trajectories.aux_outputs.values,
        )

    def get_off_policy_variables(
        self,
        model: DefaultHumanoidRNNModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> ksim.PPOVariables:
        # Use cached values from rollout.
        if not isinstance(trajectories.aux_outputs, AuxOutputs):
            raise ValueError("No aux outputs found in trajectories")

        actor_carry = trajectories.aux_outputs.actor_carry
        critic_carry = trajectories.aux_outputs.critic_carry

        # Vectorize over the time dimensions.
        par_actor_fn = jax.vmap(self._run_actor, in_axes=(None, 0, 0, 0))
        action_dist_tj, _ = par_actor_fn(model.actor, trajectories.obs, trajectories.command, actor_carry)
        log_probs_tj = action_dist_tj.log_prob(trajectories.action)

        # Gets the value by calling the critic.
        par_critic_fn = jax.vmap(self._run_critic, in_axes=(None, 0, 0, 0))
        values_t1, _ = par_critic_fn(model.critic, trajectories.obs, trajectories.command, critic_carry)

        return ksim.PPOVariables(
            log_probs_tn=log_probs_tj,
            values_t=values_t1.squeeze(-1),
        )

    def get_initial_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return jnp.zeros(shape=(self.config.hidden_size,)), jnp.zeros(shape=(self.config.hidden_size,))

    def sample_action(
        self,
        model: DefaultHumanoidRNNModel,
        carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, tuple[Array, Array], AuxOutputs]:
        actor_carry_in, critic_carry_in = carry

        # Runs the actor model to get the action distribution.
        action_dist_j, actor_carry = self._run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )

        action_j = action_dist_j.sample(seed=rng)
        action_log_prob_j = action_dist_j.log_prob(action_j)

        values_1, critic_carry = self._run_critic(
            model=model.critic,
            observations=observations,
            commands=commands,
            carry=critic_carry_in,
        )

        # Remove time dimension.
        values = values_1.squeeze(-1)

        return (
            action_j,
            (actor_carry, critic_carry),
            AuxOutputs(
                log_probs=action_log_prob_j,
                values=values,
                actor_carry=actor_carry_in,
                critic_carry=critic_carry_in,
            ),
        )


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.walking_rnn
    # To visualize the environment, use the following command:
    #   python -m examples.walking_rnn run_environment=True
    HumanoidWalkingRNNTask.launch(
        HumanoidWalkingRNNTaskConfig(
            # Training parameters.
            num_envs=4096,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=10,
            rollout_length_seconds=10.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
        ),
    )
