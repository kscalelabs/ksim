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

import ksim

from .walking import (
    NUM_JOINTS,
    AuxOutputs,
    HumanoidWalkingTask,
    HumanoidWalkingTaskConfig,
    map_tanh_distribution,
)

HIDDEN_SIZE = 64  # `_s`
DEPTH = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Carry:
    actor: Array
    critic: Array


class MultiLayerGRU(eqx.Module):
    layers: tuple[eqx.nn.GRUCell, ...]
    depth: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, key: PRNGKeyArray, *, input_size: int, hidden_size: int, depth: int) -> None:
        if depth < 1:
            raise ValueError("Depth must be at least 1")
        first_layer = eqx.nn.GRUCell(input_size=input_size, hidden_size=hidden_size, use_bias=True, key=key)

        other_layers = tuple(
            eqx.nn.GRUCell(input_size=hidden_size, hidden_size=hidden_size, use_bias=True, key=key)
            for _ in range(depth - 1)
        )

        self.layers = (first_layer, *other_layers)
        self.depth = depth
        self.input_size = input_size
        self.hidden_size = hidden_size

    def __call__(self, x_n: Array, hidden_states: Array) -> tuple[Array, Array]:
        new_h_states = []
        for layer, h_state in zip(self.layers, hidden_states):
            x_n = layer(x_n, h_state)
            new_h_states.append(x_n)
        stacked_h = jnp.stack(new_h_states, axis=0)
        return x_n, stacked_h


class DefaultHumanoidActor(eqx.Module):
    """Actor for the walking task."""

    multi_layer_gru: MultiLayerGRU
    projector: eqx.nn.MLP
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()
    hidden_size: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
    ) -> None:
        num_inputs = NUM_JOINTS + NUM_JOINTS + 3 + 3 + 2 + 1 + NUM_JOINTS
        num_outputs = NUM_JOINTS

        self.multi_layer_gru = MultiLayerGRU(
            key,
            input_size=num_inputs,
            hidden_size=hidden_size,
            depth=DEPTH,
        )

        self.projector = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=num_outputs * 2,
            width_size=64,
            depth=2,
            key=key,
            activation=jax.nn.relu,
        )

        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.hidden_size = hidden_size

    def __call__(
        self,
        dh_joint_pos_tj: Array,
        dh_joint_vel_tj: Array,
        imu_acc_t3: Array,
        imu_gyro_t3: Array,
        lin_vel_cmd_t2: Array,
        ang_vel_cmd_t1: Array,
        prev_actions_tj: Array,
        hidden_states_dn: Array,
    ) -> tuple[distrax.Distribution, Array]:
        obs_tn = jnp.concatenate(
            [
                dh_joint_pos_tj,  # NUM_JOINTS
                dh_joint_vel_tj,  # NUM_JOINTS
                imu_acc_t3,  # 3
                imu_gyro_t3,  # 3
                lin_vel_cmd_t2,  # 2
                ang_vel_cmd_t1,  # 1
                prev_actions_tj,  # NUM_JOINTS
            ],
            axis=-1,
        )

        def scan_fn(carry: Array, xt: Array) -> tuple[Array, Array]:
            xt, ht = self.multi_layer_gru(xt, carry)
            xt = self.projector(xt)
            return ht, xt

        # Process through GRU cell.
        new_hidden_states, out_tn = jax.lax.scan(scan_fn, hidden_states_dn, obs_tn)

        mean_tn = out_tn[..., :NUM_JOINTS]
        std_tn = out_tn[..., NUM_JOINTS:]

        # Softplus and clip to ensure positive standard deviations.
        std_tn = jnp.clip((jax.nn.softplus(std_tn) + self.min_std) * self.var_scale, max=self.max_std)

        # Parametrizes the action distribution.
        dist = distrax.Normal(mean_tn, std_tn)
        dist = distrax.Transformed(dist, distrax.Tanh())
        dist = map_tanh_distribution(dist)

        return dist, new_hidden_states


class DefaultHumanoidGRUCritic(eqx.Module):
    """GRU-based critic for the walking task."""

    multi_layer_gru: MultiLayerGRU
    projector: eqx.nn.MLP
    hidden_size: int = eqx.static_field()

    def __init__(self, key: PRNGKeyArray, *, hidden_size: int) -> None:
        num_inputs = NUM_JOINTS + NUM_JOINTS + 160 + 96 + 3 + 3 + NUM_JOINTS + 3 + 4 + 3 + 3 + 2 + 1
        num_outputs = 1

        self.multi_layer_gru = MultiLayerGRU(
            key,
            input_size=num_inputs,
            hidden_size=hidden_size,
            depth=DEPTH,
        )

        self.projector = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=num_outputs,
            width_size=64,
            depth=2,
            key=key,
            activation=jax.nn.relu,
        )

        self.hidden_size = hidden_size

    def __call__(
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
        lin_vel_cmd_t2: Array,
        ang_vel_cmd_t1: Array,
        hidden_states_dn: Array,
    ) -> tuple[Array, Array]:
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
                lin_vel_cmd_t2,  # 2
                ang_vel_cmd_t1,  # 1
            ],
            axis=-1,
        )

        breakpoint()

        def scan_fn(carry: Array, xt: Array) -> tuple[Array, Array]:
            xt, ht = self.multi_layer_gru(xt, carry)
            xt = self.projector(xt)
            return ht, xt

        # Process through GRU cell.
        new_hidden_states, out_tn = jax.lax.scan(scan_fn, hidden_states_dn, obs_tn)

        return out_tn, new_hidden_states


class DefaultHumanoidModel(eqx.Module):
    actor: DefaultHumanoidActor
    critic: DefaultHumanoidGRUCritic

    def __init__(self, key: PRNGKeyArray) -> None:
        self.actor = DefaultHumanoidActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=1.0,
            hidden_size=HIDDEN_SIZE,
        )
        self.critic = DefaultHumanoidGRUCritic(key, hidden_size=HIDDEN_SIZE)


@dataclass
class HumanoidWalkingGRUTaskConfig(HumanoidWalkingTaskConfig):
    pass


Config = TypeVar("Config", bound=HumanoidWalkingGRUTaskConfig)


class HumanoidWalkingGRUTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidModel:
        return DefaultHumanoidModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> Carry:
        # Initialize the hidden states for both actor and critic GRUs
        actor_hidden_states = jnp.zeros((DEPTH, HIDDEN_SIZE))
        critic_hidden_states = jnp.zeros((DEPTH, HIDDEN_SIZE))
        return Carry(actor=actor_hidden_states, critic=critic_hidden_states)

    def _run_actor(
        self,
        model: DefaultHumanoidActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        prev_actions_tj: Array,
        hidden_states_dn: Array,
    ) -> tuple[distrax.Distribution, Array]:
        dh_joint_pos_tj = observations["joint_position_observation"]
        dh_joint_vel_tj = observations["joint_velocity_observation"]
        imu_acc_t3 = observations["sensor_observation_imu_acc"]
        imu_gyro_t3 = observations["sensor_observation_imu_gyro"]
        lin_vel_cmd_t2 = commands["linear_velocity_command"]
        ang_vel_cmd_t1 = commands["angular_velocity_command"]

        return model(
            dh_joint_pos_tj=dh_joint_pos_tj,
            dh_joint_vel_tj=dh_joint_vel_tj / 10.0,
            imu_acc_t3=imu_acc_t3 / 50.0,
            imu_gyro_t3=imu_gyro_t3 / 3.0,
            lin_vel_cmd_t2=lin_vel_cmd_t2,
            ang_vel_cmd_t1=ang_vel_cmd_t1,
            prev_actions_tj=prev_actions_tj,
            hidden_states_dn=hidden_states_dn,
        )

    def _run_critic(
        self,
        model: DefaultHumanoidGRUCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        hidden_states_dn: Array,
    ) -> tuple[Array, Array]:
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
        lin_vel_cmd_t2 = commands["linear_velocity_command"]
        ang_vel_cmd_t1 = commands["angular_velocity_command"]

        return model(
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
            lin_vel_cmd_t2=lin_vel_cmd_t2,
            ang_vel_cmd_t1=ang_vel_cmd_t1,
            hidden_states_dn=hidden_states_dn,
        )

    def get_off_policy_variables(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> ksim.PPOVariables:
        initial_carry = self.get_initial_carry(rng)

        # We need to shift the actions by one time step to get the previous actions.
        actions_tj = trajectories.action
        prev_actions_tj = jnp.concatenate([jnp.zeros_like(actions_tj[..., :1, :]), actions_tj[..., :-1, :]], axis=-2)

        action_dist_tj, _ = self._run_actor(
            model=model.actor,
            observations=trajectories.obs,
            commands=trajectories.command,
            prev_actions_tj=prev_actions_tj,
            hidden_states_dn=initial_carry.actor,
        )
        log_probs_tj = action_dist_tj.log_prob(trajectories.action)

        # Vectorize over both batch and time dimensions.
        values_t1, _ = self._run_critic(
            model=model.critic,
            observations=trajectories.obs,
            commands=trajectories.command,
            hidden_states_dn=initial_carry.critic,
        )

        return ksim.PPOVariables(
            log_probs_tj=log_probs_tj,
            values_t=values_t1.squeeze(-1),
        )

    def sample_action(
        self,
        model: DefaultHumanoidModel,
        carry: Carry,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, Carry, AuxOutputs]:
        # Unsqueeze first dimension as the time dimension.
        (observations_t, commands_t, prev_actions_tj) = jax.tree.map(
            lambda x: x[None, ...],
            (observations, commands, physics_state.most_recent_action),
        )

        # Runs the actor model to get the action distribution and next hidden states.
        action_dist_tj, next_actor_hidden_states = self._run_actor(
            model=model.actor,
            observations=observations_t,
            commands=commands_t,
            prev_actions_tj=prev_actions_tj,
            hidden_states_dn=carry.actor,
        )

        action_tj = action_dist_tj.sample(seed=rng).squeeze(0)
        action_log_prob_tj = action_dist_tj.log_prob(action_tj).squeeze(0)

        # Run critic with its own hidden states
        critic_1, next_critic_hidden_states = self._run_critic(
            model=model.critic,
            observations=observations_t,
            commands=commands_t,
            hidden_states_dn=carry.critic,
        )
        value = critic_1.squeeze(-1)

        next_carry = Carry(
            actor=next_actor_hidden_states,
            critic=next_critic_hidden_states,
        )

        aux_outputs = AuxOutputs(
            log_probs=action_log_prob_tj,
            values=value,
        )

        return action_tj, next_carry, aux_outputs


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.default_humanoid.walking_gru
    # To visualize the environment, use the following command:
    #   python -m examples.default_humanoid.walking_gru run_environment=True
    HumanoidWalkingGRUTask.launch(
        HumanoidWalkingGRUTaskConfig(
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
