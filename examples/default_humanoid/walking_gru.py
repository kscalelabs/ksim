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

from .walking import NUM_JOINTS, AuxOutputs, DefaultHumanoidCritic, HumanoidWalkingTask, HumanoidWalkingTaskConfig

HIDDEN_SIZE = 512  # `_s`
DEPTH = 2


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
    mean_scale: float = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        var_scale: float,
        mean_scale: float,
        hidden_size: int,
    ) -> None:
        num_inputs = NUM_JOINTS + NUM_JOINTS + 3 + 3 + 2 + 1
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
        self.mean_scale = mean_scale
        self.hidden_size = hidden_size

    def __call__(
        self,
        dh_joint_pos_n: Array,
        dh_joint_vel_n: Array,
        imu_acc_3: Array,
        imu_gyro_3: Array,
        lin_vel_cmd_2: Array,
        ang_vel_cmd_1: Array,
        hidden_states: Array,
    ) -> tuple[distrax.Distribution, Array]:
        obs_n = jnp.concatenate(
            [
                dh_joint_pos_n,  # NUM_JOINTS
                dh_joint_vel_n,  # NUM_JOINTS
                imu_acc_3,  # 3
                imu_gyro_3,  # 3
                lin_vel_cmd_2,  # 2
                ang_vel_cmd_1,  # 1
            ],
            axis=-1,
        )

        # Process through GRU cell
        last_h, new_hidden_states = self.multi_layer_gru(obs_n, hidden_states)
        out_n = self.projector(last_h)

        mean_n = out_n[..., :NUM_JOINTS]
        std_n = out_n[..., NUM_JOINTS:]

        # Scale the mean.
        mean_n = jnp.tanh(mean_n) * self.mean_scale

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        # Parametrizes the action distribution.
        action_dist = distrax.Transformed(distrax.Normal(mean_n, std_n), distrax.Tanh())

        return action_dist, new_hidden_states


class DefaultHumanoidModel(eqx.Module):
    actor: DefaultHumanoidActor
    critic: DefaultHumanoidCritic

    def __init__(self, key: PRNGKeyArray) -> None:
        self.actor = DefaultHumanoidActor(
            key,
            min_std=0.01,
            max_std=1.0,
            var_scale=1.0,
            mean_scale=1.0,
            hidden_size=HIDDEN_SIZE,
        )
        self.critic = DefaultHumanoidCritic(key)


@dataclass
class HumanoidWalkingGRUTaskConfig(HumanoidWalkingTaskConfig):
    pass


Config = TypeVar("Config", bound=HumanoidWalkingGRUTaskConfig)


class HumanoidWalkingGRUTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidModel:
        return DefaultHumanoidModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> Array:
        # Initialize the hidden state for GRU
        return jnp.zeros((DEPTH, HIDDEN_SIZE))

    def _run_actor(
        self,
        model: DefaultHumanoidModel,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        dh_joint_pos_n = observations["joint_position_observation"]
        dh_joint_vel_n = observations["joint_velocity_observation"]
        imu_acc_3 = observations["sensor_observation_imu_acc"]
        imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        ang_vel_cmd_1 = commands["angular_velocity_command"]

        return model.actor(
            dh_joint_pos_n=dh_joint_pos_n,
            dh_joint_vel_n=dh_joint_vel_n / 10.0,
            imu_acc_3=imu_acc_3 / 50.0,
            imu_gyro_3=imu_gyro_3 / 3.0,
            lin_vel_cmd_2=lin_vel_cmd_2,
            ang_vel_cmd_1=ang_vel_cmd_1,
            hidden_states=carry,
        )

    def get_log_probs(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> tuple[Array, None]:
        def scan_fn(
            carry: Array,
            inputs: ksim.Trajectory,
        ) -> tuple[Array, tuple[Array, Array]]:
            action_dist_n, carry = self._run_actor(model, inputs.obs, inputs.command, carry)
            log_probs_n = action_dist_n.log_prob(inputs.action / model.actor.mean_scale)
            return carry, log_probs_n

        # Runs the model over the sequence.
        initial_hidden_states = self.get_initial_carry(rng)
        _, log_probs_tn = jax.lax.scan(scan_fn, initial_hidden_states, trajectories)

        return log_probs_tn, None

    def get_values(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        # Vectorize over both batch and time dimensions.
        par_fn = jax.vmap(self._run_critic, in_axes=(None, 0, 0))
        values_bt1 = par_fn(model.critic, trajectories.obs, trajectories.command)

        # Remove the last dimension.
        return values_bt1.squeeze(-1)

    def sample_action(
        self,
        model: DefaultHumanoidModel,
        carry: Array,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array, AuxOutputs]:
        action_dist_n, next_carry = self._run_actor(model, observations, commands, carry)
        action_n = action_dist_n.sample(seed=rng)
        action_log_prob_n = action_dist_n.log_prob(action_n)

        critic_n = self._run_critic(model.critic, observations, commands)
        value_n = critic_n.squeeze(-1)
        return action_n, next_carry, AuxOutputs(log_probs=action_log_prob_n, values=value_n)


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
            rollout_length_seconds=4.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
        ),
    )
