"""Defines simple task for training a walking policy for K-Bot."""

from dataclasses import dataclass

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

import ksim

from .walking import NUM_INPUTS, NUM_OUTPUTS, DHControlPenalty, DHHealthyReward, HumanoidWalkingTask
from .walking_lstm import HumanoidWalkingLSTMTask, HumanoidWalkingLSTMTaskConfig

HIDDEN_SIZE = 128  # `_s`
DEPTH = 2


@attrs.define(frozen=True, kw_only=True)
class UpwardReward(ksim.Reward):
    """Incentives forward movement."""

    velocity_clip: float = attrs.field(default=10.0)

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        # Just try to maximize the velocity in the Z direction.
        z_delta = jnp.clip(trajectory.qvel[..., 2], 0, self.velocity_clip)
        return z_delta


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


class MultiLayerLSTM(eqx.Module):
    layers: tuple[eqx.nn.LSTMCell, ...]
    depth: int = eqx.field(static=True)
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, key: PRNGKeyArray, *, input_size: int, hidden_size: int, depth: int) -> None:
        if depth < 1:
            raise ValueError("Depth must be at least 1")
        first_layer = eqx.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size, use_bias=True, key=key)

        other_layers = tuple(
            eqx.nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size, use_bias=True, key=key)
            for _ in range(depth - 1)
        )

        self.layers = (first_layer, *other_layers)
        self.depth = depth
        self.input_size = input_size
        self.hidden_size = hidden_size

    def __call__(
        self,
        x_n: Array,
        hidden_states: Array,  # (depth, 2, hidden_size)
    ) -> tuple[Array, Array, Array]:  # (output_h, output_c, new_hidden_states)
        h_states = hidden_states[:, 0]  # All h states
        c_states = hidden_states[:, 1]  # All c states

        new_h_states = []
        new_c_states = []

        h, c = self.layers[0](x_n, (h_states[0], c_states[0]))
        new_h_states.append(h)
        new_c_states.append(c)

        if self.depth > 1:
            for layer, h_state, c_state in zip(self.layers[1:], h_states[1:], c_states[1:]):
                h, c = layer(h, (h_state, c_state))
                new_h_states.append(h)
                new_c_states.append(c)

        stacked_h = jnp.stack(new_h_states, axis=0)  # (depth, hidden_size)
        stacked_c = jnp.stack(new_c_states, axis=0)  # (depth, hidden_size)

        return h, c, jnp.stack([stacked_h, stacked_c], axis=1)  # h_last, c_last, (depth, 2, hidden_size)


class DefaultHumanoidActor(eqx.Module):
    """Actor for the walking task."""

    multi_layer_lstm: MultiLayerLSTM
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
        self.multi_layer_lstm = MultiLayerLSTM(
            key,
            input_size=NUM_INPUTS,
            hidden_size=hidden_size,
            depth=DEPTH,
        )

        self.projector = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=NUM_OUTPUTS * 2,
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
        com_inertia_n: Array,
        com_vel_n: Array,
        act_frc_obs_n: Array,
        lin_vel_cmd_n: Array,
        hidden_states: Array,
    ) -> tuple[distrax.Normal, Array]:
        obs_n = jnp.concatenate([dh_joint_pos_n, dh_joint_vel_n, com_inertia_n, com_vel_n, act_frc_obs_n])

        return self.call_flat_obs(obs_n, lin_vel_cmd_n, hidden_states)

    def call_flat_obs(
        self,
        flat_obs_n: Array,
        lin_vel_cmd_n: Array,
        hidden_states: Array,
    ) -> tuple[distrax.Normal, Array]:
        x_n = jnp.concatenate([flat_obs_n, lin_vel_cmd_n], axis=-1)  # (NUM_INPUTS)

        # Process through LSTM cell
        last_h, _, new_hidden_states = self.multi_layer_lstm(x_n, hidden_states)
        out_n = self.projector(last_h)

        mean_n = out_n[..., :NUM_OUTPUTS]
        std_n = out_n[..., NUM_OUTPUTS:]

        # Scale the mean.
        mean_n = jnp.tanh(mean_n) * self.mean_scale

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        return distrax.Normal(mean_n, std_n), new_hidden_states


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
            mean_scale=1.0,
            hidden_size=HIDDEN_SIZE,
        )
        self.critic = DefaultHumanoidCritic(key)


@dataclass
class HumanoidJumpingLSTMTaskConfig(HumanoidWalkingLSTMTaskConfig):
    pass


class HumanoidJumpingLSTMTask(HumanoidWalkingTask[HumanoidJumpingLSTMTaskConfig]):

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            UpwardReward(scale=0.5),
            DHControlPenalty(scale=-0.01),
            DHHealthyReward(scale=0.5, healthy_z_upper=5.0),
        ]


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.default_humanoid.walking_lstm
    # To visualize the environment, use the following command:
    #   python -m examples.default_humanoid.walking_lstm run_environment=True
    HumanoidWalkingLSTMTask.launch(
        HumanoidWalkingLSTMTaskConfig(
            num_envs=2048,
            num_batches=64,
            num_passes=8,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=10.0,  # This needs to be shorter because of memory constraints.
            eval_rollout_length_seconds=4.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            use_mit_actuators=False,
        ),
    )
