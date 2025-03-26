# mypy: disable-error-code="override"
"""Defines simple task for training a standing policy for K-Bot."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

import ksim

from .walking import NUM_INPUTS, NUM_OUTPUTS, HumanoidWalkingTask, HumanoidWalkingTaskConfig

HIDDEN_SIZE = 128  # `_s`
DEPTH = 2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array


class MultiLayerRNN(eqx.Module):
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
        for layer, hidden_state in zip(self.layers, hidden_states):
            x_n = layer(x_n, hidden_state)
            new_h_states.append(x_n)

        stacked_h = jnp.stack(new_h_states, axis=0)
        return x_n, jnp.stack([stacked_h], axis=1)


class DefaultHumanoidActor(eqx.Module):
    """Actor for the walking task."""

    multi_layer_rnn: MultiLayerRNN
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
        self.multi_layer_rnn = MultiLayerRNN(
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

        # Concatenate observations with commands.
        x_n = jnp.concatenate([obs_n, lin_vel_cmd_n], axis=-1)

        # Process through RNN cell
        last_h, new_hidden_states = self.multi_layer_rnn(x_n, hidden_states)
        out_n = self.projector(last_h)

        # Split into mean and std.
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
class HumanoidStandingGRUTaskConfig(HumanoidWalkingTaskConfig):
    pass


Config = TypeVar("Config", bound=HumanoidStandingGRUTaskConfig)


class HumanoidStandingGRUTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidModel:
        return DefaultHumanoidModel(key)

    def get_initial_carry(self, rng: PRNGKeyArray) -> Array:
        # Initialize the hidden state for RNN
        return jnp.zeros((DEPTH, HIDDEN_SIZE))

    def _run_actor(
        self,
        model: DefaultHumanoidModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Normal, Array]:
        dh_joint_pos_n = observations.get("dhjoint_position_observation", jnp.zeros((0,)))
        dh_joint_vel_n = observations.get("dhjoint_velocity_observation", jnp.zeros((0,))) / 50.0
        com_inertia_n = observations.get("center_of_mass_inertia_observation", jnp.zeros((0,)))
        com_vel_n = observations.get("center_of_mass_velocity_observation", jnp.zeros((0,))) / 50.0
        act_frc_obs_n = observations["actuator_force_observation"] / 100.0
        lin_vel_cmd_n = commands["linear_velocity_command"]

        return model.actor(
            dh_joint_pos_n,
            dh_joint_vel_n,
            com_inertia_n,
            com_vel_n,
            act_frc_obs_n,
            lin_vel_cmd_n,
            carry,
        )

    def _run_critic(
        self,
        model: DefaultHumanoidModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
    ) -> Array:
        dh_joint_pos_n = observations.get("dhjoint_position_observation", jnp.zeros((0,)))
        dh_joint_vel_n = observations.get("dhjoint_velocity_observation", jnp.zeros((0,))) / 50.0
        com_inertia_n = observations.get("center_of_mass_inertia_observation", jnp.zeros((0,)))
        com_vel_n = observations.get("center_of_mass_velocity_observation", jnp.zeros((0,))) / 50.0
        act_frc_obs_n = observations["actuator_force_observation"] / 100.0
        lin_vel_cmd_n = commands["linear_velocity_command"]

        # Concatenate all observations
        obs_n = jnp.concatenate([dh_joint_pos_n, dh_joint_vel_n, com_inertia_n, com_vel_n, act_frc_obs_n])
        return model.critic(obs_n, lin_vel_cmd_n)

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.PushEvent(probability=0.1),
        ]

    def get_on_policy_log_probs(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        if not isinstance(trajectories.aux_outputs, AuxOutputs):
            raise ValueError("No aux outputs found in trajectories")
        return trajectories.aux_outputs.log_probs

    def get_on_policy_values(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        if not isinstance(trajectories.aux_outputs, AuxOutputs):
            raise ValueError("No aux outputs found in trajectories")
        return trajectories.aux_outputs.values

    def get_log_probs(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        def scan_fn(
            carry: Array,
            inputs: ksim.Trajectory,
        ) -> tuple[Array, tuple[Array, Array]]:
            action_dist_n, carry = self._run_actor(model, inputs.obs, inputs.command, carry)
            log_probs_n = action_dist_n.log_prob(inputs.action / model.actor.mean_scale)
            entropy_n = action_dist_n.entropy()
            return carry, (log_probs_n, entropy_n)

        initial_hidden_states = self.get_initial_carry(rng)
        _, (log_probs_tn, entropy_tn) = jax.lax.scan(scan_fn, initial_hidden_states, trajectories)

        return log_probs_tn, entropy_tn

    def get_values(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> Array:
        # Vectorize over both batch and time dimensions.
        par_fn = jax.vmap(self._run_critic, in_axes=(None, 0, 0))
        values_bt1 = par_fn(model, trajectories.obs, trajectories.command)

        # Remove the last dimension.
        return values_bt1.squeeze(-1)

    def sample_action(
        self,
        model: DefaultHumanoidModel,
        carry: Array,
        physics_model: ksim.PhysicsModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array, AuxOutputs]:
        action_dist_n, next_carry = self._run_actor(model, observations, commands, carry)
        action_n = action_dist_n.sample(seed=rng)
        action_log_prob_n = action_dist_n.log_prob(action_n)

        critic_n = self._run_critic(model, observations, commands)
        value_n = critic_n.squeeze(-1)
        return action_n, next_carry, AuxOutputs(log_probs=action_log_prob_n, values=value_n)


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.default_humanoid.standing_gru
    # To visualize the environment, use the following command:
    #   python -m examples.default_humanoid.standing_gru run_environment=True
    HumanoidStandingGRUTask.launch(
        HumanoidStandingGRUTaskConfig(
            num_envs=2048,
            num_batches=64,
            num_passes=8,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=10.0,  # This needs to be shorter because of memory constraints.
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
