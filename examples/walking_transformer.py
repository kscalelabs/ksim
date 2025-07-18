# mypy: disable-error-code="override"
"""Defines simple task for training a walking policy for the default humanoid using an transformer actor."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

import ksim

from .walking import NUM_INPUTS, NUM_JOINTS, ZEROS, HumanoidWalkingTask, HumanoidWalkingTaskConfig


class Actor(eqx.Module):
    """RNN-based actor for the walking task."""

    input_proj: eqx.nn.Linear
    transformer: xax.TransformerStack
    output_proj: eqx.nn.Linear
    num_inputs: int = eqx.field(static=True)
    num_outputs: int = eqx.field(static=True)
    min_std: float = eqx.field(static=True)
    max_std: float = eqx.field(static=True)
    var_scale: float = eqx.field(static=True)
    num_mixtures: int = eqx.field(static=True)

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        num_outputs: int,
        min_std: float,
        max_std: float,
        var_scale: float,
        hidden_size: int,
        depth: int,
        num_mixtures: int,
    ) -> None:
        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create transformer layer
        key, tf_key = jax.random.split(key)
        self.transformer = xax.TransformerStack(
            embed_dim=hidden_size,
            num_layers=depth,
            num_heads=hidden_size // 64,
            ff_dim=hidden_size,
            causal=True,
            context_length=32,
            key=tf_key,
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs * 3 * num_mixtures,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale
        self.num_mixtures = num_mixtures

    def forward(
        self, obs_tn: Array, carry: xax.TransformerCache, add_batch_dim: bool
    ) -> tuple[distrax.Distribution, xax.TransformerCache]:
        if add_batch_dim:
            obs_tn = obs_tn[None]

        x_tn = xax.vmap(self.input_proj)(obs_tn)
        x_tn, cache = self.transformer.forward(x_tn, cache=carry)
        prediction_tn = xax.vmap(self.output_proj)(x_tn)

        # Splits the predictions into means, standard deviations, and logits.
        slice_len = NUM_JOINTS * self.num_mixtures
        mean_tnm = prediction_tn[:, :slice_len].reshape(-1, NUM_JOINTS, self.num_mixtures)
        std_tnm = prediction_tn[:, slice_len : slice_len * 2].reshape(-1, NUM_JOINTS, self.num_mixtures)
        logits_tnm = prediction_tn[:, slice_len * 2 :].reshape(-1, NUM_JOINTS, self.num_mixtures)

        # Softplus and clip to ensure positive standard deviations.
        std_tnm = jnp.clip((jax.nn.softplus(std_tnm) + self.min_std) * self.var_scale, max=self.max_std)

        # Apply bias to the means.
        mean_tnm = mean_tnm + jnp.array([v for _, v in ZEROS])[None, :, None]

        if add_batch_dim:
            mean_tnm = mean_tnm.squeeze(0)
            std_tnm = std_tnm.squeeze(0)
            logits_tnm = logits_tnm.squeeze(0)

        dist_tn = ksim.MixtureOfGaussians(means_nm=mean_tnm, stds_nm=std_tnm, logits_nm=logits_tnm)
        return dist_tn, cache


class Critic(eqx.Module):
    """RNN-based critic for the walking task."""

    input_proj: eqx.nn.Linear
    transformer: xax.TransformerStack
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        num_outputs = 1

        # Project input to hidden size
        key, input_proj_key = jax.random.split(key)
        self.input_proj = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=hidden_size,
            key=input_proj_key,
        )

        # Create RNN layer
        key, tf_key = jax.random.split(key)
        self.transformer = xax.TransformerStack(
            embed_dim=hidden_size,
            num_layers=depth,
            num_heads=hidden_size // 64,
            ff_dim=hidden_size,
            causal=True,
            context_length=32,
            key=tf_key,
        )

        # Project to output
        self.output_proj = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=num_outputs,
            key=key,
        )

    def forward(
        self,
        obs_tn: Array,
        carry: xax.TransformerCache,
        add_batch_dim: bool,
    ) -> tuple[Array, xax.TransformerCache]:
        if add_batch_dim:
            obs_tn = obs_tn[None]

        x_tn = xax.vmap(self.input_proj)(obs_tn)
        x_tn, cache = self.transformer.forward(x_tn, cache=carry)
        out_tn = xax.vmap(self.output_proj)(x_tn)

        if add_batch_dim:
            out_tn = out_tn.squeeze(0)

        return out_tn, cache


class Model(eqx.Module):
    actor: Actor
    critic: Critic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        num_inputs: int,
        num_joints: int,
        hidden_size: int,
        depth: int,
        num_mixtures: int,
    ) -> None:
        self.actor = Actor(
            key,
            num_inputs=num_inputs,
            num_outputs=num_joints,
            min_std=min_std,
            max_std=max_std,
            var_scale=0.5,
            hidden_size=hidden_size,
            depth=depth,
            num_mixtures=num_mixtures,
        )
        self.critic = Critic(
            key,
            num_inputs=num_inputs,
            hidden_size=hidden_size,
            depth=depth,
        )


@dataclass
class HumanoidWalkingTransformerTaskConfig(HumanoidWalkingTaskConfig):
    pass


Config = TypeVar("Config", bound=HumanoidWalkingTransformerTaskConfig)


class HumanoidWalkingTransformerTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_model(self, key: PRNGKeyArray) -> Model:
        return Model(
            key,
            num_inputs=NUM_INPUTS,
            num_joints=NUM_JOINTS,
            min_std=0.01,
            max_std=1.0,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_mixtures=self.config.num_mixtures,
        )

    def run_actor(
        self,
        model: Actor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: xax.TransformerCache,
        add_batch_dim: bool,
    ) -> tuple[distrax.Distribution, xax.TransformerCache]:
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        # imu_acc_3 = observations["sensor_observation_imu_acc"]
        # imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]
        joystick_cmd_ohe_8 = commands["joystick_command"][..., :8]

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
                joystick_cmd_ohe_8,  # 8
            ],
            axis=-1,
        )

        dist_tn, carry = model.forward(obs_n, carry, add_batch_dim)

        return dist_tn, carry

    def run_critic(
        self,
        model: Critic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: xax.TransformerCache,
        add_batch_dim: bool,
    ) -> tuple[Array, xax.TransformerCache]:
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        # imu_acc_3 = observations["sensor_observation_imu_acc"]
        # imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]
        joystick_cmd_ohe_8 = commands["joystick_command"][..., :8]

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
                joystick_cmd_ohe_8,  # 8
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry, add_batch_dim)

    def get_ppo_variables(
        self,
        model: Model,
        trajectory: ksim.Trajectory,
        model_carry: tuple[xax.TransformerCache, xax.TransformerCache],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[xax.TransformerCache, xax.TransformerCache]]:
        actor_carry, critic_carry = model_carry

        actor_dist, next_actor_carry = self.run_actor(
            model=model.actor,
            observations=trajectory.obs,
            commands=trajectory.command,
            carry=actor_carry,
            add_batch_dim=False,
        )
        log_probs = actor_dist.log_prob(trajectory.action)
        assert isinstance(log_probs, Array)
        value, next_critic_carry = self.run_critic(
            model=model.critic,
            observations=trajectory.obs,
            commands=trajectory.command,
            carry=critic_carry,
            add_batch_dim=False,
        )

        ppo_variables = ksim.PPOVariables(
            log_probs=log_probs,
            values=value.squeeze(-1),
        )

        next_carry = (next_actor_carry, next_critic_carry)

        return ppo_variables, next_carry

    def get_initial_model_carry(
        self, model: Model, rng: PRNGKeyArray
    ) -> tuple[xax.TransformerCache, xax.TransformerCache]:
        return (
            model.actor.transformer.init_cache(dtype=jnp.float32),
            model.critic.transformer.init_cache(dtype=jnp.float32),
        )

    def sample_action(
        self,
        model: Model,
        model_carry: tuple[xax.TransformerCache, xax.TransformerCache],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = model_carry

        # Runs the actor model to get the action distribution.
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
            add_batch_dim=True,
        )

        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)

        return ksim.Action(action=action_j, carry=(actor_carry, critic_carry_in))


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.walking_transformer
    # To visualize the environment, use the following command:
    #   python -m examples.walking_transformer run_mode=view
    HumanoidWalkingTransformerTask.launch(
        HumanoidWalkingTransformerTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=2,
            epochs_per_log_step=1,
            rollout_length_seconds=8.0,
            global_grad_clip=2.0,
            learning_rate=1e-5,
            # Simulation parameters.
            dt=0.002,
            ctrl_dt=0.02,
            iterations=3,
            ls_iterations=5,
        ),
    )
