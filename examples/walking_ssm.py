# mypy: disable-error-code="override"
"""Defines simple task for training a walking policy for the default humanoid using an SSM actor."""

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from abc import ABC, abstractmethod
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
    NaiveForwardReward,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxOutputs:
    log_probs: Array
    values: Array
    actor_carry: Array
    critic_carry: Array


def glorot(key: PRNGKeyArray, shape: tuple[int, ...]) -> Array:
    return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0) * jnp.sqrt(2 / sum(shape))


class BaseSSMBlock(eqx.Module, ABC):
    @abstractmethod
    def forward(self, h: Array, x: Array) -> Array: ...

    @abstractmethod
    def forward_sequence(self, x_seq: Array, reset: Array) -> Array: ...

    @abstractmethod
    def get_a_mat(self, x: Array) -> Array: ...

    @abstractmethod
    def get_b_mat(self, x: Array) -> Array: ...


class SSMBlock(BaseSSMBlock):
    a_mat: Array
    b_mat: Array

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray) -> None:
        key_a, key_b = jax.random.split(key)
        self.a_mat = glorot(key_a, (hidden_size, hidden_size))
        self.b_mat = glorot(key_b, (hidden_size, hidden_size))

    def get_a_mat(self, x: Array) -> Array:
        return self.a_mat

    def get_b_mat(self, x: Array) -> Array:
        return self.b_mat

    def forward(self, h: Array, x: Array) -> Array:
        a_mat = self.get_a_mat(x)
        b_mat = self.get_b_mat(x)
        h = a_mat @ h + b_mat.T @ x
        return h


class DiagSSMBlock(BaseSSMBlock):
    a_diag: Array
    b_mat: Array

    def __init__(self, hidden_size: int, *, key: PRNGKeyArray) -> None:
        keys = jax.random.split(key, 2)
        self.a_diag = glorot(keys[0], (hidden_size,))
        self.b_mat = glorot(keys[1], (hidden_size, hidden_size))

    def get_a_mat(self, x: Array) -> Array:
        return self.a_diag

    def get_b_mat(self, x: Array) -> Array:
        return self.b_mat

    def forward(self, h: Array, x: Array) -> Array:
        """Performs a single forward pass."""
        a_diag = self.get_a_mat(x)
        b_mat = self.get_b_mat(x)
        h = a_diag * h + b_mat.T @ x
        return h


class DiscreteDiagSSMBlock(DiagSSMBlock):
    delta: Array

    def __init__(
        self,
        hidden_size: int,
        *,
        key: PRNGKeyArray,
        init_delta: float = 1.0,
        init_scale: float = 10.0,
    ) -> None:
        super().__init__(hidden_size, key=key)
        self.delta = jnp.array(init_delta)
        self.a_diag = jax.random.uniform(key, (hidden_size,), minval=-1.0, maxval=0.0) * init_scale

    def get_a_mat(self, x: Array) -> Array:
        """Discretize the diagonal matrix using zero-order hold."""
        a_diag_discrete = jnp.exp(self.a_diag * self.delta)
        return a_diag_discrete

    def get_b_mat(self, x: Array) -> Array:
        """Discretize the input matrix using zero-order hold."""
        delta_a_diag = self.a_diag * self.delta
        exp_a_diag = jnp.exp(delta_a_diag)
        delta_a_inv = 1 / delta_a_diag
        delta_b_mat = self.delta * self.b_mat

        b_discrete = delta_a_inv * (exp_a_diag - 1) * delta_b_mat
        return b_discrete


class DPLRSSMBlock(BaseSSMBlock):
    a_diag: Array
    p_vec: Array
    q_vec: Array
    b_mat: Array
    delta: Array

    def __init__(
        self,
        hidden_size: int,
        *,
        key: PRNGKeyArray,
        init_delta: float = 0.1,
        init_scale: float = 10.0,
        rank: int = 1,
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.delta = jnp.array(init_delta)
        self.a_diag = jax.random.uniform(k1, (hidden_size,), minval=-1.0, maxval=0.0) * init_scale
        self.p_vec = glorot(k2, (hidden_size, rank))
        self.q_vec = glorot(k3, (hidden_size, rank))
        self.b_mat = glorot(k4, (hidden_size, hidden_size))

    def get_a_mat(self, x: Array) -> Array:
        """Construct discretized A matrix: diag(a_diag) + P Q^T, exponentiated."""
        A = jnp.diag(self.a_diag) + self.p_vec @ self.q_vec.T
        A_disc = jax.scipy.linalg.expm(self.delta * A)
        return A_disc

    def get_b_mat(self, x: Array) -> Array:
        """Discretize B using: ∫ exp(A τ) dτ B ≈ A^{-1}(exp(A Δ) - I) B"""
        A = jnp.diag(self.a_diag) + self.p_vec @ self.q_vec.T
        expA = jax.scipy.linalg.expm(self.delta * A)
        A_inv = jnp.linalg.pinv(A)
        B_disc = A_inv @ (expA - jnp.eye(A.shape[0])) @ self.b_mat
        return B_disc

    def forward(self, h: Array, x: Array) -> Array:
        A = self.get_a_mat(x)
        B = self.get_b_mat(x)
        return A @ h + B @ x


class SSM(eqx.Module):
    input_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    blocks: list[BaseSSMBlock]
    num_layers: int = eqx.static_field()
    hidden_size: int = eqx.static_field()
    skip_connections: bool = eqx.static_field()

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        block_type: Literal["diagonal", "full_rank"] = "diagonal",
        skip_connections: bool = False,
        discretize: bool = False,
        *,
        key: PRNGKeyArray,
    ) -> None:
        input_key, output_key, block_key = jax.random.split(key, 3)
        self.input_proj = eqx.nn.Linear(input_size, hidden_size, key=input_key)
        self.output_proj = eqx.nn.Linear(hidden_size, output_size, key=output_key)
        block_keys = jax.random.split(block_key, num_layers)

        def get_block(key: PRNGKeyArray) -> BaseSSMBlock:
            """Returns a block of the given type."""
            match block_type:
                case "diagonal":
                    return (
                        DiscreteDiagSSMBlock(hidden_size, key=key, init_delta=0.1)
                        if discretize
                        else DiagSSMBlock(hidden_size, key=key)
                    )
                case "full_rank":
                    if discretize:
                        raise ValueError("Full rank blocks do not support discretization due to instability.")
                    return SSMBlock(hidden_size, key=key)
                case _:
                    raise ValueError(f"Unknown block type: {block_type}")

        self.blocks = [get_block(block_keys[i]) for i in range(num_layers)]
        self.skip_connections = skip_connections
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def __call__(self, hs: Array, x: Array) -> tuple[Array, Array]:
        """Performs a single forward pass."""
        x = self.input_proj(x)
        new_hs = []
        for i, block in enumerate(self.blocks):
            h = block.forward(hs[i], x)
            new_hs.append(h)
            xh = jax.nn.gelu(h)
            x = xh + x if self.skip_connections else xh
        y = self.output_proj(x)
        new_hs = jnp.stack(new_hs, axis=0)
        return new_hs, y


class DefaultHumanoidSSMActor(eqx.Module):
    """SSM-based actor for the walking task."""

    ssm: SSM
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()
    min_std: float = eqx.static_field()
    max_std: float = eqx.static_field()
    var_scale: float = eqx.static_field()

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
        block_type: Literal["diagonal", "full_rank"] = "diagonal",
        discretize: bool = False,
    ) -> None:
        # Project input to hidden size
        self.ssm = SSM(
            input_size=num_inputs,
            output_size=num_outputs * 2,
            hidden_size=hidden_size,
            num_layers=depth,
            block_type=block_type,
            discretize=discretize,
            key=key,
        )

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.min_std = min_std
        self.max_std = max_std
        self.var_scale = var_scale

    def forward(self, x: Array, carry: Array) -> tuple[distrax.Distribution, Array]:
        new_hs, out_n = self.ssm(carry, x)
        # Converts the output to a distribution.
        mean_n = out_n[..., : self.num_outputs]
        std_n = out_n[..., self.num_outputs :]

        # Softplus and clip to ensure positive standard deviations.
        std_n = jnp.clip((jax.nn.softplus(std_n) + self.min_std) * self.var_scale, max=self.max_std)

        dist_n = distrax.Normal(mean_n, std_n)
        return dist_n, new_hs


class DefaultHumanoidSSMCritic(eqx.Module):
    """SSM-based critic for the walking task."""

    ssm: SSM

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
        block_type: Literal["diagonal", "full_rank"] = "diagonal",
        discretize: bool = False,
    ) -> None:
        num_outputs = 1

        # Create SSM layer
        self.ssm = SSM(
            input_size=num_inputs,
            output_size=num_outputs,
            hidden_size=hidden_size,
            num_layers=depth,
            block_type=block_type,
            discretize=discretize,
            key=key,
        )

    def forward(self, x: Array, carry: Array) -> tuple[Array, Array]:
        new_hs, out_n = self.ssm(carry, x)
        return out_n, new_hs


class DefaultHumanoidSSMModel(eqx.Module):
    actor: DefaultHumanoidSSMActor
    critic: DefaultHumanoidSSMCritic

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
        block_type: Literal["diagonal", "full_rank"] = "diagonal",
        discretize: bool = False,
    ) -> None:
        self.actor = DefaultHumanoidSSMActor(
            key,
            num_inputs=num_inputs,
            num_outputs=num_joints,
            min_std=min_std,
            max_std=max_std,
            var_scale=0.5,
            hidden_size=hidden_size,
            depth=depth,
            block_type=block_type,
            discretize=discretize,
        )
        self.critic = DefaultHumanoidSSMCritic(
            key,
            num_inputs=num_inputs,
            hidden_size=hidden_size,
            depth=depth,
            block_type=block_type,
            discretize=discretize,
        )


@dataclass
class HumanoidWalkingSSMTaskConfig(HumanoidWalkingTaskConfig):
    block_type: Literal["diagonal", "full_rank"] = xax.field(
        value="diagonal",
        help="The type of SSM block to use.",
    )
    discretize: bool = xax.field(
        value=False,
        help="Whether to discretize the SSM blocks.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingSSMTaskConfig)


class HumanoidWalkingSSMTask(HumanoidWalkingTask[Config], Generic[Config]):
    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidSSMModel:
        return DefaultHumanoidSSMModel(
            key,
            num_inputs=NUM_INPUTS,
            num_joints=NUM_JOINTS,
            min_std=0.01,
            max_std=1.0,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            block_type=self.config.block_type,
            discretize=self.config.discretize,
        )

    def run_actor(
        self,
        model: DefaultHumanoidSSMActor,
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
        joystick_cmd_1 = commands["joystick_command"]
        joystick_cmd_ohe_6 = jax.nn.one_hot(joystick_cmd_1, num_classes=6).squeeze(-2)

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3 / 50.0,  # 3
                imu_gyro_3 / 3.0,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
                joystick_cmd_ohe_6,  # 6
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def run_critic(
        self,
        model: DefaultHumanoidSSMCritic,
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
        joystick_cmd_1 = commands["joystick_command"]
        joystick_cmd_ohe_6 = jax.nn.one_hot(joystick_cmd_1, num_classes=6).squeeze(-2)

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                imu_acc_3 / 50.0,  # 3
                imu_gyro_3 / 3.0,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
                joystick_cmd_ohe_6,  # 6
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def get_ppo_variables(
        self,
        model: DefaultHumanoidSSMModel,
        trajectory: ksim.Trajectory,
        model_carry: tuple[Array, Array],
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, tuple[Array, Array]]:
        def scan_fn(
            actor_critic_carry: tuple[Array, Array], transition: ksim.Trajectory
        ) -> tuple[tuple[Array, Array], ksim.PPOVariables]:
            actor_carry, critic_carry = actor_critic_carry
            actor_dist, next_actor_carry = self.run_actor(
                model=model.actor,
                observations=transition.obs,
                commands=transition.command,
                carry=actor_carry,
            )
            log_probs = actor_dist.log_prob(transition.action)
            assert isinstance(log_probs, Array)
            value, next_critic_carry = self.run_critic(
                model=model.critic,
                observations=transition.obs,
                commands=transition.command,
                carry=critic_carry,
            )

            transition_ppo_variables = ksim.PPOVariables(
                log_probs=log_probs,
                values=value.squeeze(-1),
            )

            initial_carry = self.get_initial_model_carry(rng)
            next_carry = jax.tree.map(
                lambda x, y: jnp.where(transition.done, x, y), initial_carry, (next_actor_carry, next_critic_carry)
            )

            return next_carry, transition_ppo_variables

        next_model_carry, ppo_variables = jax.lax.scan(scan_fn, model_carry, trajectory)

        return ppo_variables, next_model_carry

    def get_initial_model_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        return (
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
            jnp.zeros(shape=(self.config.depth, self.config.hidden_size)),
        )

    def sample_action(
        self,
        model: DefaultHumanoidSSMModel,
        model_carry: tuple[Array, Array],
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
        )

        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)

        return ksim.Action(
            action=action_j,
            carry=(actor_carry, critic_carry_in),
            aux_outputs=None,
        )


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.walking_SSM
    # To visualize the environment, use the following command:
    #   python -m examples.walking_SSM run_environment=True
    HumanoidWalkingSSMTask.launch(
        HumanoidWalkingSSMTaskConfig(
            block_type="full_rank",
            discretize=True,
            hidden_size=512,
            depth=2,
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=4,
            epochs_per_log_step=1,
            rollout_length_seconds=10.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
        ),
    )
