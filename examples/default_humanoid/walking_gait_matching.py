"""Walking default humanoid task with reference gait tracking."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import distrax
import jax
import jax.numpy as jnp
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx

import ksim
from ksim.utils import mujoco

from .reference_gait import (
    HUMANOID_MAPPING_SPEC,
    ReferenceMarker,
    generate_reference_gait,
    get_local_point_pos,
    get_reference_joint_id,
)
from .walking import (
    DefaultHumanoidModel,
    HumanoidWalkingTask,
    HumanoidWalkingTaskConfig,
    NaiveVelocityReward,
)

NUM_JOINTS = 21


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxModelOutputs:
    log_probs: Array
    values: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AuxTransitionOutputs:
    tracked_pos: dict[int, Array]


@dataclass
class HumanoidWalkingGaitMatchingTaskConfig(HumanoidWalkingTaskConfig):
    reference_gait_path: Path = xax.field(
        value=Path(__file__).parent / "data" / "reference_gait.npz",
        help="The path to the reference gait.",
    )
    gait_matching_mappings: list[ReferenceMarker] = xax.field(
        value=HUMANOID_MAPPING_SPEC,
        help="The mappings of BVH joints to Mujoco bodies",
    )
    base_id: int = xax.field(
        value=1,
        help="The Mujoco body id of the base of the humanoid",
    )


Config = TypeVar("Config", bound=HumanoidWalkingGaitMatchingTaskConfig)


class GaitMatchingReward(ksim.Reward):
    def __init__(self, reference_gait: dict[int, Array], mappings: list[ReferenceMarker]) -> None:
        self.reference_gait = reference_gait  # T, 3
        self.mappings = mappings
        self.num_frames = list(reference_gait.values())[0].shape[0]

    def __call__(self, trajectory: ksim.Trajectory) -> Array:
        assert isinstance(trajectory.aux_transition_outputs, AuxTransitionOutputs)

        # Computes MSE error between the tracked and target positions per transition.
        def compute_error(num_steps: Array, transition: ksim.Trajectory) -> tuple[Array, Array]:
            assert isinstance(transition.aux_transition_outputs, AuxTransitionOutputs)
            frame_idx = num_steps % self.num_frames
            target_pos = jax.tree.map(lambda x: x[frame_idx], self.reference_gait)  # 3
            tracked_pos = transition.aux_transition_outputs.tracked_pos  # 3
            error = jax.tree.map(lambda target, tracked: jnp.mean((target - tracked) ** 2), target_pos, tracked_pos)
            mean_error = jnp.mean(jnp.array(list(error.values())))
            next_num_steps = jax.lax.select(transition.done, 0, num_steps + 1)

            return next_num_steps, mean_error

        _, errors = jax.lax.scan(compute_error, jnp.array(0), trajectory)
        mse = jnp.mean(errors)
        jax.debug.breakpoint()
        return mse


class HumanoidWalkingGaitMatchingTask(HumanoidWalkingTask[Config], Generic[Config]):

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        with open(self.config.reference_gait_path, "rb") as f:
            reference_gait = pickle.load(f)
            reference_gait = jax.tree.map(lambda x: jnp.array(x), reference_gait)
            self.reference_gait = reference_gait
        rewards = [
            ksim.BaseHeightRangeReward(z_lower=0.8, z_upper=1.5, scale=0.5),
            ksim.LinearVelocityZPenalty(scale=-0.01),
            ksim.AngularVelocityXYPenalty(scale=-0.01),
            NaiveVelocityReward(scale=0.1),
            GaitMatchingReward(self.reference_gait, self.config.gait_matching_mappings),
        ]

        return rewards

    def _run_actor(
        self,
        model: DefaultHumanoidModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
    ) -> distrax.Normal:
        dh_joint_pos_n = observations["joint_position_observation"]
        dh_joint_vel_n = observations["joint_velocity_observation"] / 50.0
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"] / 50.0
        act_frc_obs_n = observations["actuator_force_observation"] / 100.0
        lin_vel_cmd_2 = commands["linear_velocity_step_command"]
        ang_vel_cmd_1 = commands["angular_velocity_step_command"]
        return model.actor(
            dh_joint_pos_n,
            dh_joint_vel_n,
            com_inertia_n,
            com_vel_n,
            act_frc_obs_n,
            lin_vel_cmd_2,
            ang_vel_cmd_1,
        )

    def _run_critic(
        self,
        model: DefaultHumanoidModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
    ) -> Array:
        dh_joint_pos_n = observations["joint_position_observation"]  # 26
        dh_joint_vel_n = observations["joint_velocity_observation"]  # 27
        com_inertia_n = observations["center_of_mass_inertia_observation"]  # 160
        com_vel_n = observations["center_of_mass_velocity_observation"]  # 96
        act_frc_obs_n = observations["actuator_force_observation"] / 100.0  # 21
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]  # 3
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]  # 3
        lin_vel_cmd_2 = commands["linear_velocity_step_command"]  # 2
        ang_vel_cmd_1 = commands["angular_velocity_step_command"]  # 1
        return model.critic(
            dh_joint_pos_n,
            dh_joint_vel_n,
            com_inertia_n,
            com_vel_n,
            act_frc_obs_n,
            lin_vel_obs_3,
            ang_vel_obs_3,
            lin_vel_cmd_2,
            ang_vel_cmd_1,
        )

    def get_log_probs(
        self,
        model: DefaultHumanoidModel,
        trajectories: ksim.Trajectory,
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        # Vectorize over both batch and time dimensions.
        par_fn = jax.vmap(self._run_actor, in_axes=(None, 0, 0))
        action_dist_btn = par_fn(model, trajectories.obs, trajectories.command)

        # Compute the log probabilities of the trajectory's actions according
        # to the current policy, along with the entropy of the distribution.
        action_btn = trajectories.action / model.actor.mean_scale
        log_probs_btn = action_dist_btn.log_prob(action_btn)
        entropy_btn = action_dist_btn.entropy()

        return log_probs_btn, entropy_btn

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
        carry: None,
        physics_model: ksim.PhysicsModel,
        observations: FrozenDict[str, Array],
        commands: FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, None, AuxModelOutputs]:
        action_dist_n = self._run_actor(model, observations, commands)
        action_n = action_dist_n.sample(seed=rng)
        action_log_prob_n = jnp.array(action_dist_n.log_prob(action_n))

        critic_n = self._run_critic(model, observations, commands)
        value_n = critic_n.squeeze(-1)

        return action_n, None, AuxModelOutputs(log_probs=action_log_prob_n, values=value_n)

    def _get_tracked_pos(self, mappings: list[ReferenceMarker], xpos: Array, base_id: int) -> dict[int, Array]:
        tracked_positions: dict[int, Array] = {}

        for mapping in mappings:
            body_pos = get_local_point_pos(xpos, mapping.mj_body_id, base_id)
            assert isinstance(body_pos, Array)
            tracked_positions[mapping.mj_body_id] = body_pos

        return tracked_positions

    def get_transition_aux_outputs(
        self,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        next_physics_state: ksim.PhysicsState,
        action: Array,
        terminated: Array,
    ) -> AuxTransitionOutputs:
        tracked_pos = self._get_tracked_pos(
            self.config.gait_matching_mappings, next_physics_state.data.xpos, self.config.base_id
        )
        return AuxTransitionOutputs(tracked_pos=tracked_pos)


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.default_humanoid.walking
    # To visualize the environment, use the following command:
    #   python -m examples.default_humanoid.walking run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m examples.default_humanoid.walking num_envs=8 num_batches=2
    HumanoidWalkingGaitMatchingTask.launch(
        HumanoidWalkingTaskConfig(
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=10,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=21.0,
            # PPO parameters
            gamma=0.97,
            lam=0.95,
            entropy_coef=0.001,
            learning_rate=3e-4,
            clip_param=0.3,
            max_grad_norm=1.0,
            use_mit_actuators=True,
            valid_every_n_steps=50,
        ),
    )
