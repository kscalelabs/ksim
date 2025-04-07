# mypy: disable-error-code="override"
"""Pseudo-Inverse Kinematics task for the default humanoid."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import distrax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import JointMetadataOutput
from mujoco import mjx

import ksim
from ksim.utils.mujoco import remove_joints_except

from .walking_rnn import (
    DefaultHumanoidRNNActor,
    DefaultHumanoidRNNCritic,
    DefaultHumanoidRNNModel,
    HumanoidWalkingRNNTask,
    HumanoidWalkingRNNTaskConfig,
)

NUM_JOINTS = 3

NUM_INPUTS = 2 + NUM_JOINTS + NUM_JOINTS + NUM_JOINTS + 3 + 4


@dataclass
class HumanoidPseudoIKTaskConfig(HumanoidWalkingRNNTaskConfig):
    pass


Config = TypeVar("Config", bound=HumanoidPseudoIKTaskConfig)


class HumanoidPseudoIKTask(HumanoidWalkingRNNTask[Config], Generic[Config]):
    def get_mujoco_model(self) -> tuple[mujoco.MjModel, dict[str, JointMetadataOutput]]:
        mjcf_path = (Path(__file__).parent / "data" / "scene.mjcf").resolve().as_posix()
        mj_model_joint_removed = remove_joints_except(mjcf_path, ["shoulder1_right", "shoulder2_right", "elbow_right"])
        mj_model = mujoco.MjModel.from_xml_string(mj_model_joint_removed)

        mj_model.opt.timestep = jnp.array(self.config.dt)
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6
        mj_model.opt.disableflags = mjx.DisableBit.EULERDAMP
        mj_model.opt.solver = mjx.SolverType.CG

        return mj_model

    def get_model(self, key: PRNGKeyArray) -> DefaultHumanoidRNNModel:
        return DefaultHumanoidRNNModel(
            key,
            num_inputs=NUM_INPUTS,
            num_joints=NUM_JOINTS,
            min_std=0.0001,
            max_std=1.0,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
        )

    def run_actor(
        self,
        model: DefaultHumanoidRNNActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[distrax.Distribution, Array]:
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        xyz_target_3 = commands["cartesian_body_target_command_upper_arm_right"]
        quat_target_4 = commands["global_body_quaternion_command_hand_right"]

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                xyz_target_3,  # 3
                quat_target_4,  # 4
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def run_critic(
        self,
        model: DefaultHumanoidRNNCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        carry: Array,
    ) -> tuple[Array, Array]:
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        xyz_target_3 = commands["cartesian_body_target_command_upper_arm_right"]
        quat_target_4 = commands["global_body_quaternion_command_hand_right"]

        obs_n = jnp.concatenate(
            [
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                xyz_target_3,  # 3
                quat_target_4,  # 4
            ],
            axis=-1,
        )

        return model.forward(obs_n, carry)

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: dict[str, JointMetadataOutput] | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.MITPositionActuators(
            physics_model=physics_model,
            joint_name_to_metadata=metadata,
            freejoint_first=False,
        )

    def get_randomization(self, physics_model: ksim.PhysicsModel) -> list[ksim.Randomization]:
        return []

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return []

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        return []

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(freejoint_first=False),
            ksim.JointVelocityObservation(freejoint_first=False),
            ksim.ActuatorForceObservation(),
            ksim.TimestepObservation(),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.CartesianBodyTargetCommand.create(
                model=physics_model,
                # pivot_name="upper_arm_right",
                pivot_point=(0.0, 0.0, 0.0),
                base_name="pelvis",
                sample_sphere_radius=0.5,
                positive_x=True,
                positive_y=False,
                positive_z=False,
                switch_prob=self.config.ctrl_dt * 4,  # Will last 1/4 seconds in expectation
                vis_radius=0.05,
                vis_color=(1.0, 0.0, 0.0, 0.8),
            ),
            ksim.GlobalBodyQuaternionCommand.create(
                model=physics_model,
                base_name="hand_right",
                switch_prob=self.config.ctrl_dt * 4,  # Will last 1/4 seconds in expectation
                null_prob=0.5,
                vis_magnitude=0.5,
                vis_size=0.05,
                vis_color=(0.0, 0.0, 1.0, 0.8),
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.PositionTrackingReward.create(
                model=physics_model,
                tracked_body_name="hand_right",
                base_body_name="pelvis",
                norm="l2",
                scale=1.0,
                command_name="cartesian_body_target_command_upper_arm_right",
            ),
            ksim.QuaternionTrackingReward.create(
                model=physics_model,
                tracked_body_name="hand_right",
                base_body_name="pelvis",
                norm="l2",
                scale=0.001,
                command_name="global_body_quaternion_command_hand_right",
            ),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.FastAccelerationTermination(),
            # TODO: add for collisions
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.ConstantCurriculum(level=1.0)

    def sample_action(
        self,
        model: DefaultHumanoidRNNModel,
        carry: tuple[Array, Array],
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> ksim.Action:
        actor_carry_in, critic_carry_in = carry

        # Runs the actor model to get the action distribution.
        action_dist_j, actor_carry = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
            carry=actor_carry_in,
        )

        action_j = action_dist_j.sample(seed=rng)

        return ksim.Action(
            action=action_j,
            carry=(actor_carry, critic_carry_in),
            aux_outputs=None,
        )


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.pseudo_ik
    # To visualize the environment, use the following command:
    #   python -m examples.pseudo_ik run_environment=True
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m examples.pseudo_ik num_envs=8 batch_size=4
    HumanoidPseudoIKTask.launch(
        HumanoidPseudoIKTaskConfig(
            # Training parameters.
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            # Logging parameters.
            # log_full_trajectory_every_n_seconds=60,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            max_action_latency=0.0,
            min_action_latency=0.0,
            rollout_length_seconds=4.0,
            # If you experience segfaults, try disabling the markers.
            # render_markers=True,
        ),
    )
