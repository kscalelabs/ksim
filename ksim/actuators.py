"""Defines the base actuators class, along with some implementations."""

__all__ = [
    "Actuators",
    "StatefulActuators",
    "TorqueActuators",
    "PositionActuators",
    "PositionVelocityActuator",
]

import logging
from abc import ABC, abstractmethod
from typing import Literal

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.types import Metadata, PhysicsData, PhysicsModel
from ksim.utils.mujoco import get_ctrl_data_idx_by_name

logger = logging.getLogger(__name__)

NoiseType = Literal["none", "uniform", "gaussian"]


class Actuators(ABC):
    """Collection of actuators."""

    @classmethod
    def add_noise(cls, noise: float, noise_type: NoiseType, action: Array, rng: PRNGKeyArray) -> Array:
        match noise_type:
            case "none":
                return action
            case "uniform":
                return action + (jax.random.uniform(rng, action.shape) * 2 - 1) * noise
            case "gaussian":
                return action + jax.random.normal(rng, action.shape) * noise
            case _:
                raise ValueError(f"Invalid noise type: {noise_type}")

    @abstractmethod
    def get_ctrl(self, action: Array, physics_data: PhysicsData, rng: PRNGKeyArray) -> Array:
        """Get the control signal from the action vector."""

    def get_default_action(self, physics_data: PhysicsData) -> Array:
        """Get the default action for the actuators."""
        return physics_data.ctrl


class StatefulActuators(Actuators):
    @abstractmethod
    def get_stateful_ctrl(
        self,
        action: Array,
        physics_data: PhysicsData,
        actuator_state: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[Array, PyTree]:
        """Get the control signal from the action vector."""

    @abstractmethod
    def get_initial_state(self, physics_data: PhysicsData, rng: PRNGKeyArray) -> PyTree:
        """Get the initial state for the actuator."""

    def get_ctrl(self, action: Array, physics_data: PhysicsData, rng: PRNGKeyArray) -> Array:
        raise NotImplementedError("Stateful actuators should use `get_stateful_ctrl` instead.")


class TorqueActuators(Actuators):
    """Direct torque control."""

    def __init__(self, noise: float = 0.0, noise_type: NoiseType = "none") -> None:
        super().__init__()

        self.noise = noise
        self.noise_type = noise_type

    def get_ctrl(self, action: Array, physics_data: PhysicsData, rng: PRNGKeyArray) -> Array:
        """Just use the action as the torque, the simplest actuator model."""
        return self.add_noise(self.noise, self.noise_type, action, rng)


class PositionActuators(Actuators):
    """MIT Cheetah-style actuator controller operating on position."""

    def __init__(
        self,
        physics_model: PhysicsModel,
        metadata: Metadata,
        action_noise: float = 0.0,
        action_noise_type: NoiseType = "none",
        torque_noise: float = 0.0,
        torque_noise_type: NoiseType = "none",
    ) -> None:
        """Creates easily vector multipliable kps and kds."""
        ctrl_name_to_idx = get_ctrl_data_idx_by_name(physics_model)
        kps_list = [-1.0] * len(ctrl_name_to_idx)
        kds_list = [-1.0] * len(ctrl_name_to_idx)
        ctrl_clip_list = [jnp.inf] * len(ctrl_name_to_idx)

        if metadata.joint_name_to_metadata is None:
            raise ValueError("Joint metadata is required for MITPositionActuators")
        joint_name_to_metadata = metadata.joint_name_to_metadata

        for joint_name, params in joint_name_to_metadata.items():
            actuator_name = self.get_actuator_name(joint_name)
            if actuator_name not in ctrl_name_to_idx:
                if actuator_name != "root":
                    logger.warning("Joint %s has no actuator name. Skipping.", joint_name)
                continue
            actuator_idx = ctrl_name_to_idx[actuator_name]

            kp = params.kp
            kd = params.kd
            ctrl_clip = params.soft_torque_limit
            assert kp is not None and kd is not None, f"Missing kp or kd for joint {joint_name}"

            if ctrl_clip is not None:
                if ctrl_clip < 0:
                    raise ValueError(f"Soft torque limit for joint {joint_name} is negative: {ctrl_clip}")
                ctrl_clip_list[actuator_idx] = ctrl_clip

            kps_list[actuator_idx] = kp
            kds_list[actuator_idx] = kd

        if any(kp == -1 for kp in kps_list):
            raise ValueError("Some KPs are not set. Check the provided metadata.")
        if any(kd == -1 for kd in kds_list):
            raise ValueError("Some KDs are not set. Check the provided metadata.")

        self.kps = jnp.array(kps_list)
        self.kds = jnp.array(kds_list)
        self.ctrl_clip = jnp.array(ctrl_clip_list)

        self.action_noise = action_noise
        self.action_noise_type = action_noise_type
        self.torque_noise = torque_noise
        self.torque_noise_type = torque_noise_type

        if any(self.kps < 0) or any(self.kds < 0):
            raise ValueError("Some KPs or KDs are negative. Check the provided metadata.")
        if any(self.kps == 0) or any(self.kds == 0):
            logger.warning("Some KPs or KDs are 0. Check the provided metadata.")

    def get_actuator_name(self, joint_name: str) -> str:
        # This can be overridden if necessary.
        return f"{joint_name}_ctrl"

    def get_ctrl(self, action: Array, physics_data: PhysicsData, rng: PRNGKeyArray) -> Array:
        """Get the control signal from the (position) action vector."""
        pos_rng, tor_rng = jax.random.split(rng)
        current_pos = physics_data.qpos[7:]  # First 7 are always root pos.
        current_vel = physics_data.qvel[6:]  # First 6 are always root vel.
        target_velocities = jnp.zeros_like(action)
        pos_delta = self.add_noise(self.action_noise, self.action_noise_type, action - current_pos, pos_rng)
        vel_delta = target_velocities - current_vel

        ctrl = self.kps * pos_delta + self.kds * vel_delta

        return jnp.clip(
            self.add_noise(self.torque_noise, self.torque_noise_type, ctrl, tor_rng),
            -self.ctrl_clip,
            self.ctrl_clip,
        )


class PositionVelocityActuator(PositionActuators):
    """MIT Cheetah-style actuator controller operating on both position and velocity."""

    def __init__(
        self,
        physics_model: PhysicsModel,
        metadata: Metadata,
        pos_action_noise: float = 0.0,
        pos_action_noise_type: NoiseType = "none",
        vel_action_noise: float = 0.0,
        vel_action_noise_type: NoiseType = "none",
        torque_noise: float = 0.0,
        torque_noise_type: NoiseType = "none",
    ) -> None:
        super().__init__(
            physics_model=physics_model,
            metadata=metadata,
            action_noise=pos_action_noise,
            action_noise_type=pos_action_noise_type,
            torque_noise=torque_noise,
            torque_noise_type=torque_noise_type,
        )

        self.vel_action_noise = vel_action_noise
        self.vel_action_noise_type = vel_action_noise_type

    def get_ctrl(self, action: Array, physics_data: PhysicsData, rng: PRNGKeyArray) -> Array:
        """Get the control signal from the (position and velocity) action vector."""
        pos_rng, vel_rng, tor_rng = jax.random.split(rng, 3)

        current_pos = physics_data.qpos[7:]  # First 7 are always root pos.
        current_vel = physics_data.qvel[6:]  # First 6 are always root vel.

        # Adds position and velocity noise.
        target_position = action[: len(current_pos)]
        target_velocity = action[len(current_pos) :]
        chex.assert_equal_shape([current_pos, target_position, current_vel, target_velocity])
        target_position = self.add_noise(self.action_noise, self.action_noise_type, target_position, pos_rng)
        target_velocity = self.add_noise(self.vel_action_noise, self.vel_action_noise_type, target_velocity, vel_rng)

        pos_delta = target_position - current_pos
        vel_delta = target_velocity - current_vel

        ctrl = self.kps * pos_delta + self.kds * vel_delta
        return jnp.clip(
            self.add_noise(self.torque_noise, self.torque_noise_type, ctrl, tor_rng),
            -self.ctrl_clip,
            self.ctrl_clip,
        )

    def get_default_action(self, physics_data: PhysicsData) -> Array:
        """Get the default action (zeros) with the correct shape."""
        qpos_dim = len(physics_data.qpos[7:])
        return jnp.zeros(qpos_dim * 2)
