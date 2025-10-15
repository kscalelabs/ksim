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

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

<<<<<<< HEAD
from ksim.noise import Noise, NoNoise
from ksim.types import Metadata, PhysicsModel
=======
from ksim.noise import Noise, NoNoise, RandomVariable
from ksim.types import Metadata, PhysicsData, PhysicsModel
>>>>>>> bb6a67e (unify actuators)
from ksim.utils.mujoco import get_ctrl_data_idx_by_name

logger = logging.getLogger(__name__)


class Actuators(ABC):
    """Collection of actuators."""

    @abstractmethod
    def get_ctrl(
        self,
        action: Array,
        qpos: Array,
        qvel: Array,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        """Get the control signal from the action vector."""


class StatefulActuators(Actuators):
    @abstractmethod
    def get_stateful_ctrl(
        self,
        action: Array,
        qpos: Array,
        qvel: Array,
        curriculum_level: Array,
        actuator_state: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[Array, PyTree]:
        """Get the control signal from the action vector."""

    @abstractmethod
    def get_initial_state(self, qpos: Array, qvel: Array, rng: PRNGKeyArray) -> PyTree:
        """Get the initial state for the actuator."""


class TorqueActuators(Actuators):
    """Direct torque control."""

    def __init__(self, noise: Noise | None = None) -> None:
        super().__init__()

        self.noise = NoNoise() if noise is None else noise

    def get_ctrl(
        self,
        action: Array,
        qpos: Array,
        qvel: Array,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        """Just use the action as the torque, the simplest actuator model."""
        return self.noise.add_noise(action, curriculum_level, rng)


class ActuatorState(TypedDict):
    action: Array
    torque: Array


class PositionActuators(StatefulActuators):
    """MIT Cheetah-style actuator controller operating on position."""

    def __init__(
        self,
        physics_model: PhysicsModel,
        metadata: Metadata,
        action_noise: Noise | None = None,
        torque_noise: Noise | None = None,
        action_scale: float = 1.0,
        action_bias: RandomVariable | None = None,
        torque_bias: RandomVariable | None = None,
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

        self.action_noise = NoNoise() if action_noise is None else action_noise
        self.torque_noise = NoNoise() if torque_noise is None else torque_noise

        self.action_scale = action_scale

        self.action_bias = action_bias
        self.torque_bias = torque_bias

        if any(self.kps < 0) or any(self.kds < 0):
            raise ValueError("Some KPs or KDs are negative. Check the provided metadata.")
        if any(self.kps == 0) or any(self.kds == 0):
            logger.warning("Some KPs or KDs are 0. Check the provided metadata.")

    def get_actuator_name(self, joint_name: str) -> str:
        # This can be overridden if necessary.
        return f"{joint_name}_ctrl"

    def get_stateful_ctrl(
        self,
        action: Array,
        qpos: Array,
        qvel: Array,
        curriculum_level: Array,
        actuator_state: ActuatorState,
        rng: PRNGKeyArray,
    ) -> tuple[Array, ActuatorState]:
        """Get the control signal from the (position) action vector with optional biases."""
        action_bias = actuator_state["action"]
        torque_bias = actuator_state["torque"]

        scaled = action * self.action_scale

        pos_rng, tor_rng = jax.random.split(rng)

        # Calling function removes root position and velocity.
        current_pos = qpos
        current_vel = qvel

        # Add position and velocity noise
        target_position = self.action_noise.add_noise(scaled, curriculum_level, pos_rng) + action_bias
        target_velocity = jnp.zeros_like(action)

        pos_delta = target_position - current_pos
        vel_delta = target_velocity - current_vel

        ctrl = self.kps * pos_delta + self.kds * vel_delta
        ctrl = self.torque_noise.add_noise(ctrl, curriculum_level, tor_rng) + torque_bias
        return jnp.clip(ctrl, -self.ctrl_clip, self.ctrl_clip), actuator_state

    def get_initial_state(self, physics_data: PhysicsData, rng: PRNGKeyArray) -> ActuatorState:
        """Get the initial state for the actuator."""
        shape = physics_data.qpos[..., 7:].shape
        action_bias_value = self.action_bias.get_random_variable(shape, rng) if self.action_bias else jnp.zeros(shape)
        torque_bias_value = self.torque_bias.get_random_variable(shape, rng) if self.torque_bias else jnp.zeros(shape)
        return {"action": action_bias_value, "torque": torque_bias_value}


class PositionVelocityActuator(PositionActuators):
    """MIT Cheetah-style actuator controller operating on both position and velocity."""

    def __init__(
        self,
        physics_model: PhysicsModel,
        metadata: Metadata,
        pos_action_noise: Noise | None = None,
        vel_action_noise: Noise | None = None,
        torque_noise: Noise | None = None,
    ) -> None:
        super().__init__(
            physics_model=physics_model,
            metadata=metadata,
            action_noise=pos_action_noise,
            torque_noise=torque_noise,
        )

        self.vel_action_noise = NoNoise() if vel_action_noise is None else vel_action_noise

    def get_ctrl(
        self,
        action: Array,
        qpos: Array,
        qvel: Array,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> Array:
        """Get the control signal from the (position and velocity) action vector."""
        pos_rng, vel_rng, tor_rng = jax.random.split(rng, 3)

        # Calling function removes root position and velocity.
        current_pos = qpos
        current_vel = qvel

        # Extract position and velocity targets
        target_position = action[: len(current_pos)]
        target_velocity = action[len(current_pos) :]
        chex.assert_equal_shape([current_pos, target_position, current_vel, target_velocity])

        # Add position and velocity noise
        target_position = self.action_noise.add_noise(target_position, curriculum_level, pos_rng)
        target_velocity = self.vel_action_noise.add_noise(target_velocity, curriculum_level, vel_rng)

        pos_delta = target_position - current_pos
        vel_delta = target_velocity - current_vel

        ctrl = self.kps * pos_delta + self.kds * vel_delta
        return jnp.clip(
            self.torque_noise.add_noise(ctrl, curriculum_level, tor_rng),
            -self.ctrl_clip,
            self.ctrl_clip,
        )
