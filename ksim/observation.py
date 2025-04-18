"""Defines the base observation class."""

__all__ = [
    "ObservationInput",
    "Observation",
    "StatefulObservation",
    "BasePositionObservation",
    "BaseOrientationObservation",
    "BaseLinearVelocityObservation",
    "BaseAngularVelocityObservation",
    "JointPositionObservation",
    "JointVelocityObservation",
    "CenterOfMassInertiaObservation",
    "CenterOfMassVelocityObservation",
    "ActuatorForceObservation",
    "SensorObservation",
    "BaseLinearAccelerationObservation",
    "BaseAngularAccelerationObservation",
    "ProjectedGravityObservation",
    "ActuatorAccelerationObservation",
    "ContactObservation",
    "FeetContactObservation",
    "FeetPositionObservation",
    "FeetOrientationObservation",
    "TimestepObservation",
    "ActPosObservation",
    "ActVelObservation",
]

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Literal, Self

import attrs
import jax
import numpy as np
import xax
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.types import PhysicsModel, PhysicsState
from ksim.utils.mujoco import (
    geoms_colliding,
    get_body_data_idx_from_name,
    get_geom_data_idx_from_name,
    get_joint_names_in_order,
    get_qpos_data_idxs_by_name,
    get_sensor_data_idxs_by_name,
)
from ksim.vis import Marker

NoiseType = Literal["gaussian", "uniform"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ObservationInput:
    commands: xax.FrozenDict[str, Array]
    physics_state: PhysicsState
    obs_carry: PyTree


def add_noise(
    observation: Array,
    rng: PRNGKeyArray,
    noise_type: NoiseType,
    noise: float,
    curriculum_level: Array,
) -> Array:
    match noise_type:
        case "gaussian":
            return observation + jax.random.normal(rng, observation.shape) * noise * curriculum_level
        case "uniform":
            return observation + (jax.random.uniform(rng, observation.shape) * 2 - 1) * noise * curriculum_level
        case _:
            raise ValueError(f"Invalid noise type: {noise_type}")


@attrs.define(frozen=True, kw_only=True)
class Observation(ABC):
    """Base class for observations."""

    noise: float = attrs.field(default=0.0)
    noise_type: NoiseType = attrs.field(default="gaussian")

    @abstractmethod
    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        """Gets the observation from the state.

        Args:
            state: The inputs from which the obseravtion can be extracted.
            rng: A PRNGKeyArray to use for the noise
            curriculum_level: The current curriculum level, a scalar between
                zero and one.

        Returns:
            The observation
        """

    def __call__(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        obs_rng, noise_rng = jax.random.split(rng)
        raw_observation = self.observe(state, obs_rng, curriculum_level)
        return self.add_noise(raw_observation, curriculum_level, noise_rng)

    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        """Override to add noise to the observation.

        Args:
            observation: The raw observation from the state
            curriculum_level: The current curriculum level, a scalar between
                zero and one.
            rng: A PRNGKeyArray to use for the noise

        Returns:
            The observation with noise added
        """
        return jax.tree.map(lambda x: add_noise(x, rng, self.noise_type, self.noise, curriculum_level), observation)

    def get_markers(self) -> Collection[Marker]:
        return []

    def get_name(self) -> str:
        """Get the name of the observation."""
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def observation_name(self) -> str:
        return self.get_name()


@attrs.define(frozen=True, kw_only=True)
class StatefulObservation(Observation):
    """Defines an observation that uses a carry to store some continuous state."""

    @abstractmethod
    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> tuple[Array, PyTree]:
        """Gets the observation from the state.

        Args:
            state: The inputs from which the obseravtion can be extracted.
            rng: A PRNGKeyArray to use for the noise
            curriculum_level: The current curriculum level, a scalar between
                zero and one.

        Returns:
            The observation and the next carry.
        """

    @abstractmethod
    def init_carry(self, rng: PRNGKeyArray) -> PyTree:
        """Initialize the carry for the observation.

        Args:
            state: The state of the observation
            rng: A PRNGKeyArray to use for the noise
        """

    def __call__(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> tuple[Array, PyTree]:
        obs_rng, noise_rng = jax.random.split(rng)
        output = self.observe(state, obs_rng, curriculum_level)
        assert isinstance(output, tuple) and len(output) == 2, "StatefulObservation should return (obs, new_state)"
        raw_observation, next_state = output
        return self.add_noise(raw_observation, curriculum_level, noise_rng), next_state


@attrs.define(frozen=True, kw_only=True)
class BasePositionObservation(Observation):
    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        qpos = state.physics_state.data.qpos[0:3]  # (3,)
        return qpos


@attrs.define(frozen=True, kw_only=True)
class BaseOrientationObservation(Observation):
    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        qpos = state.physics_state.data.qpos[3:7]  # (4,)
        return qpos


@attrs.define(frozen=True, kw_only=True)
class BaseLinearVelocityObservation(Observation):
    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        qvel = state.physics_state.data.qvel[0:3]  # (3,)
        return qvel


@attrs.define(frozen=True, kw_only=True)
class BaseAngularVelocityObservation(Observation):
    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        qvel = state.physics_state.data.qvel[3:6]  # (3,)
        return qvel


@attrs.define(frozen=True, kw_only=True)
class JointPositionObservation(Observation):
    freejoint_first: bool = attrs.field(default=True)

    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        if self.freejoint_first:
            return state.physics_state.data.qpos[7:]  # (N,)
        else:
            return state.physics_state.data.qpos  # (N,)


@attrs.define(frozen=True, kw_only=True)
class JointVelocityObservation(Observation):
    freejoint_first: bool = attrs.field(default=True)

    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        if self.freejoint_first:
            return state.physics_state.data.qvel[6:]  # (N,)
        else:
            return state.physics_state.data.qvel  # (N,)


@attrs.define(frozen=True, kw_only=True)
class CenterOfMassInertiaObservation(Observation):
    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        # Skip the first entry (world body) and flatten
        cinert = state.physics_state.data.cinert[1:].ravel()  # Shape will be (nbody-1, 10)
        return cinert


@attrs.define(frozen=True, kw_only=True)
class CenterOfMassVelocityObservation(Observation):
    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        # Skip the first entry (world body) and flatten
        cvel = state.physics_state.data.cvel[1:].ravel()  # Shape will be (nbody-1, 6)
        return cvel


@attrs.define(frozen=True, kw_only=True)
class ActuatorForceObservation(Observation):
    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        return state.physics_state.data.actuator_force  # Shape will be (nu,)


@attrs.define(frozen=True, kw_only=True)
class SensorObservation(Observation):
    sensor_name: str = attrs.field()
    sensor_idx_range: tuple[int, int | None] = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        sensor_name: str,
        noise: float = 0.0,
        noise_type: NoiseType = "gaussian",
    ) -> Self:
        """Create a sensor observation from a physics model.

        Args:
            physics_model: MuJoCo physics model
            sensor_name: Name of sensor to observe
            noise: Amount of noise to add
            noise_type: Type of noise to add
        """
        sensor_name_to_idx_range = get_sensor_data_idxs_by_name(physics_model)
        if sensor_name not in sensor_name_to_idx_range:
            options = "\n".join(sorted(sensor_name_to_idx_range.keys()))
            raise ValueError(f"{sensor_name} not found in model. Available:\n{options}")

        return cls(
            noise=noise,
            noise_type=noise_type,
            sensor_name=sensor_name,
            sensor_idx_range=sensor_name_to_idx_range[sensor_name],
        )

    def get_name(self) -> str:
        return f"{super().get_name()}_{self.sensor_name}"

    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        start, end = self.sensor_idx_range
        sensor_data = state.physics_state.data.sensordata[start:end].ravel()
        return sensor_data


@attrs.define(frozen=True, kw_only=True)
class BaseLinearAccelerationObservation(Observation):
    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        return state.physics_state.data.qacc[0:3]


@attrs.define(frozen=True, kw_only=True)
class BaseAngularAccelerationObservation(Observation):
    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        return state.physics_state.data.qacc[3:6]


@attrs.define(frozen=True, kw_only=True)
class ProjectedGravityObservation(StatefulObservation):
    acc_noise: float = attrs.field(default=0.0)
    gyro_noise: float = attrs.field(default=0.0)
    acc_idx_range: tuple[int, int | None] = attrs.field()
    gyro_idx_range: tuple[int, int | None] = attrs.field()
    gravity: tuple[float, float, float] = attrs.field()

    # Kalman filter parameters
    process_noise: float = attrs.field(default=0.01)  # Process noise covariance
    acc_covariance: float = attrs.field(default=0.1)  # Accelerometer measurement noise covariance
    gyro_covariance: float = attrs.field(default=0.01)  # Gyroscope measurement noise covariance
    dt: float = attrs.field(default=0.002)  # Time step (default 2ms for 500Hz IMU)

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        acc_name: str,
        gyro_name: str,
        ctrl_dt: float,
        acc_noise: float = 0.0,
        gyro_noise: float = 0.0,
        process_noise: float = 0.01,
        acc_covariance: float = 0.1,
        gyro_covariance: float = 0.01,
    ) -> Self:
        """Create a projected gravity observation from a physics model.

        Args:
            physics_model: MuJoCo physics model
            acc_name: Name of accelerometer sensor
            gyro_name: Name of gyroscope sensor
            ctrl_dt: Time step for IMU updates
            acc_noise: Amount of noise to add to accelerometer
            gyro_noise: Amount of noise to add to gyroscope
            process_noise: Process noise covariance for Kalman filter
            acc_covariance: Accelerometer measurement noise covariance
            gyro_covariance: Gyroscope measurement noise covariance
            dt: Time step for IMU updates
        """
        sensor_name_to_idx_range = get_sensor_data_idxs_by_name(physics_model)
        for sensor_name in [acc_name, gyro_name]:
            if sensor_name not in sensor_name_to_idx_range:
                options = "\n".join(sorted(sensor_name_to_idx_range.keys()))
                raise ValueError(f"{sensor_name} not found in model. Available:\n{options}")

        # Gets the gravity vector from the physics model.
        gx, gy, gz = np.array(physics_model.opt.gravity).flatten().tolist()

        return cls(
            acc_noise=acc_noise,
            gyro_noise=gyro_noise,
            acc_idx_range=sensor_name_to_idx_range[acc_name],
            gyro_idx_range=sensor_name_to_idx_range[gyro_name],
            gravity=(gx, gy, gz),
            process_noise=process_noise,
            acc_covariance=acc_covariance,
            gyro_covariance=gyro_covariance,
            dt=ctrl_dt,
        )

    def init_carry(self, rng: PRNGKeyArray) -> tuple[Array, Array]:
        """Initialize the Kalman filter state.

        Returns:
            A 3x1 state vector (projected gravity in body frame), and a
            3x3 error covariance matrix
        """
        x = jnp.array(self.gravity)
        P = jnp.eye(3) * 10.0
        return (x, P)

    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        """Update the Kalman filter with new IMU measurements and return projected gravity.

        The Kalman filter estimates the gravity vector in the body frame using:
        - State: Projected gravity vector (3x1)
        - Process model: Gravity vector rotates with body
        - Measurement model: Accelerometer measures gravity + acceleration
        """
        acc_start, acc_end = self.acc_idx_range
        gyro_start, gyro_end = self.gyro_idx_range
        acc_data = state.physics_state.data.sensordata[acc_start:acc_end].ravel()
        gyro_data = state.physics_state.data.sensordata[gyro_start:gyro_end].ravel()

        # Add noise to measurements
        acc_rng, gyro_rng = jax.random.split(rng)
        acc_data = add_noise(acc_data, acc_rng, "gaussian", self.acc_noise, curriculum_level)
        gyro_data = add_noise(gyro_data, gyro_rng, "gaussian", self.gyro_noise, curriculum_level)

        # Get current Kalman filter state
        x, P = state.obs_carry

        # Process model: Gravity vector rotates with body
        # F = I + dt * [w]x where [w]x is the skew-symmetric matrix of angular velocity
        w = gyro_data
        wx = jnp.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        F = jnp.eye(3) + self.dt * wx

        # Process noise covariance
        Q = jnp.eye(3) * self.process_noise

        # Measurement model: Accelerometer measures gravity + acceleration
        # For simplicity, assume acceleration is zero-mean noise
        H = jnp.eye(3)
        R = jnp.eye(3) * self.acc_covariance

        # Kalman filter prediction step
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        # Kalman filter update step
        y = acc_data - x_pred  # Innovation
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ jnp.linalg.inv(S)  # Kalman gain
        x_new = x_pred + K @ y
        P_new = (jnp.eye(3) - K @ H) @ P_pred

        # Normalize the gravity vector
        x_new = x_new / jnp.linalg.norm(x_new)

        return x_new, (x_new, P_new)


@attrs.define(frozen=True, kw_only=True)
class ActuatorAccelerationObservation(Observation):
    freejoint_first: bool = attrs.field(default=True)

    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        if self.freejoint_first:
            return state.physics_state.data.qacc[6:]
        else:
            return state.physics_state.data.qacc


@attrs.define(frozen=True, kw_only=True)
class ContactObservation(Observation):
    geom_idxs: tuple[int, ...] = attrs.field()
    contact_group: str | None = attrs.field(default=None)

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        geom_names: str | Collection[str],
        contact_group: str | None = None,
        noise: float = 0.0,
    ) -> Self:
        """Create a sensor observation from a physics model."""
        if isinstance(geom_names, str):
            geom_names = [geom_names]
        geom_idxs = [get_geom_data_idx_from_name(physics_model, name) for name in geom_names]
        return cls(
            geom_idxs=tuple(geom_idxs),
            noise=noise,
            contact_group=contact_group,
        )

    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        geom_idxs = jnp.array(self.geom_idxs)
        contact = geoms_colliding(state.physics_state.data, geom_idxs, geom_idxs).any(axis=-1)
        return contact

    def get_name(self) -> str:
        if self.contact_group is not None:
            return f"{super().get_name()}_{self.contact_group}"
        else:
            return super().get_name()


@attrs.define(frozen=True, kw_only=True)
class FeetContactObservation(Observation):
    foot_left: tuple[int, ...] = attrs.field()
    foot_right: tuple[int, ...] = attrs.field()
    floor_geom: tuple[int, ...] = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        foot_left_geom_names: str | Collection[str],
        foot_right_geom_names: str | Collection[str],
        floor_geom_names: str | Collection[str],
        noise: float = 0.0,
    ) -> Self:
        """Create a sensor observation from a physics model."""
        if isinstance(foot_left_geom_names, str):
            foot_left_geom_names = [foot_left_geom_names]
        if isinstance(foot_right_geom_names, str):
            foot_right_geom_names = [foot_right_geom_names]
        if isinstance(floor_geom_names, str):
            floor_geom_names = [floor_geom_names]

        foot_left_idxs = [get_geom_data_idx_from_name(physics_model, name) for name in foot_left_geom_names]
        foot_right_idxs = [get_geom_data_idx_from_name(physics_model, name) for name in foot_right_geom_names]
        floor_geom_idxs = [get_geom_data_idx_from_name(physics_model, name) for name in floor_geom_names]
        return cls(
            foot_left=tuple(foot_left_idxs),
            foot_right=tuple(foot_right_idxs),
            floor_geom=tuple(floor_geom_idxs),
            noise=noise,
        )

    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        foot_left = jnp.array(self.foot_left)
        foot_right = jnp.array(self.foot_right)
        floor = jnp.array(self.floor_geom)
        contact_1 = geoms_colliding(state.physics_state.data, foot_left, floor).any(axis=-1)
        contact_2 = geoms_colliding(state.physics_state.data, foot_right, floor).any(axis=-1)
        return jnp.stack([contact_1, contact_2], axis=-1)


@attrs.define(frozen=True, kw_only=True)
class FeetPositionObservation(Observation):
    foot_left: int = attrs.field()
    foot_right: int = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        foot_left_body_name: str,
        foot_right_body_name: str,
        noise: float = 0.0,
    ) -> Self:
        foot_left_idx = get_body_data_idx_from_name(physics_model, foot_left_body_name)
        foot_right_idx = get_body_data_idx_from_name(physics_model, foot_right_body_name)
        return cls(
            foot_left=foot_left_idx,
            foot_right=foot_right_idx,
            noise=noise,
        )

    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        foot_left_pos = state.physics_state.data.xpos[self.foot_left]
        foot_right_pos = state.physics_state.data.xpos[self.foot_right]
        return jnp.stack([foot_left_pos, foot_right_pos], axis=-2)


@attrs.define(frozen=True, kw_only=True)
class FeetOrientationObservation(Observation):
    foot_left: int = attrs.field()
    foot_right: int = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        foot_left_body_name: str,
        foot_right_body_name: str,
        noise: float = 0.0,
    ) -> Self:
        foot_left_idx = get_body_data_idx_from_name(physics_model, foot_left_body_name)
        foot_right_idx = get_body_data_idx_from_name(physics_model, foot_right_body_name)
        return cls(
            foot_left=foot_left_idx,
            foot_right=foot_right_idx,
            noise=noise,
        )

    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        foot_left_quat = state.physics_state.data.xquat[self.foot_left]
        foot_right_quat = state.physics_state.data.xquat[self.foot_right]
        return jnp.stack([foot_left_quat, foot_right_quat], axis=-2)


@attrs.define(frozen=True, kw_only=True)
class TimestepObservation(Observation):
    """Returns the current timestep in the episode."""

    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        time = state.physics_state.data.time
        if not isinstance(time, Array):
            time = jnp.array(time)
        return time.reshape(1)


@attrs.define(frozen=True)
class ActPosObservation(Observation):
    """Observation that returns a specific joint's action and position.

    This observation is for debugging purposes, to check how well a given joint is following
    the corresponding action. It is not intended to be passed to a model or used for training.
    """

    noise: float = attrs.field(default=0.0)
    joint_name: str = attrs.field(default=None)
    joint_idx: int = attrs.field(default=0)

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        joint_name: str | None = None,
        action_index: int | None = None,
        noise: float = 0.0,
    ) -> Self:
        """Create an observation for a specific joint's action and position.

        At least one of joint_name or action_index must be provided.
        The other will be inferred from the physics model.
        """
        if joint_name is None and action_index is None:
            raise ValueError("At least one of joint_name or action_index must be provided")

        qpos_mappings = get_qpos_data_idxs_by_name(physics_model)
        ordered_joints = get_joint_names_in_order(physics_model)
        ordered_joints = [name for name in ordered_joints if "free" not in name.lower()]

        # Get both joint_idx and joint_name
        joint_idx = None
        if action_index is not None:
            joint_idx = action_index
            joint_name = ordered_joints[joint_idx]
        else:
            if joint_name not in qpos_mappings:
                available_joints = list(qpos_mappings.keys())
                raise ValueError(f"Joint name '{joint_name}' not found. Available joints: {available_joints}")
            start, _ = qpos_mappings[joint_name]
            joint_idx = start - 7

        return cls(
            joint_name=joint_name,
            joint_idx=joint_idx,
            noise=noise,
        )

    def get_name(self) -> str:
        """Get the name of the observation with joint details."""
        base_name = super().get_name()
        return f"{base_name}_{self.joint_name}"

    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        action_val = state.physics_state.most_recent_action[self.joint_idx]
        joint_pos = state.physics_state.data.qpos[7 + self.joint_idx]
        return jnp.array([action_val, joint_pos])


@attrs.define(frozen=True)
class ActVelObservation(Observation):
    """Observation that returns a specific joint's velocity action and actual velocity.

    This observation is for debugging purposes, to check how well a given joint's velocity
    is following the corresponding velocity action. It is not intended to be passed to a model
    or used for training.
    """

    noise: float = attrs.field(default=0.0)
    joint_name: str = attrs.field(default=None)
    joint_idx: int = attrs.field(default=0)
    action_idx: int = attrs.field(default=0)

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        joint_name: str | None = None,
        action_index: int | None = None,
        noise: float = 0.0,
    ) -> Self:
        """Create an observation for a specific joint's velocity action and actual velocity.

        At least one of joint_name or action_index must be provided.
        The other will be inferred from the physics model.
        """
        if joint_name is None and action_index is None:
            raise ValueError("At least one of joint_name or action_index must be provided")

        qpos_mappings = get_qpos_data_idxs_by_name(physics_model)
        ordered_joints = get_joint_names_in_order(physics_model)
        ordered_joints = [name for name in ordered_joints if "free" not in name.lower()]

        # Get both joint_idx and joint_name
        joint_idx = None
        if action_index is not None:
            # For velocity actions, the joint index is half of the action index
            # since the action space is split into positions and velocities
            joint_idx = action_index % len(ordered_joints)
            joint_name = ordered_joints[joint_idx]
        else:
            if joint_name not in qpos_mappings:
                available_joints = list(qpos_mappings.keys())
                raise ValueError(f"Joint name '{joint_name}' not found. Available joints: {available_joints}")
            start, _ = qpos_mappings[joint_name]
            joint_idx = start - 7

        # For velocity actions, the action index is in the second half of the action space
        action_idx = joint_idx + len(ordered_joints)

        return cls(
            joint_name=joint_name,
            joint_idx=joint_idx,
            action_idx=action_idx,
            noise=noise,
        )

    def get_name(self) -> str:
        """Get the name of the observation with joint details."""
        base_name = super().get_name()
        return f"{base_name}_{self.joint_name}"

    def observe(self, state: ObservationInput, rng: PRNGKeyArray, curriculum_level: Array) -> Array:
        action_val = state.physics_state.most_recent_action[self.action_idx]
        joint_vel = state.physics_state.data.qvel[6 + self.joint_idx]
        return jnp.array([action_val, joint_vel])
