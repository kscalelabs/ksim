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
    "DelayedJointPositionObservation",
    "DelayedJointVelocityObservation",
    "CenterOfMassInertiaObservation",
    "CenterOfMassVelocityObservation",
    "ActuatorForceObservation",
    "SensorObservation",
    "BaseLinearAccelerationObservation",
    "BaseAngularAccelerationObservation",
    "ProjectedGravityObservation",
    "ActuatorAccelerationObservation",
    "ContactObservation",
    "BodyOrientationObservation",
    "SiteOrientationObservation",
    "FeetContactObservation",
    "BodyPositionObservation",
    "FeetForceObservation",
    "FeetTorqueObservation",
    "FeetOrientationObservation",
    "TimestepObservation",
    "ActPosObservation",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Iterable, Self

import attrs
import jax
import numpy as np
import xax
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.noise import Noise
from ksim.types import PhysicsModel, PhysicsState
from ksim.utils.mujoco import (
    geoms_colliding,
    get_body_data_idx_from_name,
    get_ctrl_data_idx_by_name,
    get_geom_data_idx_from_name,
    get_qpos_data_idxs_by_name,
    get_sensor_data_idxs_by_name,
    get_site_data_idx_from_name,
)
from ksim.vis import Marker


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ObservationInput:
    commands: xax.FrozenDict[str, PyTree]
    physics_state: PhysicsState
    obs_carry: PyTree


@attrs.define(frozen=True, kw_only=True)
class Observation(ABC):
    """Base class for observations."""

    noise: Noise | None = attrs.field(default=None)

    @abstractmethod
    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        """Gets the observation from the state.

        Args:
            state: The inputs from which the obseravtion can be extracted.
            curriculum_level: The current curriculum level, a scalar between
                zero and one.
            rng: A PRNGKeyArray to use for the noise

        Returns:
            The observation
        """

    def get_markers(self, name: str) -> Collection[Marker]:
        return []


@attrs.define(frozen=True, kw_only=True)
class StatefulObservation(Observation):
    """Defines an observation that uses a carry to store some continuous state."""

    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        raise NotImplementedError("StatefulObservation should use `observe_stateful` instead.")

    @abstractmethod
    def observe_stateful(
        self,
        state: ObservationInput,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[Array, PyTree]:
        """Gets the observation from the state.

        Args:
            state: The inputs from which the observation can be extracted.
            rng: A PRNGKeyArray to use for the noise
            curriculum_level: The current curriculum level, a scalar between
                zero and one.

        Returns:
            The observation and the next carry.
        """

    @abstractmethod
    def initial_carry(self, physics_state: PhysicsState, rng: PRNGKeyArray) -> PyTree:
        """Initialize the carry for the observation.

        Args:
            physics_state: The current physics state
            rng: A PRNGKeyArray to use for the noise
        """


@attrs.define(frozen=True, kw_only=True)
class BasePositionObservation(Observation):
    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        qpos = state.physics_state.data.qpos[0:3]  # (3,)
        return qpos


@attrs.define(frozen=True, kw_only=True)
class BaseOrientationObservation(Observation):
    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        qpos = state.physics_state.data.qpos[3:7]  # (4,)
        return qpos


@attrs.define(frozen=True, kw_only=True)
class BaseLinearVelocityObservation(Observation):
    in_robot_frame: bool = attrs.field(default=False)

    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        qvel = state.physics_state.data.qvel[0:3]  # (3,)
        if self.in_robot_frame:
            qvel = xax.rotate_vector_by_quat(qvel, state.physics_state.data.qpos[3:7], inverse=True)
        return qvel


@attrs.define(frozen=True, kw_only=True)
class BaseAngularVelocityObservation(Observation):
    in_robot_frame: bool = attrs.field(default=False)

    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        qvel = state.physics_state.data.qvel[3:6]  # (3,)
        if self.in_robot_frame:
            qvel = xax.rotate_vector_by_quat(qvel, state.physics_state.data.qpos[3:7], inverse=True)
        return qvel


@attrs.define(frozen=True, kw_only=True)
class JointPositionObservation(Observation):
    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.qpos[7:]  # (N,)


@attrs.define(frozen=True, kw_only=True)
class DelayedJointPositionObservation(StatefulObservation):
    delay_steps: int = attrs.field(default=1, validator=attrs.validators.ge(1))

    def initial_carry(self, physics_state: PhysicsState, rng: PRNGKeyArray) -> PyTree:
        current_qpos = physics_state.data.qpos[7:]

        # Create a buffer of shape (delay_steps, num_joints) filled with current_qpos
        return jnp.tile(current_qpos[None, :], (self.delay_steps, 1))

    def observe_stateful(
        self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray
    ) -> tuple[Array, PyTree]:
        current_qpos = state.physics_state.data.qpos[7:]

        carry_buffer = state.obs_carry

        delayed_qpos = carry_buffer[0]

        new_carry = jnp.roll(carry_buffer, -1, axis=0)
        new_carry = new_carry.at[-1].set(current_qpos)

        return delayed_qpos, new_carry


@attrs.define(frozen=True, kw_only=True)
class JointVelocityObservation(Observation):
    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.qvel[6:]  # (N,)


@attrs.define(frozen=True, kw_only=True)
class DelayedJointVelocityObservation(StatefulObservation):
    delay_steps: int = attrs.field(default=1, validator=attrs.validators.ge(1))

    def initial_carry(self, physics_state: PhysicsState, rng: PRNGKeyArray) -> PyTree:
        current_qvel = physics_state.data.qvel[6:]
        # Create a buffer of shape (delay_steps, num_joints) filled with current_qvel
        return jnp.tile(current_qvel[None, :], (self.delay_steps, 1))

    def observe_stateful(
        self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray
    ) -> tuple[Array, PyTree]:
        current_qvel = state.physics_state.data.qvel[6:]

        carry_buffer = state.obs_carry

        delayed_qvel = carry_buffer[0]

        new_carry = jnp.roll(carry_buffer, -1, axis=0)
        new_carry = new_carry.at[-1].set(current_qvel)

        return delayed_qvel, new_carry


@attrs.define(frozen=True, kw_only=True)
class CenterOfMassInertiaObservation(Observation):
    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        # Skip the first entry (world body) and flatten
        cinert = state.physics_state.data.cinert[1:].ravel()  # Shape will be (nbody-1, 10)
        return cinert


@attrs.define(frozen=True, kw_only=True)
class CenterOfMassVelocityObservation(Observation):
    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        # Skip the first entry (world body) and flatten
        cvel = state.physics_state.data.cvel[1:].ravel()  # Shape will be (nbody-1, 6)
        return cvel


@attrs.define(frozen=True, kw_only=True)
class ActuatorForceObservation(Observation):
    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
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
        noise: Noise | None = None,
    ) -> Self:
        """Create a sensor observation from a physics model.

        Args:
            physics_model: MuJoCo physics model
            sensor_name: Name of sensor to observe
            noise: The observation noise
        """
        sensor_name_to_idx_range = get_sensor_data_idxs_by_name(physics_model)
        if sensor_name not in sensor_name_to_idx_range:
            options = "\n".join(sorted(sensor_name_to_idx_range.keys()))
            raise ValueError(f"{sensor_name} not found in model. Available:\n{options}")

        return cls(
            sensor_name=sensor_name,
            sensor_idx_range=sensor_name_to_idx_range[sensor_name],
            noise=noise,
        )

    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        start, end = self.sensor_idx_range
        sensor_data = state.physics_state.data.sensordata[start:end].ravel()
        return sensor_data


@attrs.define(frozen=True, kw_only=True)
class BaseLinearAccelerationObservation(Observation):
    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.qacc[0:3]


@attrs.define(frozen=True, kw_only=True)
class BaseAngularAccelerationObservation(Observation):
    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.qacc[3:6]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ProjectedGravityCarry:
    x: Array
    lag: Array
    bias: Array


@attrs.define(frozen=True, kw_only=True)
class ProjectedGravityObservation(StatefulObservation):
    """Observes the projected gravity vector.

    This provides an approximation of reading the projected gravity vector from
    the IMU on the physical robot. The `framequat_name` should be the name of
    the framequat sensor attached to the IMU.
    """

    framequat_idx_range: tuple[int, int | None] = attrs.field()
    gravity: tuple[float, float, float] = attrs.field()
    min_lag: float = attrs.field(validator=attrs.validators.and_(attrs.validators.ge(0.0), attrs.validators.lt(1.0)))
    max_lag: float = attrs.field(validator=attrs.validators.and_(attrs.validators.ge(0.0), attrs.validators.lt(1.0)))
    bias: float = attrs.field(validator=attrs.validators.ge(0.0))

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        framequat_name: str,
        min_lag: float = 0.0,
        max_lag: float = 0.0,
        bias: float = 0.0,
        noise: Noise | None = None,
    ) -> Self:
        """Create a projected gravity observation from a physics model.

        Args:
            physics_model: MuJoCo physics model
            framequat_name: The name of the framequat sensor
            min_lag: The minimum EMA factor to use, to approximate the
                variation in the amount of smoothing of the Kalman filter.
            max_lag: The maximum EMA factor to use, to approximate the
                variation in the amount of smoothing of the Kalman filter.
            bias: The bias of the gravity vector, in radians.
            noise: The observation noise.
        """
        sensor_name_to_idx_range = get_sensor_data_idxs_by_name(physics_model)
        if framequat_name not in sensor_name_to_idx_range:
            options = "\n".join(sorted(sensor_name_to_idx_range.keys()))
            raise ValueError(f"{framequat_name} not found in model. Available:\n{options}")

        gx, gy, gz = np.array(physics_model.opt.gravity).flatten().tolist()

        return cls(
            framequat_idx_range=sensor_name_to_idx_range[framequat_name],
            gravity=(float(gx), float(gy), float(gz)),
            min_lag=min_lag,
            max_lag=max_lag,
            noise=noise,
            bias=bias,
        )

    def initial_carry(self, physics_state: PhysicsState, rng: PRNGKeyArray) -> ProjectedGravityCarry:
        lrng, brng = jax.random.split(rng)
        lag = jax.random.uniform(lrng, (1,), minval=self.min_lag, maxval=self.max_lag)
        bias = jax.random.uniform(brng, (3,), minval=-self.bias, maxval=self.bias)
        return ProjectedGravityCarry(x=jnp.zeros((3,)), lag=lag, bias=bias)

    def observe_stateful(
        self,
        state: ObservationInput,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> tuple[Array, ProjectedGravityCarry]:
        carry: ProjectedGravityCarry = state.obs_carry
        framequat_start, framequat_end = self.framequat_idx_range
        framequat_data = state.physics_state.data.sensordata[framequat_start:framequat_end].ravel()

        # Orients the gravity vector according to the quaternion.
        gravity = jnp.array(self.gravity)
        bias_quat = xax.euler_to_quat(carry.bias)
        proj_gravity = xax.rotate_vector_by_quat(gravity, framequat_data, inverse=True)
        proj_gravity = xax.rotate_vector_by_quat(proj_gravity, bias_quat)

        # Get current lagging state.
        x = carry.x * carry.lag + proj_gravity * (1 - carry.lag)

        return x, ProjectedGravityCarry(x=x, lag=carry.lag, bias=carry.bias)


@attrs.define(frozen=True, kw_only=True)
class ActuatorAccelerationObservation(Observation):
    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.qacc[6:]


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
        noise: Noise | None = None,
    ) -> Self:
        """Create a sensor observation from a physics model."""
        if isinstance(geom_names, str):
            geom_names = [geom_names]
        geom_idxs = [get_geom_data_idx_from_name(physics_model, name) for name in geom_names]
        return cls(
            geom_idxs=tuple(geom_idxs),
            contact_group=contact_group,
            noise=noise,
        )

    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        geom_idxs = jnp.array(self.geom_idxs)
        contact = geoms_colliding(state.physics_state.data, geom_idxs, geom_idxs).any(axis=-1)
        return contact


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
        noise: Noise | None = None,
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

    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        foot_left = jnp.array(self.foot_left)
        foot_right = jnp.array(self.foot_right)
        floor = jnp.array(self.floor_geom)
        contact_1 = geoms_colliding(state.physics_state.data, foot_left, floor).any(axis=-1)
        contact_2 = geoms_colliding(state.physics_state.data, foot_right, floor).any(axis=-1)
        return jnp.stack([contact_1, contact_2], axis=-1)


@attrs.define(frozen=True, kw_only=True)
class BodyPositionObservation(Observation):
    body_idxs: tuple[int, ...] = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        body_names: Iterable[str],
        noise: Noise | None = None,
    ) -> Self:
        body_idxs = tuple(get_body_data_idx_from_name(physics_model, name) for name in body_names)
        return cls(body_idxs=body_idxs, noise=noise)

    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return state.physics_state.data.xpos[self.body_idxs, :]


@attrs.define(frozen=True, kw_only=True)
class FeetForceObservation(Observation):
    foot_left: int = attrs.field()
    foot_right: int = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        foot_left_body_name: str,
        foot_right_body_name: str,
        noise: Noise | None = None,
    ) -> Self:
        foot_left_idx = get_body_data_idx_from_name(physics_model, foot_left_body_name)
        foot_right_idx = get_body_data_idx_from_name(physics_model, foot_right_body_name)
        return cls(
            foot_left=foot_left_idx,
            foot_right=foot_right_idx,
            noise=noise,
        )

    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        foot_left_pos = state.physics_state.data.cfrc_ext[self.foot_left][..., :3]
        foot_right_pos = state.physics_state.data.cfrc_ext[self.foot_right][..., :3]
        return jnp.stack([foot_left_pos, foot_right_pos], axis=-2)


@attrs.define(frozen=True, kw_only=True)
class FeetTorqueObservation(Observation):
    foot_left: int = attrs.field()
    foot_right: int = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        foot_left_body_name: str,
        foot_right_body_name: str,
        noise: Noise | None = None,
    ) -> Self:
        foot_left_idx = get_body_data_idx_from_name(physics_model, foot_left_body_name)
        foot_right_idx = get_body_data_idx_from_name(physics_model, foot_right_body_name)
        return cls(
            foot_left=foot_left_idx,
            foot_right=foot_right_idx,
            noise=noise,
        )

    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        foot_left_pos = state.physics_state.data.cfrc_ext[self.foot_left][..., 3:]
        foot_right_pos = state.physics_state.data.cfrc_ext[self.foot_right][..., 3:]
        return jnp.stack([foot_left_pos, foot_right_pos], axis=-2)


@attrs.define(frozen=True, kw_only=True)
class BodyOrientationObservation(Observation):
    body_ids: tuple[int, ...]
    name: str = attrs.field(default="body_orientation")

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        body_names: Collection[str],
        noise: Noise | None = None,
    ) -> Self:
        body_ids = tuple(get_body_data_idx_from_name(physics_model, name) for name in body_names)
        name = f"{xax.camelcase_to_snakecase(cls.__name__)}_{'_'.join(body_names)}"
        return cls(
            body_ids=body_ids,
            name=name,
            noise=noise,
        )

    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        quats = state.physics_state.data.xquat[jnp.asarray(self.body_ids)]
        return quats


@attrs.define(frozen=True, kw_only=True)
class SiteOrientationObservation(Observation):
    site_ids: tuple[int, ...]
    name: str = attrs.field(default="site_orientation")

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        site_names: tuple[str, ...],
        noise: Noise | None = None,
    ) -> Self:
        site_ids = tuple(get_site_data_idx_from_name(physics_model, name) for name in site_names)
        name = f"{xax.camelcase_to_snakecase(cls.__name__)}_{'_'.join(site_names)}"
        return cls(
            site_ids=site_ids,
            name=name,
            noise=noise,
        )

    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        rot_mats = state.physics_state.data.site_xmat[jnp.asarray(self.site_ids)].reshape(-1, 3, 3)
        rot6ds = xax.rotation_matrix_to_rotation6d(rot_mats)
        return rot6ds


@attrs.define(frozen=True, kw_only=True)
class FeetOrientationObservation(BodyOrientationObservation):
    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        body_names: Collection[str],
        noise: Noise | None = None,
    ) -> Self:  # <- same return type
        if len(body_names) != 2:
            raise ValueError("FeetOrientationObservation expects exactly two body names (left and right foot).")
        return super().create(
            physics_model=physics_model,
            body_names=body_names,
            noise=noise,
        )

    @classmethod
    def create_from_feet(
        cls,
        *,
        physics_model: PhysicsModel,
        foot_left_body_name: str,
        foot_right_body_name: str,
        noise: Noise | None = None,
    ) -> Self:
        return super().create(
            physics_model=physics_model,
            body_names=(foot_left_body_name, foot_right_body_name),
            noise=noise,
        )


@attrs.define(frozen=True, kw_only=True)
class TimestepObservation(Observation):
    """Returns the current timestep in the episode."""

    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
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

    joint_name: str = attrs.field()
    ctrl_idx: int = attrs.field()
    qpos_idx: int = attrs.field()

    @classmethod
    def create(
        cls,
        *,
        physics_model: PhysicsModel,
        joint_name: str,
    ) -> Self:
        qpos_idx, _ = get_qpos_data_idxs_by_name(physics_model)[joint_name]
        ctrl_idx = get_ctrl_data_idx_by_name(physics_model)[f"{joint_name}_ctrl"]

        return cls(
            joint_name=joint_name,
            ctrl_idx=ctrl_idx,
            qpos_idx=qpos_idx,
        )

    def observe(self, state: ObservationInput, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        action_val = state.physics_state.most_recent_action[self.ctrl_idx]
        joint_pos = state.physics_state.data.qpos[self.qpos_idx]
        return jnp.array([action_val, joint_pos])
