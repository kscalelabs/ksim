"""Base Types for Environments."""

__all__ = [
    "PhysicsData",
    "PhysicsModel",
    "PhysicsState",
    "Trajectory",
    "RewardState",
    "Action",
    "Histogram",
    "Metrics",
    "LoggedTrajectory",
    "JointMetadata",
    "ActuatorMetadata",
    "Metadata",
]

from dataclasses import dataclass
from typing import Mapping, TypeAlias

import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PyTree
from kscale.web.gen.api import (
    ActuatorMetadataOutput,
    JointMetadataOutput,
    RobotURDFMetadataOutput,
)
from mujoco import mjx

PhysicsData: TypeAlias = mjx.Data | mujoco.MjData
PhysicsModel: TypeAlias = mjx.Model | mujoco.MjModel


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhysicsState:
    """Everything you need for the engine to take an action and step physics."""

    most_recent_action: Array
    data: PhysicsData
    event_states: xax.FrozenDict[str, PyTree]
    actuator_state: PyTree
    action_latency: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Trajectory:
    """Stackable structure of transitions.

    Note that `qpos`, `qvel`, `xpos`, `xquat`, `timestep`, `done` and `success`
    are the values from *after* the action has been taken, while `obs` and
    `command` are the values from *before* the action has been taken.
    """

    qpos: Array
    qvel: Array
    xpos: Array
    xquat: Array
    ctrl: Array
    obs: xax.FrozenDict[str, PyTree]
    command: xax.FrozenDict[str, PyTree]
    event_state: xax.FrozenDict[str, Array]
    action: Array
    done: Array
    success: Array
    timestep: Array
    termination_components: xax.FrozenDict[str, Array]
    aux_outputs: xax.FrozenDict[str, PyTree]

    def episode_length(self) -> Array:
        done_mask = self.done.at[..., -1].set(True)
        termination_sum = jnp.sum(jnp.where(done_mask, self.timestep, 0.0), axis=-1)
        episode_length = termination_sum / done_mask.sum(axis=-1)
        return episode_length


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RewardState:
    total: Array
    components: xax.FrozenDict[str, Array]
    carry: xax.FrozenDict[str, PyTree]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Action:
    action: Array
    carry: PyTree | None = None
    aux_outputs: xax.FrozenDict[str, PyTree] = xax.FrozenDict()


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Histogram:
    counts: Array
    limits: Array
    min: Array
    max: Array
    sum: Array
    sum_squares: Array
    mean: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Metrics:
    train: Mapping[str, Array | Histogram]
    reward: Mapping[str, Array | Histogram]
    termination: Mapping[str, Array | Histogram]
    curriculum_level: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LoggedTrajectory:
    trajectory: Trajectory
    rewards: RewardState
    metrics: xax.FrozenDict[str, Array]


def _parse_float(value: str | None) -> float | None:
    return None if value is None else float(value)


@jax.tree_util.register_dataclass
@dataclass
class JointMetadata:
    kp: float | None = None
    kd: float | None = None
    armature: float | None = None
    friction: float | None = None
    actuator_type: str | None = None
    soft_torque_limit: float | None = None

    @classmethod
    def from_kscale_joint_metadata(cls, metadata: JointMetadataOutput) -> "JointMetadata":
        return cls(
            kp=_parse_float(metadata.kp),
            kd=_parse_float(metadata.kd),
            armature=_parse_float(metadata.armature),
            friction=_parse_float(metadata.friction),
            actuator_type=metadata.actuator_type,
            soft_torque_limit=_parse_float(metadata.soft_torque_limit),
        )

    @classmethod
    def from_model(
        cls,
        model: PhysicsModel,
        kp: float | None = None,
        kd: float | None = None,
        armature: float | None = None,
        friction: float | None = None,
        actuator_type: str | None = None,
        soft_torque_limit: float | None = None,
    ) -> dict[str, "JointMetadata"]:
        def _get_name(i: int) -> str:
            return model.names[model.name_jntadr[i] :].decode("utf-8").split("\x00", 1)[0]

        return {
            _get_name(i): cls(
                kp=kp,
                kd=kd,
                armature=armature,
                friction=friction,
                actuator_type=actuator_type,
                soft_torque_limit=soft_torque_limit,
            )
            for i in range(model.njnt)
        }


@jax.tree_util.register_dataclass
@dataclass
class ActuatorMetadata:
    actuator_type: str | None = None
    sys_id: str | None = None
    max_torque: float | None = None
    armature: float | None = None
    damping: float | None = None
    frictionloss: float | None = None
    vin: float | None = None
    kt: float | None = None
    R: float | None = None
    vmax: float | None = None
    amax: float | None = None
    max_velocity: float | None = None
    max_pwm: float | None = None
    error_gain: float | None = None

    @classmethod
    def from_kscale_actuator_metadata(cls, metadata: ActuatorMetadataOutput) -> "ActuatorMetadata":
        return cls(
            actuator_type=metadata.actuator_type,
            sys_id=metadata.sysid,
            max_torque=_parse_float(metadata.max_torque),
            armature=_parse_float(metadata.armature),
            damping=_parse_float(metadata.damping),
            frictionloss=_parse_float(metadata.frictionloss),
            vin=_parse_float(metadata.vin),
            kt=_parse_float(metadata.kt),
            R=_parse_float(metadata.R),
            vmax=_parse_float(metadata.vmax),
            amax=_parse_float(metadata.amax),
            max_velocity=_parse_float(metadata.max_velocity),
            max_pwm=_parse_float(metadata.max_pwm),
            error_gain=_parse_float(metadata.error_gain),
        )

    @classmethod
    def from_model(cls, model: PhysicsModel) -> dict[str, "ActuatorMetadata"]:
        return {"motor": cls(actuator_type="motor")}


@jax.tree_util.register_dataclass
@dataclass
class Metadata:
    joint_name_to_metadata: dict[str, JointMetadata]
    actuator_type_to_metadata: dict[str, ActuatorMetadata]
    control_frequency: float | None = None

    @classmethod
    def from_kscale_metadata(cls, metadata: RobotURDFMetadataOutput) -> "Metadata":
        return cls(
            joint_name_to_metadata=(
                {}
                if metadata.joint_name_to_metadata is None
                else {
                    k: JointMetadata.from_kscale_joint_metadata(v) for k, v in metadata.joint_name_to_metadata.items()
                }
            ),
            actuator_type_to_metadata=(
                {}
                if metadata.actuator_type_to_metadata is None
                else {
                    k: ActuatorMetadata.from_kscale_actuator_metadata(v)
                    for k, v in metadata.actuator_type_to_metadata.items()
                }
            ),
            control_frequency=_parse_float(metadata.control_frequency),
        )

    @classmethod
    def from_model(
        cls,
        model: PhysicsModel,
        kp: float | None = None,
        kd: float | None = None,
        armature: float | None = None,
        friction: float | None = None,
        soft_torque_limit: float | None = None,
    ) -> "Metadata":
        return cls(
            joint_name_to_metadata=JointMetadata.from_model(
                model,
                kp=kp,
                kd=kd,
                armature=armature,
                friction=friction,
                soft_torque_limit=soft_torque_limit,
            ),
            actuator_type_to_metadata=ActuatorMetadata.from_model(model),
            control_frequency=None,
        )
