"""Defines common types used throughout the project."""

__all__ = [
    "JointMetadata",
    "ActuatorMetadata",
    "Metadata",
]

from dataclasses import dataclass
from typing import TypeVar

from kscale.web.gen.api import (
    ActuatorMetadataOutput,
    JointMetadataOutput,
    RobotURDFMetadataOutput,
)

from ksim.types import PhysicsModel

T = TypeVar("T")


def _parse_float(value: str | None) -> float | None:
    return None if value is None else float(value)


def _nn(value: T | None, name: str) -> T:
    if value is None:
        raise ValueError(f"'{name}' from the K-Scale API is None")
    return value


@dataclass
class JointMetadata:
    kp: float | None = None
    kd: float | None = None
    armature: float | None = None
    friction: float | None = None
    soft_torque_limit: float | None = None

    @classmethod
    def from_kscale_joint_metadata(cls, metadata: JointMetadataOutput) -> "JointMetadata":
        return cls(
            kp=_parse_float(metadata.kp),
            kd=_parse_float(metadata.kd),
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
    ) -> dict[str, "JointMetadata"]:
        return {
            model.names[model.name_jntadr[i] :]
            .decode("utf-8")
            .split("\x00", 1)[0]: cls(
                kp=kp,
                kd=kd,
                armature=armature,
                friction=friction,
                soft_torque_limit=soft_torque_limit,
            )
            for i in range(model.njnt)
        }


@dataclass
class ActuatorMetadata:
    actuator_type: str
    max_torque: float | None = None

    @classmethod
    def from_kscale_actuator_metadata(cls, metadata: ActuatorMetadataOutput) -> "ActuatorMetadata":
        return cls(
            actuator_type="motor",
        )

    @classmethod
    def from_model(cls, model: PhysicsModel) -> dict[str, "ActuatorMetadata"]:
        return {"motor": cls(actuator_type="motor")}


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
            joint_name_to_metadata=JointMetadata.from_model(model),
            actuator_type_to_metadata=ActuatorMetadata.from_model(model),
            control_frequency=None,
        )
