"""Defines common types used throughout the project."""

from dataclasses import dataclass
from typing import TypeVar

from kscale.web.gen.api import (
    ActuatorMetadataOutput,
    JointMetadataOutput,
    RobotURDFMetadataOutput,
)

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

    @classmethod
    def from_kscale_joint_metadata(cls, metadata: JointMetadataOutput) -> "JointMetadata":
        return cls(
            kp=_parse_float(metadata.kp),
            kd=_parse_float(metadata.kd),
        )


@dataclass
class ActuatorMetadata:
    actuator_type: str
    max_torque: float

    @classmethod
    def from_kscale_actuator_metadata(cls, metadata: ActuatorMetadataOutput) -> "ActuatorMetadata":
        return cls(
            actuator_type=_nn(metadata.actuator_type, "actuator_type"),
            max_torque=_nn(_parse_float(metadata.max_torque), "max_torque"),
        )


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
