"""Utils for downloading and caching robot models."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from kscale import K
from kscale.web.utils import get_robots_dir, should_refresh_file
from omegaconf import OmegaConf

from ksim.env.mjx.actuators.base_actuator import BaseActuatorMetadata
from ksim.env.mjx.actuators.mit_actuator import MITPositionActuatorMetadata

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    actuators: dict[str, Any]  # NOTE: using BaseActuatorMetadata breaks OmegaConf
    control_frequency: float


async def get_model_path(model_name: str, cache: bool = True) -> str | Path:
    """Downloads and caches the model URDF."""
    async with K() as api:
        urdf_dir = await api.download_and_extract_urdf(model_name, cache=cache)

    try:
        mjcf_path = next(urdf_dir.glob("*.mjcf"))
    except StopIteration:
        raise ValueError(f"No MJCF file found for {model_name} (in {urdf_dir})")

    return mjcf_path


async def get_model_metadata(model_name: str, cache: bool = True) -> ModelMetadata:
    """Downloads and caches the model metadata."""
    metadata_path = get_robots_dir() / model_name / "metadata.yaml"

    # Downloads and caches the metadata if it doesn't exist.
    if not cache or not (metadata_path.exists() and not should_refresh_file(metadata_path)):
        async with K() as api:
            robot_class = await api.get_robot_class(model_name)
            if (metadata := robot_class.metadata) is None:
                raise ValueError(f"No metadata found for {model_name}")

        if (control_frequency := metadata.control_frequency) is None:
            raise ValueError(f"No control frequency found for {model_name}")
        if (actuators := metadata.joint_name_to_metadata) is None:
            raise ValueError(f"No actuators found for {model_name}")

        actuator_metadata = {}

        for name, metadata in actuators.items():
            if hasattr(metadata, "kp") and hasattr(metadata, "kd"):
                actuator_metadata[name] = MITPositionActuatorMetadata(
                    kp=cast(float, metadata.kp),
                    kd=cast(float, metadata.kd),
                )
            else:
                raise ValueError(f"Unknown actuator metadata: {metadata}")

        control_frequency = float(control_frequency)
        model_metadata = ModelMetadata(
            actuators=actuator_metadata, control_frequency=control_frequency
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(model_metadata, metadata_path)

    config = OmegaConf.structured(ModelMetadata)
    return cast(ModelMetadata, OmegaConf.merge(config, OmegaConf.load(metadata_path)))


async def get_model_and_metadata(model_name: str, cache: bool = True) -> tuple[str, ModelMetadata]:
    """Downloads and caches the model URDF and metadata."""
    return await asyncio.gather(
        get_model_path(
            model_name=model_name,
            cache=not cache,
        ),
        get_model_metadata(
            model_name=model_name,
            cache=not cache,
        ),
    )
