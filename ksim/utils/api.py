"""Defines utility functions for pulling from the K-Scale API."""

__all__ = [
    "get_mujoco_model_path",
    "get_mujoco_model_metadata",
    "get_mujoco_model_and_metadata",
]

import asyncio
import json
import logging
from pathlib import Path

from kscale import K
from kscale.web.gen.api import RobotURDFMetadataOutput
from kscale.web.utils import get_robots_dir, should_refresh_file

from ksim.types import Metadata

logger = logging.getLogger(__name__)


async def get_mujoco_model_path(model_name: str, cache: bool = True, name: str | None = None) -> str | Path:
    """Downloads and caches the model URDF if it doesn't exists in a local directory."""
    try:
        urdf_dir = Path(model_name)
        if not urdf_dir.exists():
            raise ValueError(f"Model {model_name} does not exist")
    except ValueError:
        async with K() as api:
            urdf_dir = await api.download_and_extract_urdf(model_name, cache=cache)

    if name is None:
        try:
            mjcf_path = next(urdf_dir.glob("*.mjcf"))
        except StopIteration as err:
            raise ValueError(f"No MJCF file found for {model_name} (in {urdf_dir})") from err
    else:
        mjcf_path = urdf_dir / f"{name}.mjcf"
        if not mjcf_path.exists():
            raise ValueError(f"MJCF file {name} does not exist for {model_name} (in {urdf_dir})")

    return mjcf_path


async def get_mujoco_model_metadata(model_name: str, cache: bool = True) -> Metadata:
    """Downloads and caches the model metadata."""
    try:
        directory = Path(model_name)
        if not directory.exists():
            raise ValueError(f"Model {model_name} does not exist")
        metadata_path = directory / "metadata.json"

    except ValueError as err:
        metadata_path = get_robots_dir() / model_name / "metadata.json"

        # Downloads and caches the metadata if it doesn't exist.
        if not cache or not (metadata_path.exists() and not should_refresh_file(metadata_path)):
            async with K() as api:
                robot_class = await api.get_robot_class(model_name)
                if (metadata := robot_class.metadata) is None:
                    raise ValueError(f"No metadata found for {model_name}") from err

            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w") as f:
                json.dump(metadata.model_dump(), f, indent=2)

    with open(metadata_path, "r") as f:
        metadata = RobotURDFMetadataOutput.model_validate_json(f.read())

    return Metadata.from_kscale_metadata(metadata)


async def get_mujoco_model_and_metadata(
    model_name: str,
    cache: bool = True,
) -> tuple[str, Metadata]:
    """Downloads and caches the model URDF and metadata."""
    urdf_path, metadata = await asyncio.gather(
        get_mujoco_model_path(
            model_name=model_name,
            cache=cache,
        ),
        get_mujoco_model_metadata(
            model_name=model_name,
            cache=cache,
        ),
    )
    return str(urdf_path), metadata
