"""Defines an interface for loading environments from the K-Scale API."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import xax
from kscale import K
from omegaconf import MISSING

from ksim.action.mjcf import MjcfAction
from ksim.env.mjcf import MjcfEnvironment, MjcfEnvironmentConfig
from ksim.state.mjcf import MjcfState


@dataclass(kw_only=True)
class KScaleEnvironmentConfig(MjcfEnvironmentConfig):
    model_name: str = xax.field(value=MISSING, help="Name of the robot model")
    ignore_cached_urdf: bool = xax.field(value=False, help="Whether to ignore the cached URDF.")


Tconfig = TypeVar("Tconfig", bound=KScaleEnvironmentConfig)
Tstate = TypeVar("Tstate", bound=MjcfState)
Taction = TypeVar("Taction", bound=MjcfAction)


class KScaleEnvironment(MjcfEnvironment[Tconfig, Tstate, Taction], Generic[Tconfig, Tstate, Taction]):
    async def get_model_path(self) -> str | Path:
        async with K() as api:
            urdf_dir = await api.download_and_extract_urdf(
                self.config.model_name,
                cache=not self.config.ignore_cached_urdf,
            )

        try:
            urdf_path = next(urdf_dir.glob("*.mjcf"))
        except StopIteration:
            raise ValueError(f"No MJCF file found for {self.config.model_name} (in {urdf_dir})")

        return urdf_path
