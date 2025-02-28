"""Base class for controllers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import mujoco.mjx as mjx
from jaxtyping import Array

from ksim.utils.mujoco import MujocoMappings


@dataclass
class BaseActuatorMetadata(ABC):
    """Base class for actuator metadata."""


class Actuators(ABC):
    """Collection of actuators."""

    @abstractmethod
    def __init__(
        self,
        actuators_metadata: Dict[str, BaseActuatorMetadata],
        mujoco_mappings: MujocoMappings,
        **kwargs: Any,
    ) -> None: ...

    @abstractmethod
    def get_ctrl(self, mjx_data: mjx.Data, action: Array) -> Array:
        """Get the control signal from the action vector."""
        ...

    @property
    @abstractmethod
    def actuator_input_size(self) -> int:
        """Get the size of the actuator input vector."""
        ...
