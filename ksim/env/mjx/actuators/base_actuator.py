"""Base class for controllers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

from jaxtyping import Array

from ksim.env.mjx.mjx_env import MjxEnvState
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
    def get_ctrl(self, env_state: MjxEnvState, action: Array) -> Array:
        """Get the control signal from the action vector."""
        ...
