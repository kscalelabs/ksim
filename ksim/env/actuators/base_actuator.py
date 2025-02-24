"""Base class for controllers."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from jaxtyping import Array

from ksim.env.mjx_env import ActuatorMetadata, EnvState
from ksim.utils.mujoco import MujocoMappings


class Actuators(ABC):
    """Collection of actuators."""

    @abstractmethod
    def __init__(
        self,
        actuators_metadata: Dict[str, ActuatorMetadata],
        mujoco_mappings: MujocoMappings,
        **kwargs: Any,
    ) -> None:
        ...

    @abstractmethod
    def get_ctrl(self, env_state: EnvState, action: Array) -> Array:
        """Get the control signal from the action vector."""
        
