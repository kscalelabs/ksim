"""Base class for controllers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from jaxtyping import Array
from mujoco import mjx


class Actuators(ABC):
    """Collection of actuators."""

    @abstractmethod
    def get_ctrl(self, mjx_data: mjx.Data, action: Array) -> Array:
        """Get the control signal from the action vector."""
        ...

    @property
    @abstractmethod
    def actuator_input_size(self) -> int:
        """Get the size of the actuator input vector."""
        ...
