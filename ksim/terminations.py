"""Defines the base termination class."""

import functools
from abc import ABC, abstractmethod
from typing import Collection, Generic, TypeVar

import equinox as eqx
import jax.numpy as jnp
import xax
from brax.base import State
from brax.mjx.base import State as MjxState

from ksim.utils.data import BuilderData
from ksim.utils.mujoco import lookup_in_dict


class Termination(eqx.Module, ABC):
    @abstractmethod
    def __call__(self, state: State) -> jnp.ndarray:
        """Checks if the environment has terminated.

        Args:
            state: The current state to check.

        Returns:
            A boolean tensor indicating whether the environment has terminated.
            This tensor should usually be of shape (num_envs).
        """

    def get_name(self) -> str:
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def termination_name(self) -> str:
        return self.get_name()


T = TypeVar("T", bound=Termination)


class TerminationBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, data: BuilderData) -> T:
        """Builds a termination from a MuJoCo model.

        Args:
            data: The data to build the termination from.

        Returns:
            A termination that can be applied to a state.
        """


class PitchTooGreatTermination(Termination):
    """Terminates the episode if the pitch is too great."""

    max_pitch: float

    def __init__(self, max_pitch: float) -> None:
        super().__init__()

        self.max_pitch = max_pitch

    def __call__(self, state: State) -> jnp.ndarray:
        quat = state.q[3:7]
        pitch = jnp.arctan2(2 * quat[1] * quat[2] - 2 * quat[0] * quat[3], 1 - 2 * quat[1] ** 2 - 2 * quat[2] ** 2)
        return jnp.abs(pitch) > self.max_pitch


class RollTooGreatTermination(Termination):
    """Terminates the episode if the roll is too great."""

    max_roll: float

    def __init__(self, max_roll: float) -> None:
        super().__init__()

        self.max_roll = max_roll

    def __call__(self, state: State) -> jnp.ndarray:
        quat = state.q[3:7]
        roll = jnp.arctan2(2 * quat[1] * quat[2] + 2 * quat[0] * quat[3], 1 - 2 * quat[2] ** 2 - 2 * quat[3] ** 2)
        return jnp.abs(roll) > self.max_roll


class MinimumHeightTermination(Termination):
    """Terminates the episode if the robot is too low."""

    min_height: float

    def __init__(self, min_height: float) -> None:
        super().__init__()

        self.min_height = min_height

    def __call__(self, state: State) -> jnp.ndarray:
        return state.q[2] < self.min_height


class IllegalContactTermination(Termination):
    """Terminates the episode if the robot is in an illegal contact."""

    body_ids: jnp.ndarray
    contact_eps: float

    def __init__(self, body_ids: Collection[int], contact_eps: float = -1e-3) -> None:
        super().__init__()

        self.body_ids = jnp.array(sorted(body_ids))
        self.contact_eps = contact_eps

    def __call__(self, state: State) -> jnp.ndarray:
        if state.contact is None:
            return jnp.zeros_like(state.q[0])

        contact = state.contact

        if isinstance(state, MjxState):
            has_contact = jnp.any(contact.geom[:, :, None] == self.body_ids[None, None, :], axis=(1, 2))
            return jnp.where(has_contact, contact.dist, 1e4).min() <= self.contact_eps

        else:
            raise NotImplementedError(f"IllegalContactTermination is not implemented for {type(state)}")


class IllegalContactTerminationBuilder(TerminationBuilder[IllegalContactTermination]):
    def __init__(self, body_names: Collection[str]) -> None:
        super().__init__()

        self.body_names = body_names

    def __call__(self, data: BuilderData) -> IllegalContactTermination:
        body_ids = lookup_in_dict(self.body_names, data.body_name_to_idx, "Body")
        return IllegalContactTermination(body_ids)
