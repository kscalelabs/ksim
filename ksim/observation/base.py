"""Defines the base observation class."""

import functools
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from brax.base import State
from jaxtyping import PRNGKeyArray

from ksim.utils.data import BuilderData

NoiseType = Literal["gaussian", "uniform"]


class Observation(eqx.Module, ABC):
    noise: float
    noise_type: NoiseType

    def __init__(
        self,
        noise: float = 0.0,
        noise_type: NoiseType = "gaussian",
    ) -> None:
        super().__init__()

        self.noise = noise
        self.noise_type = noise_type

    @abstractmethod
    def __call__(self, state: State) -> jnp.ndarray:
        """Gets the observation from the state."""

    def add_noise(self, observation: jnp.ndarray, rng: PRNGKeyArray) -> jnp.ndarray:
        match self.noise_type:
            case "gaussian":
                return observation + jax.random.normal(rng, observation.shape) * self.noise
            case "uniform":
                return observation + jax.random.uniform(rng, observation.shape, minval=-self.noise, maxval=self.noise)
            case _:
                raise ValueError(f"Invalid noise type: {self.noise_type}")

    @classmethod
    def get_name(cls) -> str:
        return xax.camelcase_to_snakecase(cls.__name__)

    @functools.cached_property
    def observation_name(self) -> str:
        return self.get_name()


T = TypeVar("T", bound=Observation)


class ObservationBuilder(ABC, Generic[T]):
    @abstractmethod
    def __call__(self, data: BuilderData) -> T:
        """Builds an observation from a MuJoCo model.

        Args:
            data: The data to build the observation from.

        Returns:
            An observation that can be applied to a state.
        """
