"""Defines types of noise to add to observations."""

__all__ = [
    "Noise",
    "AdditiveGaussianNoise",
    "MultiplicativeGaussianNoise",
    "AdditiveUniformNoise",
    "MultiplicativeUniformNoise",
    "ChainedNoise",
]

from abc import ABC, abstractmethod

import attrs
import jax
from jaxtyping import Array, PRNGKeyArray


@attrs.define(frozen=True, kw_only=True)
class Noise(ABC):
    @abstractmethod
    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array: ...


@attrs.define(frozen=True, kw_only=True)
class AdditiveGaussianNoise(Noise):
    std: float = attrs.field()

    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return observation + jax.random.normal(rng, observation.shape) * self.std * curriculum_level


@attrs.define(frozen=True, kw_only=True)
class MultiplicativeGaussianNoise(Noise):
    std: float = attrs.field()
    bias: float = attrs.field(default=1.0)

    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        noise = (jax.random.normal(rng, observation.shape) * self.std * curriculum_level) + self.bias
        return observation * noise


@attrs.define(frozen=True, kw_only=True)
class AdditiveUniformNoise(Noise):
    mag: float = attrs.field()

    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        noise = (jax.random.uniform(rng, observation.shape) * 2 - 1) * self.mag * curriculum_level
        return observation + noise


@attrs.define(frozen=True, kw_only=True)
class MultiplicativeUniformNoise(Noise):
    mag: float = attrs.field()
    bias: float = attrs.field(default=1.0)

    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        noise = ((jax.random.uniform(rng, observation.shape) * 2 - 1) * self.mag * curriculum_level) + self.bias
        return observation * noise


@attrs.define(frozen=True, kw_only=True)
class ChainedNoise(Noise):
    noises: tuple[Noise, ...] = attrs.field()

    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        for noise in self.noises:
            observation = noise.add_noise(observation, curriculum_level, rng)
        return observation
