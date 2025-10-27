"""Defines types of noise to add to observations.

This is a bit verbose, but doing this provides a common interface for passing
random variables to other parts of the code, for type checking.
"""

__all__ = [
    # Random Variable
    "RandomVariable",
    "ZeroRandomVariable",
    "UniformRandomVariable",
    "GaussianRandomVariable",
    "UniformGaussianRandomVariable",
    # Bias
    "Bias",
    "AdditiveBias",
    "MultiplicativeBias",
    "AdditiveGaussianBias",
    "MultiplicativeGaussianBias",
    "AdditiveUniformBias",
    "MultiplicativeUniformBias",
    "NoBias",
    # Noise
    "Noise",
    "AdditiveNoise",
    "MultiplicativeNoise",
    "AdditiveGaussianNoise",
    "MultiplicativeGaussianNoise",
    "AdditiveUniformNoise",
    "MultiplicativeUniformNoise",
    "NoNoise",
    "ChainedNoise",
]

from abc import ABC, abstractmethod

import attrs
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array, PRNGKeyArray

from ksim.debugging import JitLevel

# ---------------
# Random Variable
# ---------------


@attrs.define(frozen=True, kw_only=True)
class RandomVariable(ABC):
    @abstractmethod
    def get_random_variable(
        self,
        shape: tuple[int, ...],
        rng: PRNGKeyArray,
        curriculum_level: Array | float = 1.0,
    ) -> Array: ...


@attrs.define(frozen=True, kw_only=True)
class ZeroRandomVariable(RandomVariable):
    """A random variable that always returns zero."""

    def get_random_variable(
        self,
        shape: tuple[int, ...],
        rng: PRNGKeyArray,
        curriculum_level: Array | float = 1.0,
    ) -> Array:
        return jnp.zeros(shape)


@attrs.define(frozen=True, kw_only=True)
class UniformRandomVariable(RandomVariable):
    mean: float = attrs.field()
    mag: float = attrs.field(validator=attrs.validators.gt(0.0))

    @xax.jit(static_argnames=["self", "shape"], donate_argnames=["rng"], jit_level=JitLevel.HELPER_FUNCTIONS)
    def get_random_variable(
        self,
        shape: tuple[int, ...],
        rng: PRNGKeyArray,
        curriculum_level: Array | float = 1.0,
    ) -> Array:
        minv, maxv = self.mean - self.mag * curriculum_level, self.mean + self.mag * curriculum_level
        return jax.random.uniform(rng, shape, minval=minv, maxval=maxv)


@attrs.define(frozen=True, kw_only=True)
class GaussianRandomVariable(RandomVariable):
    mean: float = attrs.field()
    std: float = attrs.field(validator=attrs.validators.gt(0.0))

    @xax.jit(static_argnames=["self", "shape"], donate_argnames=["rng"], jit_level=JitLevel.HELPER_FUNCTIONS)
    def get_random_variable(
        self,
        shape: tuple[int, ...],
        rng: PRNGKeyArray,
        curriculum_level: Array | float = 1.0,
    ) -> Array:
        return jax.random.normal(rng, shape) * (self.std * curriculum_level) + self.mean


@attrs.define(frozen=True, kw_only=True)
class UniformGaussianRandomVariable(RandomVariable):
    mean: float = attrs.field()
    std: float = attrs.field(validator=attrs.validators.gt(0.0))
    mag: float = attrs.field(validator=attrs.validators.gt(0.0))

    @xax.jit(static_argnames=["self", "shape"], donate_argnames=["rng"], jit_level=JitLevel.HELPER_FUNCTIONS)
    def get_random_variable(
        self,
        shape: tuple[int, ...],
        rng: PRNGKeyArray,
        curriculum_level: Array | float = 1.0,
    ) -> Array:
        minv, maxv = -self.mag * curriculum_level, self.mag * curriculum_level
        normal = jax.random.normal(rng, shape) * (self.std * curriculum_level)
        uniform = jax.random.uniform(rng, shape, minval=minv, maxval=maxv)
        return normal + uniform + self.mean


# ----
# Bias
# ----


@attrs.define(frozen=True, kw_only=True)
class Bias(RandomVariable):
    @abstractmethod
    def apply_bias(self, observation: Array, random_variable: Array) -> Array: ...


@attrs.define(frozen=True, kw_only=True)
class AdditiveBias(Bias):
    def apply_bias(self, observation: Array, random_variable: Array) -> Array:
        return observation + random_variable


@attrs.define(frozen=True, kw_only=True)
class MultiplicativeBias(Bias):
    def apply_bias(self, observation: Array, random_variable: Array) -> Array:
        return observation * random_variable


@attrs.define(frozen=True, kw_only=True)
class AdditiveGaussianBias(AdditiveBias, GaussianRandomVariable):
    mean: float = attrs.field(default=0.0)


@attrs.define(frozen=True, kw_only=True)
class MultiplicativeGaussianBias(MultiplicativeBias, GaussianRandomVariable):
    mean: float = attrs.field(default=1.0)


@attrs.define(frozen=True, kw_only=True)
class AdditiveUniformBias(AdditiveBias, UniformRandomVariable):
    mean: float = attrs.field(default=0.0)


@attrs.define(frozen=True, kw_only=True)
class MultiplicativeUniformBias(MultiplicativeBias, UniformRandomVariable):
    mean: float = attrs.field(default=1.0)


@attrs.define(frozen=True, kw_only=True)
class NoBias(Bias):
    def apply_bias(self, observation: Array, random_variable: Array) -> Array:
        return observation


# -----
# Noise
# -----


@attrs.define(frozen=True, kw_only=True)
class Noise(ABC):
    @abstractmethod
    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array: ...


@attrs.define(frozen=True, kw_only=True)
class AdditiveNoise(Noise, RandomVariable):
    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return observation + self.get_random_variable(observation.shape, rng, curriculum_level)


@attrs.define(frozen=True, kw_only=True)
class MultiplicativeNoise(Noise, RandomVariable):
    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return observation * self.get_random_variable(observation.shape, rng, curriculum_level)


@attrs.define(frozen=True, kw_only=True)
class AdditiveGaussianNoise(AdditiveNoise, GaussianRandomVariable):
    mean: float = attrs.field(default=0.0)


@attrs.define(frozen=True, kw_only=True)
class MultiplicativeGaussianNoise(MultiplicativeNoise, GaussianRandomVariable):
    mean: float = attrs.field(default=1.0)


@attrs.define(frozen=True, kw_only=True)
class AdditiveUniformNoise(AdditiveNoise, UniformRandomVariable):
    mean: float = attrs.field(default=0.0)


@attrs.define(frozen=True, kw_only=True)
class MultiplicativeUniformNoise(MultiplicativeNoise, UniformRandomVariable):
    mean: float = attrs.field(default=1.0)


@attrs.define(frozen=True, kw_only=True)
class NoNoise(Noise):
    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        return observation


@attrs.define(frozen=True, kw_only=True)
class ChainedNoise(Noise):
    noises: tuple[Noise, ...] = attrs.field()

    def add_noise(self, observation: Array, curriculum_level: Array, rng: PRNGKeyArray) -> Array:
        for noise in self.noises:
            observation = noise.add_noise(observation, curriculum_level, rng)
        return observation
