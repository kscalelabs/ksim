"""Probability distributions implemented using the stateless class interface."""

from abc import ABC, abstractmethod

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ksim.utils.constants import EPSILON


@attrs.define(kw_only=True, frozen=True)
class ActionDistribution(ABC):
    """Abstract class for parametrized action distribution."""

    action_dim: int = attrs.field()
    """Shape of the distribution's output vector."""

    @property
    @abstractmethod
    def num_params(self) -> int:
        """Number of parameters of the distribution. Function of action_dim."""
        ...

    @abstractmethod
    def sample(self, parameters: Array, rng: Array) -> Array:
        """Returns a sample from the postprocessed distribution."""
        ...

    @abstractmethod
    def mode(self, parameters: Array) -> Array:
        """Returns the mode of the postprocessed distribution."""
        ...

    @abstractmethod
    def log_prob(self, parameters: Array, actions: Array) -> Array:
        """Compute the log probability of actions."""
        ...

    @abstractmethod
    def entropy(self, parameters: Array, rng: Array) -> Array:
        """Return the entropy of the given distribution.

        Note that we pass in rng because some distributions may require
        sampling to compute the entropy.
        """
        ...


@attrs.define(kw_only=True, frozen=True)
class GaussianDistribution(ActionDistribution):
    """Normal distribution."""

    min_std: float = attrs.field(default=0.001)
    """Minimum standard deviation for numerical stability. Brax defaults."""

    var_scale: float = attrs.field(default=1.0)
    """Scaling factor for the variance. Using Brax default values."""

    def get_mean_std(self, parameters: Array) -> tuple[Array, Array]:
        """Split the parameters into the mean and standard deviation.

        Following Brax's method of using softplus to ensure positive std.
        """
        # Validate that parameters has the expected shape
        if parameters.shape[-1] != 2 * self.action_dim:
            raise ValueError(
                f"Expected parameters with last dimension of size {2 * self.action_dim}, "
                f"but got {parameters.shape[-1]}. Make sure the parameters match the "
                f"distribution's num_params ({self.num_params})."
            )

        mean, std = jnp.split(parameters, 2, axis=-1)
        std = (jax.nn.softplus(std) + self.min_std) * self.var_scale
        return mean, std

    @property
    def num_params(self) -> int:
        """Number of parameters of the distribution. Function of action_dim."""
        return 2 * self.action_dim

    def sample(self, parameters: Array, rng: Array) -> Array:
        """Sample from the normal distribution.

        Parameters should be the concatenation of the mean and standard
        deviation. As such, it should have shape (..., 2 * action_dim).
        """
        mean, std = self.get_mean_std(parameters)
        return jax.random.normal(rng, shape=mean.shape) * std + mean

    def mode(self, parameters: Array) -> Array:
        """Returns the mode of the normal distribution.

        Parameters should be the concatenation of the mean and standard
        deviation. As such, it should have shape (..., 2 * action_dim).
        """
        mean, _ = self.get_mean_std(parameters)
        return mean

    def log_prob(self, parameters: Array, actions: Array) -> Array:
        """Compute the log probability of actions."""
        mean, std = self.get_mean_std(parameters)
        log_probs = (
            -0.5 * jnp.square((actions - mean) / std) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
        )
        return jnp.sum(log_probs, axis=-1)

    def entropy(self, parameters: Array, rng: Array) -> Array:
        """Return the entropy of the normal distribution

        Parameters should be the concatenation of the mean and standard
        deviation. As such, it should have shape (..., 2 * action_dim).
        """
        _, std = self.get_mean_std(parameters)
        entropies = 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(std)
        return jnp.sum(entropies, axis=-1)


@attrs.define(kw_only=True, frozen=True)
class TanhGaussianDistribution(GaussianDistribution):
    """Normal distribution followed by tanh."""

    def _log_det_jacobian(self, actions: Array) -> Array:
        """Compute the log determinant of the jacobian of the tanh transform.

        $p(x) = p(y) * |dy/dx| = p(tanh(x)) * |1 - tanh(x)^2|$
        """
        return 2.0 * (jnp.log(2.0) - actions - jax.nn.softplus(-2.0 * actions))

    def sample(self, parameters: Array, rng: Array) -> Array:
        """Sample from the normal distribution and apply tanh.

        Parameters should be the concatenation of the mean and standard
        deviation parameters. As such, it should have shape (..., 2 * action_dim).
        """
        normal_sample = super().sample(parameters, rng)
        return jnp.tanh(normal_sample)

    def mode(self, parameters: Array) -> Array:
        """Returns the mode of the normal-tanh distribution.

        For the normal distribution, the mode is the mean.
        After applying tanh, the mode is tanh(mean).
        """
        return jnp.tanh(super().mode(parameters))

    def log_prob(self, parameters: Array, actions: Array) -> Array:
        """Compute the log probability of actions.

        This formulation computes the Gaussian log density on the pre-tanh values and then
        subtracts the Jacobian correction computed directly from the final actions.
        """

        mean, std = self.get_mean_std(parameters)
        # Compute the pre-tanh values from the actions (with clipping for stability)
        pre_tanh = jnp.arctanh(jnp.clip(actions, -1 + EPSILON, 1 - EPSILON))
        # Compute the base log probability from the Gaussian density
        base_log_prob = (
            -0.5 * jnp.square((pre_tanh - mean) / std) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
        )
        base_log_prob = jnp.sum(base_log_prob, axis=-1)
        # Compute the log-determinant of the Jacobian for the tanh transformation
        jacobian_correction = jnp.sum(jnp.log(1 - jnp.square(actions) + EPSILON), axis=-1)
        return base_log_prob - jacobian_correction

    def entropy(self, parameters: Array, rng: Array) -> Array:
        """Return the entropy of the normal-tanh distribution.

        Approximates entropy using sampling since there is no closed-form
        solution.
        """
        _, std = self.get_mean_std(parameters)
        normal_entropies = 0.5 + 0.5 * jnp.log(2 * jnp.pi) + jnp.log(std)
        # getting log det jac of tanh transformed sample...
        normal_sample = super().sample(parameters, rng)
        log_det_jacobian = self._log_det_jacobian(normal_sample)

        # since we're in log space, can subtract the log det jacobian
        entropies = normal_entropies - log_det_jacobian
        return jnp.sum(entropies, axis=-1)


@attrs.define(kw_only=True, frozen=True)
class CategoricalDistribution(ActionDistribution):
    """Categorical distribution."""

    @property
    def num_params(self) -> int:
        """Number of parameters of the distribution. Function of action_dim."""
        return self.action_dim

    def sample(self, parameters: Array, rng: Array) -> Array:
        """Sample from the categorical distribution. Parameters are logits.

        Parameters should have shape (..., num_actions).
        """
        return jax.random.categorical(rng, parameters)

    def mode(self, parameters: Array) -> Array:
        """Returns the mode of the categorical distribution.

        Parameters should have shape (..., num_actions).
        """
        return jnp.argmax(parameters, axis=-1)

    def log_prob(self, parameters: Array, actions: Array) -> Array:
        """Compute the log probability of actions."""
        logits = parameters
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        batch_shape = actions.shape
        flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
        flat_actions = actions.reshape(-1)
        flat_action_log_prob = flat_log_probs[jnp.arange(flat_log_probs.shape[0]), flat_actions]
        action_log_prob = flat_action_log_prob.reshape(batch_shape)

        return action_log_prob

    def entropy(self, parameters: Array, rng: Array) -> Array:
        """Return the entropy of the categorical distribution."""
        logits = parameters
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        entropies = -jnp.sum(log_probs * jnp.exp(log_probs), axis=-1)
        return entropies
