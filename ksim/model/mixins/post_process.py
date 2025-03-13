"""Different post-processing implementations for different distributions."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array

from ksim.model.base import KSimModel


class DirectActionMixin(KSimModel):
    """Mixin for direct action post-processing."""

    def post_process(self, prediction: Array) -> Array:
        """Post-process the output of the network."""
        return prediction


class MeanStdHeadMixin(KSimModel):
    """Mixin for mean and std post-processing."""

    min_std: float = xax.field(value=0.01)
    """Minimum std value."""

    max_std: float = xax.field(value=1.0)
    """Maximum std value."""

    var_scale: float = xax.field(value=1.0)
    """Scale for the std."""

    def post_process(self, prediction: Array) -> Array:
        """Post-process the output of the network.

        We enforce that the first `action_dim` of the output are the mean, and
        the next `action_dim` are the std.
        """
        mean = prediction[..., : self.distribution.action_dim]
        std = prediction[..., self.distribution.action_dim :]

        # softplus and clipping for stability
        std = (jax.nn.softplus(std) + self.min_std) * self.var_scale
        std = jnp.clip(std, self.min_std, self.max_std)

        # concat because Gaussian-like distributions expect the parameters
        # to be mean concat std
        parametrization = jnp.concatenate([mean, std], axis=-1)

        return parametrization


class MeanHeadMixin(KSimModel):
    """Mixin for models that output mean, stds are learned as parameters."""

    min_std: float = xax.field(value=0.01)
    """Minimum std value."""

    max_std: float = xax.field(value=1.0)
    """Maximum std value."""

    std_init: float = xax.field(value=0.3)
    """Initial std value."""

    def post_process(self, prediction: Array) -> Array:
        """Format the output of the network for the distribution.

        We enforce that the first `action_dim` of the output are the mean, and
        the next `action_dim` are the std.
        """
        mean = prediction
        assert isinstance(mean, Array)

        std = self.param(
            "std",
            nn.initializers.constant(self.std_init),
            (self.distribution.action_dim,),
        )

        std = jnp.clip(std, self.min_std, self.max_std)
        std = jnp.tile(std, (*mean.shape[:-1], 1))

        parametrization = jnp.concatenate([mean, std], axis=-1)

        return parametrization
