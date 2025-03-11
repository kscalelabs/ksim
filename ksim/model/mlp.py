"""Base MLP implementation using Flax Linen."""

from typing import Callable

import flax.linen as nn
import jax


class MLP(nn.Module):
    """A multi-layer perceptron."""

    out_dim: int
    """The number of features in the output layer."""

    hidden_dims: tuple[int, ...]
    """The number of features in the hidden layers."""

    activation: Callable[[jax.Array], jax.Array] = nn.swish
    """The activation function to use in the MLP."""

    bias_init: nn.initializers.Initializer = nn.initializers.zeros
    """The initializer to use for the bias of the dense layers."""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass of the MLP.

        Args:
            x: The input to the MLP [..., in_features]

        Returns:
            The output of the MLP [..., out_features]
        """
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(
                features=hidden_dim,
                kernel_init=nn.initializers.lecun_uniform(),
            )(x)
            x = self.activation(x)

        x = nn.Dense(
            features=self.out_dim,
            kernel_init=nn.initializers.lecun_uniform(),
            bias_init=self.bias_init,
        )(x)
        return x
