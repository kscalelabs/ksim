"""Base MLP implementation using Flax Linen."""

from typing import Callable

import flax.linen as nn
from jaxtyping import Array


class MLP(nn.Module):
    """A multi-layer perceptron."""

    out_dim: int
    """The number of features in the output layer."""

    hidden_dims: tuple[int, ...]
    """The number of features in the hidden layers."""

    activation: Callable[[Array], Array] = nn.relu
    """The activation function to use in the MLP."""

    kernel_initialization: nn.initializers.Initializer = nn.initializers.lecun_normal()
    """The initializer to use for the kernel of the dense layers."""

    bias_initialization: nn.initializers.Initializer = nn.initializers.zeros
    """The initializer to use for the bias of the dense layers."""

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """Forward pass of the MLP."""
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(
                features=hidden_dim,
                kernel_init=self.kernel_initialization,
                bias_init=self.bias_initialization,
            )(x)
            x = self.activation(x)

        x = nn.Dense(
            features=self.out_dim,
            kernel_init=self.kernel_initialization,
            bias_init=self.bias_initialization,
        )(x)
        return x
