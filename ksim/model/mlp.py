"""Base MLP implementation using Flax Linen."""

import flax.linen as nn
import jax
from dataclasses import dataclass, field


class MLP(nn.Module):
    """A multi-layer perceptron."""

    num_hidden_layers: int
    """The number of hidden layers in the MLP."""

    hidden_features: int
    """The number of features in the hidden layers."""

    out_features: int
    """The number of features in the output layer."""

    dense_init_scale: float
    """The scale of the dense layer initialization."""

    output_layer_init_scale: float
    """The scale of the output layer initialization."""

    def setup(self):
        # Create hidden layers
        self.hidden_layers = [
            nn.Dense(
                features=self.hidden_features,
                kernel_init=nn.initializers.orthogonal(self.dense_init_scale),
                bias_init=nn.initializers.zeros,
            )
            for _ in range(self.num_hidden_layers)
        ]

        # Create output layer
        self.output_layer = nn.Dense(
            features=self.out_features,
            kernel_init=nn.initializers.orthogonal(self.output_layer_init_scale),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass of the MLP.

        Args:
            x: The input to the MLP [..., in_features]

        Returns:
            The output of the MLP [..., out_features]
        """
        # Pass through hidden layers with ReLU activation
        for layer in self.hidden_layers:
            x = layer(x)
            x = nn.relu(x)

        # Pass through output layer
        x = self.output_layer(x)
        return x
