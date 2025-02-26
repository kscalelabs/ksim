"""Base MLP implementation using Flax Linen."""

import flax.linen as nn
import jax


class MLP(nn.Module):
    """A multi-layer perceptron."""

    num_hidden_layers: int
    """The number of hidden layers in the MLP."""

    hidden_features: int
    """The number of features in the hidden layers."""

    out_features: int
    """The number of features in the output layer."""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass of the MLP.

        Args:
            x: The input to the MLP [..., in_features]

        Returns:
            The output of the MLP [..., out_features]
        """
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(features=self.hidden_features, 
                         kernel_init=nn.initializers.orthogonal(jax.numpy.sqrt(2)),
                         bias_init=nn.initializers.zeros)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.out_features, kernel_init=nn.initializers.orthogonal(0.01),
                     bias_init=nn.initializers.zeros)(x)
        return x


class CriticMLP(MLP):
    """A multi-layer perceptron."""

    num_hidden_layers: int
    """The number of hidden layers in the MLP."""

    hidden_features: int
    """The number of features in the hidden layers."""

    out_features: int
    """The number of features in the output layer."""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass of the MLP.

        Args:
            x: The input to the MLP [..., in_features]

        Returns:
            The output of the MLP [..., out_features]
        """
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(features=self.hidden_features, 
                         kernel_init=nn.initializers.orthogonal(jax.numpy.sqrt(2)),
                         bias_init=nn.initializers.zeros)(x)
            x = nn.relu(x)
        x = nn.Dense(features=self.out_features, kernel_init=nn.initializers.orthogonal(1.0),
                     bias_init=nn.initializers.zeros)(x)
        return x
