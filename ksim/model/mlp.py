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

    activation: nn.Module = nn.eelu
    """The activation function to use in the MLP."""

    bias_init: nn.initializers.Initializer = nn.initializers.zeros
    """The initializer to use for the bias of the dense layers."""

    output_layer_scale: float = 0.01
    """The scale to use for the output layer."""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass of the MLP.

        Args:
            x: The input to the MLP [..., in_features]

        Returns:
            The output of the MLP [..., out_features]
        """
        for _ in range(self.num_hidden_layers):
            x = nn.Dense(
                features=self.hidden_features, 
                kernel_init=nn.initializers.orthogonal(scale=1.0),
                bias_init=nn.initializers.zeros
            )(x)
            x = self.activation(x)

        x = nn.Dense(
            features=self.out_features, 
            kernel_init=nn.initializers.orthogonal(scale=self.output_layer_scale),
            bias_init=self.bias_init
        )(x)
        return x
