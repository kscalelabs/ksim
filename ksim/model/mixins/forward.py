"""Implements standard models for ksim.

Model definitions should not be an active area of research and should mostly be
taken from here.
"""

from typing import Callable

import flax.linen as nn
import jax.numpy as jnp
from jaxtyping import Array
from omegaconf import MISSING

from ksim.model.base import KSimModel
from ksim.model.components.mlp import MLP
from ksim.model.types import ModelInput


class MLPMixin(KSimModel):
    """MLP model that predicts mean and std.

    Naturally can only takes in proprioceptive and command inputs.
    """

    out_dim: int = MISSING
    """Output dimension of the actor MLP."""

    actor_hidden_dims: tuple[int, ...] = (64,) * 5
    """Hidden dimensions for the actor MLP."""

    activation: Callable[[Array], Array] = nn.relu
    """Activation function for the actor MLP."""

    kernel_initialization: nn.initializers.Initializer = nn.initializers.lecun_normal()
    """Kernel initialization for the actor MLP."""

    bias_initialization: nn.initializers.Initializer = nn.initializers.zeros
    """Bias initialization for the actor MLP."""

    def setup(self) -> None:
        """Setup the model."""
        self.mlp = MLP(
            out_dim=self.out_dim,
            hidden_dims=self.actor_hidden_dims,
            activation=self.activation,
            kernel_initialization=self.kernel_initialization,
            bias_initialization=self.bias_initialization,
        )

    def forward(self, x: ModelInput) -> Array:
        """Forward pass of the actor model."""
        input_to_mlp = jnp.concatenate([x.obs_proprio_vec, x.command_vec], axis=-1)
        return self.mlp(input_to_mlp)
