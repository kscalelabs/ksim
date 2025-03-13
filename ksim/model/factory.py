"""Your one-stop-shop for common model formulations."""

from typing import Any, Callable

import flax.linen as nn
from jaxtyping import Array

from ksim.model.base import ActorCriticAgent, KSimModel
from ksim.model.distributions import (
    ActionDistribution,
    CategoricalDistribution,
    GaussianDistribution,
    TanhGaussianDistribution,
)
from ksim.model.mixins.forward import MLPMixin
from ksim.model.mixins.post_process import (
    DirectActionMixin,
    MeanHeadMixin,
    MeanStdHeadMixin,
)
from ksim.model.types import DistributionType, PredictionType


class MLPMeanStdActor(MLPMixin, MeanStdHeadMixin):
    """MLP model that predicts mean and std."""

    pass


class MLPMeanActor(MLPMixin, MeanHeadMixin):
    """MLP model that predicts mean."""

    pass


class MLPDirect(MLPMixin, DirectActionMixin):
    """MLP model without parametric distribution post-processing."""

    pass


def mlp_actor_critic_agent(
    *,
    num_actions: int,
    prediction_type: PredictionType,
    distribution_type: DistributionType,
    actor_hidden_dims: tuple[int, ...] = (64,) * 5,
    critic_hidden_dims: tuple[int, ...] = (64,) * 5,
    filter_actor_obs: tuple[str, ...] | None = None,
    filter_critic_obs: tuple[str, ...] | None = None,
    activation: Callable[[Array], Array] = nn.relu,
    kernel_initialization: nn.initializers.Initializer = nn.initializers.lecun_normal(),
    bias_initialization: nn.initializers.Initializer = nn.initializers.zeros,
    post_process_kwargs: dict[str, Any] = {},
    distribution_kwargs: dict[str, Any] = {},
) -> ActorCriticAgent:
    """This is a simple factory function for an actor-critic mlp agent."""
    # first create the distribution
    distribution: ActionDistribution
    if distribution_type == "gaussian":
        distribution = GaussianDistribution(action_dim=num_actions, **distribution_kwargs)
    elif distribution_type == "tanh_gaussian":
        distribution = TanhGaussianDistribution(action_dim=num_actions, **distribution_kwargs)
    elif distribution_type == "categorical":
        if "sampling_temperature" not in distribution_kwargs:
            raise ValueError("sampling_temperature must be provided for categorical distribution.")
        distribution = CategoricalDistribution(
            action_dim=num_actions,
            sampling_temperature=distribution_kwargs["sampling_temperature"],
        )

    # Then create the actor module.
    actor_module: KSimModel
    if prediction_type == "mean_std":
        actor_module = MLPMeanStdActor(
            out_dim=distribution.num_params,
            distribution=distribution,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            obs_names=filter_actor_obs,
            kernel_initialization=kernel_initialization,
            bias_initialization=bias_initialization,
            **post_process_kwargs,
        )
    elif prediction_type == "mean":
        actor_module = MLPMeanActor(
            out_dim=distribution.num_params,
            distribution=distribution,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            obs_names=filter_actor_obs,
            kernel_initialization=kernel_initialization,
            bias_initialization=bias_initialization,
            **post_process_kwargs,
        )
    elif prediction_type == "direct":
        actor_module = MLPDirect(
            out_dim=distribution.num_params,
            distribution=distribution,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            obs_names=filter_actor_obs,
            kernel_initialization=kernel_initialization,
            bias_initialization=bias_initialization,
            **post_process_kwargs,
        )

    critic_module = MLPDirect(
        out_dim=1,
        distribution=distribution,
        actor_hidden_dims=critic_hidden_dims,
        activation=activation,
        obs_names=filter_critic_obs,
        kernel_initialization=kernel_initialization,
        bias_initialization=bias_initialization,
    )

    return ActorCriticAgent(
        actor_module=actor_module,
        critic_module=critic_module,
        distribution=distribution,
    )
