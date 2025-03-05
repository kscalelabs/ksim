"""High Level Formulations of RL Models."""

from abc import ABC, abstractmethod

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.types import EnvState


class ActionModel(nn.Module, ABC):
    """Action model."""

    num_outputs: int

    @abstractmethod
    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        """Forward pass of the model."""
        ...

    @abstractmethod
    def calc_log_prob(self, prediction: Array, action: Array) -> Array:
        """Calculate the log probability of the action."""
        ...

    @abstractmethod
    def sample_and_log_prob(
        self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array], rng: PRNGKeyArray
    ) -> tuple[Array, Array]:
        """Sample and calculate the log probability of the action."""
        ...


class GaussianActionModel(ActionModel, ABC):
    """Gaussian action model."""

    init_log_std: float

    def setup(self) -> None:
        self.log_std = self.param(
            "log_std", nn.initializers.constant(self.init_log_std), (self.num_outputs,)
        )

    def calc_log_prob(self, prediction: Array, action: Array) -> Array:
        mean = prediction
        std = jnp.exp(self.log_std)

        log_prob = (
            -0.5 * jnp.square((action - mean) / std) - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
        )  # (...batch_dims..., action_dim)
        # if we assume indep gaussians, can just sum over action dim in log space for log prob
        return jnp.sum(log_prob, axis=-1)

    def sample_and_log_prob(
        self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array], rng: PRNGKeyArray
    ) -> tuple[Array, Array]:
        mean = self(obs, cmd)
        std = jnp.exp(self.log_std)

        noise = jax.random.normal(rng, mean.shape)
        action = mean + noise * std
        log_prob = self.calc_log_prob(mean, action)

        return action, log_prob


class CategoricalActionModel(ActionModel, ABC):
    """Categorical action model.

    Assume action space is tokenized such that the last dimension is
    the logits for each action.
    """

    sampling_temperature: float

    def calc_log_prob(self, prediction: Array, action: Array) -> Array:
        logits = prediction
        log_probs = jax.nn.log_softmax(logits, axis=-1)

        # get the log probs for the selected actions (inefficient but compiler should optimize)
        batch_shape = action.shape
        flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
        flat_actions = action.reshape(-1)
        flat_action_log_prob = flat_log_probs[jnp.arange(flat_log_probs.shape[0]), flat_actions]
        action_log_prob = flat_action_log_prob.reshape(batch_shape)

        return action_log_prob

    def sample_and_log_prob(
        self,
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        logits = self(obs, cmd)
        log_probs = jax.nn.log_softmax(logits)
        sampled_actions = jax.random.categorical(rng, log_probs)
        action_log_prob = log_probs[jnp.arange(log_probs.shape[0]), sampled_actions]
        return sampled_actions, action_log_prob


class ActorCriticAgent(nn.Module):
    """Actor-Critic model."""

    actor_module: ActionModel
    critic_module: nn.Module

    def setup(self) -> None:
        self.returns_std = self.variable(
            "normalization", "returns_std", nn.initializers.ones, key=(), shape=()
        )  # used in downstream algorithm, initialized here for consistency of statistic interface

    @nn.compact
    def normalize_obs(self, obs: FrozenDict[str, Array]) -> FrozenDict[str, Array]:
        """Normalize the observations."""
        obs_mean = {
            obs_name: self.variable(
                "normalization",
                f"obs_mean_{obs_name}",
                nn.initializers.zeros,
                key=(),
                shape=obs_vec.shape[-1],
            )
            for obs_name, obs_vec in obs.items()
        }
        obs_std = {
            obs_name: self.variable(
                "normalization",
                f"obs_std_{obs_name}",
                nn.initializers.ones,
                key=(),
                shape=obs_vec.shape[-1],
            )
            for obs_name, obs_vec in obs.items()
        }
        # note: initialized here once, will be updated in the training loop

        # do normalization on inputs
        normalized_obs_dict = {
            obs_name: (obs_vec - obs_mean[obs_name].value) / obs_std[obs_name].value
            for obs_name, obs_vec in obs.items()
        }

        normalized_obs: FrozenDict[str, Array] = FrozenDict(normalized_obs_dict)

        return normalized_obs

    @nn.compact
    def __call__(
        self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]
    ) -> tuple[Array, Array]:
        """Forward pass of the model."""
        return self.actor(obs, cmd), self.critic(obs, cmd)

    @nn.compact
    def actor(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        """Actor forward pass."""
        # initialize normalization variables if not already done
        normalized_obs = self.normalize_obs(obs)
        return self.actor_module(normalized_obs, cmd)

    @nn.compact
    def critic(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        """Critic forward pass."""
        normalized_obs = self.normalize_obs(obs)
        return self.critic_module(normalized_obs, cmd)

    def actor_calc_log_prob(self, prediction: Array, action: Array) -> Array:
        """Calculate the log probability of the action."""
        return self.actor_module.calc_log_prob(prediction, action)

    @nn.compact
    def actor_sample_and_log_prob(
        self,
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        """Sample and calculate the log probability of the action."""
        normalized_obs = self.normalize_obs(obs)
        return self.actor_module.sample_and_log_prob(normalized_obs, cmd, rng)


def update_actor_critic_normalization(
    variables: PyTree,
    returns: Array,
    returns_norm_alpha: float,
    obs_norm_alpha: float,
    trajectories_dataset: EnvState,
) -> PyTree:
    """Update the normalization parameters for the observations and returns.

    High alpha means more weight is given to the new data.
    """
    # update the returns normalization parameters
    returns_std = jnp.std(returns)
    old_returns_std = variables["normalization"]["returns_std"]
    assert isinstance(old_returns_std, Array)

    variables["normalization"]["returns_std"] = (
        old_returns_std * returns_norm_alpha + returns_std * (1 - returns_norm_alpha)
    )

    # update the observations normalization parameters
    for obs_name, obs_vec in trajectories_dataset.obs.items():
        assert isinstance(obs_vec, Array)
        obs_mean = jnp.mean(obs_vec, axis=tuple(range(obs_vec.ndim - 1)))
        obs_std = jnp.std(obs_vec, axis=tuple(range(obs_vec.ndim - 1)))
        old_obs_mean = variables["normalization"][f"obs_mean_{obs_name}"]
        old_obs_std = variables["normalization"][f"obs_std_{obs_name}"]

        variables["normalization"][f"obs_mean_{obs_name}"] = (
            old_obs_mean * obs_norm_alpha + obs_mean * (1 - obs_norm_alpha)
        )
        variables["normalization"][f"obs_std_{obs_name}"] = (
            old_obs_std * obs_norm_alpha + obs_std * (1 - obs_norm_alpha)
        )

    return variables


def get_batch_shapes(obs: FrozenDict[str, Array]) -> tuple[int, ...]:
    """Get the batch shapes of the observations."""
    batch_shapes = None
    for obs_name, obs_vec in obs.items():
        assert isinstance(obs_vec, Array)
        if batch_shapes is None:
            batch_shapes = obs_vec.shape[:-1]
        else:
            assert (
                batch_shapes == obs_vec.shape[:-1]
            ), f"{obs_name} has batch shape {obs_vec.shape[:-1]} but expected {batch_shapes}"
    assert batch_shapes is not None, "No observations provided"
    return batch_shapes
