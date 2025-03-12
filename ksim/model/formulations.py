"""High Level Formulations of RL Models."""

from abc import abstractmethod

import flax.linen as nn
import jax.numpy as jnp
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.types import EnvState
from ksim.model.distributions import ActionDistribution
from ksim.task.loss_helpers import compute_returns
from ksim.utils.constants import EPSILON


class ActorModel(nn.Module):
    """Actor model."""

    @abstractmethod
    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        """Forward pass of the actor model."""
        ...


class ActorCriticAgent(nn.Module):
    """Actor-Critic model."""

    actor_module: ActorModel
    critic_module: nn.Module
    distribution: ActionDistribution

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
            obs_name: (obs_vec - obs_mean[obs_name].value) / (obs_std[obs_name].value + EPSILON)
            for obs_name, obs_vec in obs.items()
        }

        normalized_obs: FrozenDict[str, Array] = FrozenDict(normalized_obs_dict)

        return normalized_obs

    def actor_obs(self, obs: FrozenDict[str, Array]) -> FrozenDict[str, Array]:
        """Transform the observations for the actor.

        NOTE: override this if you need flexibility."""
        return obs

    def critic_obs(self, obs: FrozenDict[str, Array]) -> FrozenDict[str, Array]:
        """Transform the observations for the critic.

        NOTE: override this if you need flexibility."""
        return obs

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
        transformed_obs = self.actor_obs(obs)
        normalized_obs = self.normalize_obs(transformed_obs)
        return self.actor_module(normalized_obs, cmd)

    @nn.compact
    def critic(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        """Critic forward pass."""
        transformed_obs = self.critic_obs(obs)
        normalized_obs = self.normalize_obs(transformed_obs)
        return self.critic_module(normalized_obs, cmd)

    def actor_calc_log_prob(self, prediction: Array, action: Array) -> Array:
        """Calculate the log probability of the action."""
        return self.distribution.log_prob(prediction, action)

    @nn.compact
    def actor_sample_and_log_prob(
        self,
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        """Sample and calculate the log probability of the action."""
        transformed_obs = self.actor_obs(obs)
        normalized_obs = self.normalize_obs(transformed_obs)
        distribution_params = self.actor_module(normalized_obs, cmd)
        sample = self.distribution.sample(distribution_params, rng)
        log_prob = self.actor_calc_log_prob(distribution_params, sample)
        return sample, log_prob


def update_actor_critic_normalization(
    variables: PyTree,
    returns_norm_alpha: float,
    obs_norm_alpha: float,
    trajectories_dataset: EnvState,
    gamma: float,
) -> PyTree:
    """Update the normalization parameters for the observations and returns.

    High alpha means more weight is given to the old data.
    """
    # update the returns normalization parameters
    returns = compute_returns(trajectories_dataset.reward, trajectories_dataset.done, gamma)
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
            err_msg = f"{obs_name} has batch shape {obs_vec.shape[:-1]} but expected {batch_shapes}"
            assert batch_shapes == obs_vec.shape[:-1], err_msg
    assert batch_shapes is not None, "No observations provided"
    return batch_shapes
