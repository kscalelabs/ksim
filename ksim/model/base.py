"""High Level Formulations of RL Models.

To define a new model:
1) implement a KSimModel subclass or inherit from `actors.py` to populate
   `forward` method.
2) inherit either ProprioOnlyModel | ProprioCommandModel ... to populate the
   `get_input` method.
"""

from abc import abstractmethod

import flax.linen as nn
import jax.numpy as jnp
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree
from omegaconf import MISSING

from ksim.env.types import EnvState
from ksim.model.distributions import ActionDistribution
from ksim.model.types import ModelInput
from ksim.task.loss_helpers import compute_returns
from ksim.utils.constants import EPSILON


class KSimModel(nn.Module):
    """Base class for all KSim models, helps with observation selection.

    You will likely want to mix and match input formulations and forward
    methodologies. As such, the pattern is have your downstream model inherit
    from a `KSimInputMixin` class that provides the `get_input` method as well
    as a `KSimForwardMixin` class that provides the `forward` method.
    """

    obs_names: tuple[str, ...] | None = xax.field(value=None)
    """List of observation names accessible to the model."""

    distribution: ActionDistribution = xax.field(value=MISSING)
    """Distribution to use for the actor model."""

    # TODO: implement means of specifying history length

    ################
    # Abstract API #
    ################

    @abstractmethod
    def forward(self, x: ModelInput) -> Array:
        """Forward pass of the model, returns action distribution parameters."""
        ...

    @abstractmethod
    def post_process(self, prediction: Array) -> Array:
        """Post-process the output of the network."""
        ...

    ########################
    # Base Implementations #
    ########################

    def get_input(
        self,
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
        prev_action: Array | None = None,
        prev_model_input: ModelInput | None = None,
        recurrent_state: Array | FrozenDict[str, Array] | None = None,
    ) -> ModelInput:
        """Get the input vector, tensors, or pytrees for the model."""
        obs_proprios = []
        obs_images = []
        for obs_name, obs_value in obs.items():
            if "proprio" in obs_name:
                obs_proprios.append(obs_value)
            elif "image" in obs_name:
                obs_images.append(obs_value)

        cmd_vecs = []
        cmd_tokens = []
        for cmd_name, cmd_value in cmd.items():
            if "vector" in cmd_name:
                cmd_vecs.append(cmd_value)
            elif "text" in cmd_name:
                cmd_tokens.append(cmd_value)

        # TODO: implement way of adding history to the input

        return ModelInput(
            obs_proprio_vec=(
                jnp.concatenate(obs_proprios, axis=-1) if obs_proprios else jnp.zeros(0)
            ),
            obs_image_tensor=jnp.concatenate(obs_images, axis=-1) if obs_images else None,
            command_vec=jnp.concatenate(cmd_vecs, axis=-1) if cmd_vecs else jnp.zeros(0),
            command_text_tokens=jnp.concatenate(cmd_tokens, axis=-1) if cmd_tokens else None,
            action_history_vec=prev_action,
            recurrent_state=recurrent_state,
        )

    def __call__(
        self,
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
        prev_action: Array | None,
        prev_model_input: ModelInput | None,
        recurrent_state: Array | None,
    ) -> Array:
        """Forward pass of the model, returns action distribution parameters."""
        model_inputs = self.get_input(obs, cmd, prev_action, prev_model_input, recurrent_state)
        prediction = self.forward(model_inputs)
        distribution_parametrization = self.post_process(prediction)
        return distribution_parametrization


class ActorCriticAgent(nn.Module):
    """Actor-Critic model."""

    actor_module: KSimModel
    critic_module: KSimModel
    distribution: ActionDistribution

    def setup(self) -> None:
        self.returns_std = self.variable(
            "normalization", "returns_std", nn.initializers.ones, key=(), shape=()
        )  # used in downstream algorithm, initialized here for consistency of statistic interface

    @nn.compact
    def normalize_obs(self, obs: FrozenDict[str, Array]) -> FrozenDict[str, Array]:
        """Normalize the observations."""
        # getting or creating the normalization variables
        obs_mean = {}
        for obs_name, obs_vec in obs.items():
            assert isinstance(obs_vec, Array)
            obs_mean[obs_name] = self.variable(
                "normalization",
                f"obs_mean_{obs_name}",
                nn.initializers.zeros,
                key=(),
                shape=obs_vec.shape[-1],
            )

        obs_std = {}
        for obs_name, obs_vec in obs.items():
            assert isinstance(obs_vec, Array)
            obs_std[obs_name] = self.variable(
                "normalization",
                f"obs_std_{obs_name}",
                nn.initializers.ones,
                key=(),
                shape=obs_vec.shape[-1],
            )

        # do normalization on inputs
        normalized_obs_dict = {}
        for obs_name, obs_vec in obs.items():
            mean = obs_mean[obs_name].value
            std = obs_std[obs_name].value
            assert isinstance(obs_vec, Array)
            assert isinstance(mean, Array)
            assert isinstance(std, Array)
            normalized_obs_dict[obs_name] = (obs_vec - mean) / (std + EPSILON)

        normalized_obs: FrozenDict[str, Array] = FrozenDict(normalized_obs_dict)

        return normalized_obs

    @nn.compact
    def __call__(
        self,
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
        prev_action: Array | None,
        prev_model_input: ModelInput | None,
        recurrent_state: Array | None,
    ) -> tuple[Array, Array]:
        """Forward pass of the model."""
        return (
            self.actor(obs, cmd, prev_action, prev_model_input, recurrent_state),
            self.critic(obs, cmd, prev_action, prev_model_input, recurrent_state),
        )

    @nn.compact
    def actor(
        self,
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
        prev_action: Array | None,
        prev_model_input: ModelInput | None,
        recurrent_state: Array | None,
    ) -> Array:
        """Actor forward pass."""
        normalized_obs = self.normalize_obs(obs)
        return self.actor_module(
            normalized_obs, cmd, prev_action, prev_model_input, recurrent_state
        )

    @nn.compact
    def critic(
        self,
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
        prev_action: Array | None,
        prev_model_input: ModelInput | None,
        recurrent_state: Array | None,
    ) -> Array:
        """Critic forward pass."""
        normalized_obs = self.normalize_obs(obs)
        return self.critic_module(
            normalized_obs, cmd, prev_action, prev_model_input, recurrent_state
        )

    def actor_calc_log_prob(self, prediction: Array, action: Array) -> Array:
        """Calculate the log probability of the action."""
        return self.distribution.log_prob(prediction, action)

    @nn.compact
    def actor_sample_and_log_prob(
        self,
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
        prev_action: Array | None,
        prev_model_input: ModelInput | None,
        recurrent_state: Array | None,
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        """Sample and calculate the log probability of the action."""
        normalized_obs = self.normalize_obs(obs)
        distribution_params = self.actor_module(
            normalized_obs, cmd, prev_action, prev_model_input, recurrent_state
        )
        sample = self.distribution.sample(distribution_params, rng)
        log_prob = self.actor_calc_log_prob(distribution_params, sample)
        return sample, log_prob

    def apply_actor_calc_log_prob(
        self,
        variables: PyTree[Array],
        prediction: Array,
        action: Array,
    ) -> Array:
        """Apply the actor calc log prob."""
        res = self.apply(variables, prediction, action, method="actor_calc_log_prob")
        assert isinstance(res, Array)
        return res

    def apply_actor_sample_and_log_prob(
        self,
        variables: PyTree[Array],
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
        prev_action: Array | None,
        prev_model_input: ModelInput | None,
        recurrent_state: Array | None,
        rng: PRNGKeyArray,
    ) -> tuple[Array, Array]:
        """Apply the actor sample and log prob."""
        sample, log_prob = self.apply(
            variables,
            obs,
            cmd,
            prev_action,
            prev_model_input,
            recurrent_state,
            rng,
            method="actor_sample_and_log_prob",
        )
        assert isinstance(sample, Array)
        assert isinstance(log_prob, Array)
        return sample, log_prob

    def apply_actor(
        self,
        variables: PyTree[Array],
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
        prev_action: Array | None,
        prev_model_input: ModelInput | None,
        recurrent_state: Array | None,
    ) -> Array:
        """Apply the actor."""
        res = self.apply(
            variables,
            obs,
            cmd,
            prev_action,
            prev_model_input,
            recurrent_state,
            method="actor",
        )
        assert isinstance(res, Array)
        return res

    def apply_critic(
        self,
        variables: PyTree[Array],
        obs: FrozenDict[str, Array],
        cmd: FrozenDict[str, Array],
        prev_action: Array | None,
        prev_model_input: ModelInput | None,
        recurrent_state: Array | None,
    ) -> Array:
        """Apply the critic."""
        res = self.apply(
            variables,
            obs,
            cmd,
            prev_action,
            prev_model_input,
            recurrent_state,
            method="critic",
        )
        assert isinstance(res, Array)
        return res


def update_actor_critic_normalization(
    variables: PyTree[Array],
    returns_norm_alpha: float,
    obs_norm_alpha: float,
    trajectories_dataset: EnvState,
    gamma: float,
) -> PyTree[Array]:
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
