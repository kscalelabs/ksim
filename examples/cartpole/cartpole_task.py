"""Cartpole example task."""

import logging
import time
from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

from ksim.env.toy.cartpole_env import CartPoleEnv
from ksim.model.formulations import ActionModel, ActorCriticModel
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask

logger = logging.getLogger(__name__)


class CartPoleActionModel(ActionModel):
    """Action model for CartPole."""

    mlp: MLP

    @nn.compact
    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        observation = obs["observations"]
        assert isinstance(observation, Array)
        return self.mlp(observation)

    def calc_log_prob(self, prediction: Array, action: Array) -> Array:
        logits = prediction
        log_probs = jax.nn.log_softmax(logits)
        action_log_prob = log_probs[
            jnp.arange(log_probs.shape[0])[:, None], jnp.arange(log_probs.shape[1]), action
        ]
        # NOTE: assumes two batching dimensions
        return action_log_prob

    def sample_and_log_prob(
        self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array], rng: PRNGKeyArray
    ) -> tuple[Array, Array]:
        logits = self(obs, cmd)
        log_probs = jax.nn.log_softmax(logits)
        sampled_actions = jax.random.categorical(rng, log_probs)
        action_log_prob = log_probs[jnp.arange(log_probs.shape[0]), sampled_actions]
        return sampled_actions, action_log_prob


class CartPoleCriticModel(nn.Module):
    """Critic model for CartPole."""

    mlp: MLP

    @nn.compact
    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        observation = obs["observations"]
        assert isinstance(observation, Array)
        return self.mlp(observation)


@dataclass
class CartPoleConfig(PPOConfig):
    """Configuration for CartPole training."""

    # ML model parameters.
    actor_hidden_dims: int = xax.field(value=128, help="Hidden dimensions for the actor.")
    actor_num_layers: int = xax.field(value=2, help="Number of layers for the actor.")
    critic_hidden_dims: int = xax.field(value=128, help="Hidden dimensions for the critic.")
    critic_num_layers: int = xax.field(value=2, help="Number of layers for the critic.")

    # Environment parameters
    render_mode: str | None = xax.field(
        value=None, help="Render mode for the environment. Options: 'human', 'rgb_array', None"
    )
    observation_size: int = 4
    action_size: int = 1


class CartPoleTask(PPOTask[CartPoleConfig]):
    """Task for CartPole training."""

    def get_environment(self) -> CartPoleEnv:
        """Get the environment."""
        return CartPoleEnv(render_mode=self.config.render_mode)

    def get_model(self, key: PRNGKeyArray) -> ActorCriticModel:
        """Get the model."""
        return ActorCriticModel(
            actor_module=CartPoleActionModel(
                mlp=MLP(
                    num_hidden_layers=self.config.actor_num_layers,
                    hidden_features=self.config.actor_hidden_dims,
                    out_features=2,  # two discrete actions for CartPole
                ),
            ),
            critic_module=CartPoleCriticModel(
                mlp=MLP(
                    num_hidden_layers=self.config.critic_num_layers,
                    hidden_features=self.config.critic_hidden_dims,
                    out_features=1,
                ),
            ),
        )

    def viz_environment(
        self,
    ) -> None:
        """Run the environment with visualization.

        Uses trained policy from latest checkpoint, otherwise uses a randomly initialized policy.
        """
        rng = self.prng_key()
        env = self.get_environment()
        model = self.get_model(rng)
        params = None

        # Load checkpoint if available, otherwise use random initialization
        ckpt_path = self.get_ckpt_path()
        if ckpt_path.exists():
            try:
                logger.info("Loading checkpoint: %s", ckpt_path)
                params = self.load_checkpoint(ckpt_path, part="model")
            except Exception as e:
                logger.error("Failed loading checkpoint: %s", e)

        if params is None:
            logger.warning("Using randomly initialized policy")
            params = self.get_init_params(rng)

        episode_count = 0
        try:
            while True:  # Keep running episodes until interrupted
                logger.info("Starting Episode %d", episode_count)
                total_reward = 0
                env_state = env.reset(rng)
                episode_length = 0

                while True:
                    # Get observations and use policy
                    rng, action_rng = jax.random.split(rng)
                    action, log_prob = model.apply(
                        params, env_state, action_rng, method="actor_sample_and_log_prob"
                    )
                    assert isinstance(log_prob, Array)

                    # Take step
                    rng, step_rng = jax.random.split(rng)
                    env_state = env.step(env_state, action, step_rng, log_prob)
                    reward = env_state.reward.item()
                    done = env_state.done.item()
                    total_reward += reward

                    episode_length += 1

                    if done:
                        logger.info(
                            "Episode %d finished after %d steps with total reward: %f",
                            episode_count,
                            episode_length,
                            total_reward,
                        )
                        episode_count += 1
                        time.sleep(1.0)  # Pause briefly between episodes
                        break

        except KeyboardInterrupt:
            logger.info("Stopping episodes - cleaning up...")
        finally:
            env.env.close()


if __name__ == "__main__":
    # python -m examples.cartpole.cartpole_task action=train
    # python -m examples.cartpole.cartpole_task action=viz
    CartPoleTask.launch(
        CartPoleConfig(
            num_envs=1,
            max_trajectory_seconds=10.0,
            valid_every_n_steps=5,
            learning_rate=3e-4,
        ),
    )
