"""Cartpole example task."""

import logging
import time
from dataclasses import dataclass

import flax.linen as nn
import jax
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

from examples.cartpole.cartpole_env import CartPoleEnv
from ksim.model.formulations import ActorCriticAgent, CategoricalActionModel
from ksim.model.mlp import MLP
from ksim.task.ppo import PPOConfig, PPOTask

logger = logging.getLogger(__name__)


class CartPoleActionModel(CategoricalActionModel):
    """Action model for CartPole."""

    mlp: MLP

    @nn.compact
    def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
        observation = obs["observations"]
        assert isinstance(observation, Array)
        return self.mlp(observation)


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
        value=None,
        help="Render mode for the environment. Options: 'human', 'rgb_array', None",
    )


class CartPoleTask(PPOTask[CartPoleConfig]):
    """Task for CartPole training."""

    def get_environment(self) -> CartPoleEnv:
        """Get the environment."""
        return CartPoleEnv(render_mode=self.config.render_mode)

    def get_model(self, key: PRNGKeyArray) -> ActorCriticAgent:
        """Get the model."""
        return ActorCriticAgent(
            actor_module=CartPoleActionModel(
                mlp=MLP(
                    num_hidden_layers=self.config.actor_num_layers,
                    hidden_features=self.config.actor_hidden_dims,
                    out_features=2,  # two discrete actions for CartPole
                ),
                num_outputs=2,
                sampling_temperature=0.0,
            ),
            critic_module=CartPoleCriticModel(
                mlp=MLP(
                    num_hidden_layers=self.config.critic_num_layers,
                    hidden_features=self.config.critic_hidden_dims,
                    out_features=1,
                ),
            ),
        )

    def viz_environment(self) -> None:
        """Run the environment with visualization.

        Uses trained policy from latest checkpoint, otherwise uses a randomly initialized policy.
        """
        rng = self.prng_key()
        env = self.get_environment()
        model = self.get_model(rng)
        variables = None

        # Load checkpoint if available, otherwise use random initialization
        ckpt_path = self.get_ckpt_path()
        if ckpt_path.exists():
            try:
                logger.info("Loading checkpoint: %s", ckpt_path)
                variables = self.load_checkpoint(ckpt_path, part="model")
            except Exception as e:
                logger.error("Failed loading checkpoint: %s", e)

        if variables is None:
            logger.warning("Using randomly initialized policy")
            variables = self.get_init_variables(rng)

        episode_count = 0
        try:
            while True:  # Keep running episodes until interrupted
                logger.info("Starting Episode %d", episode_count)
                total_reward = 0
                env_state = env.reset(model, variables, rng)
                episode_length = 0

                while True:
                    # Get observations and use policy
                    rng, action_rng = jax.random.split(rng)
                    _, log_prob = model.apply(
                        variables,
                        env_state,
                        action_rng,
                        method="actor_sample_and_log_prob",
                    )
                    assert isinstance(log_prob, Array)

                    # Take step
                    rng, step_rng = jax.random.split(rng)
                    env_state = env.step(
                        model, variables, env_state, step_rng, current_gym_obs=None
                    )
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
    # python -m examples.cartpole.cartpole_task action=env
    CartPoleTask.launch(
        CartPoleConfig(
            num_envs=1,
            num_steps_per_trajectory=500,
            valid_every_n_steps=5,
            minibatch_size=100,
            learning_rate=1e-3,
        ),
    )
