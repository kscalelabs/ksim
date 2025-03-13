"""Cartpole example task."""

import logging
import time
from dataclasses import dataclass

import jax
import xax
from jaxtyping import PRNGKeyArray

from examples.cartpole.cartpole_env import CartPoleEnv
from ksim.model.base import ActorCriticAgent
from ksim.model.factory import mlp_actor_critic_agent
from ksim.task.ppo import PPOConfig, PPOTask

logger = logging.getLogger(__name__)


@dataclass
class CartPoleConfig(PPOConfig):
    """Configuration for CartPole training."""

    # ML model parameters.
    actor_hidden_dims: int = xax.field(value=128, help="Hidden dimensions for the actor.")
    actor_num_layers: int = xax.field(value=2, help="Number of layers for the actor.")
    critic_hidden_dims: int = xax.field(value=128, help="Hidden dimensions for the critic.")
    critic_num_layers: int = xax.field(value=2, help="Number of layers for the critic.")


class CartPoleTask(PPOTask[CartPoleConfig]):
    """Task for CartPole training."""

    def get_environment(self) -> CartPoleEnv:
        """Get the environment."""
        return CartPoleEnv(config=self.config)

    def get_model(self, key: PRNGKeyArray) -> ActorCriticAgent:
        """Get the model."""
        return mlp_actor_critic_agent(
            num_actions=2,
            prediction_type="direct",
            distribution_type="categorical",
            actor_hidden_dims=(self.config.actor_hidden_dims,) * self.config.actor_num_layers,
            critic_hidden_dims=(self.config.critic_hidden_dims,) * self.config.critic_num_layers,
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
                env_state, _ = env.reset(model, variables, rng, physics_model_L=None)
                episode_length = 0

                while True:
                    # Take step
                    rng, step_rng = jax.random.split(rng)
                    env_state, _ = env.step(
                        model,
                        variables,
                        env_state,
                        step_rng,
                        physics_data_L_t=None,
                        physics_model_L=None,
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
            num_learning_epochs=1,
            num_env_states_per_minibatch=100,
            num_minibatches=1,
            learning_rate=1e-3,
        ),
    )
