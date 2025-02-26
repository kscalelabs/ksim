"""Default Humanoid environment from Gymnasium."""

from typing import Any, Callable, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ksim.env.base_env import BaseEnv, EnvState


class DefaultHumanoidEnv(BaseEnv):
    """Humanoid environment wrapper to match the KSim interface."""

    def __init__(self, render_mode: str | None = None) -> None:
        """Initialize the environment.

        Args:
            render_mode: The render mode for the environment.
        """
        self.env = gym.make("Humanoid-v5", render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, rng: PRNGKeyArray) -> EnvState:
        """Reset the environment.

        Args:
            rng: PRNG key.

        Returns:
            EnvState: The initial state of the environment.
        """
        # Use rng properly for seed
        seed = int(jax.random.randint(rng, (), 0, 2**31 - 1))
        obs, info = self.env.reset(seed=seed)

        return EnvState(
            obs={"observations": jnp.array(obs)[None, :]},
            reward=jnp.array(0.0)[None],
            done=jnp.array(False)[None],
            info={"rng": rng, **info},
        )

    def step(self, prev_state: EnvState, action: Array) -> EnvState:
        """Step the environment.

        Args:
            prev_state: The previous state of the environment.
            action: The action to take.

        Returns:
            EnvState: The next state of the environment.
        """
        # Convert JAX array to NumPy for Gymnasium
        np_action = jax.device_get(action[0])

        try:
            obs, reward, terminated, truncated, info = self.env.step(np_action)
            done = terminated or truncated
        except Exception as e:
            print(f"Error stepping environment: {e}")
            obs = jnp.zeros_like(prev_state.obs["observations"][0])
            reward = 0.0
            done = True
            info = {}

        return EnvState(
            obs={"observations": jnp.array(obs)[None, :]},
            reward=jnp.array(reward)[None],
            done=jnp.array(done)[None],
            info={"rng": prev_state.info["rng"], **info},
        )

    def unroll_trajectories(
        self,
        action_log_prob_fn: Callable[[EnvState, PRNGKeyArray], Tuple[Array, Array]],
        rng: PRNGKeyArray,
        num_steps: int,
        num_envs: int,
        **kwargs: Any,
    ) -> EnvState:
        """Rollout the model for a given number of steps.

        Args:
            action_log_prob_fn: Function to get actions and log probs from states.
            rng: The random key.
            num_steps: Number of steps to roll out.
            num_envs: Number of environments to run in parallel.

        Returns:
            A EnvState containing trajectories with shape (num_steps, ...) in leaves.
        """
        assert num_envs == 1, "DefaultHumanoidEnv only supports a single environment"
        observations = []
        actions = []
        rewards = []
        done = []
        action_log_probs = []

        state = self.reset(rng)
        rng, _ = jax.random.split(rng)

        for _ in range(num_steps):
            rng, action_rng = jax.random.split(rng)
            action, log_probs = action_log_prob_fn(state, action_rng)

            observations.append(state.obs)
            done.append(state.done)
            actions.append(action)
            action_log_probs.append(log_probs)
            rewards.append(state.reward)

            if state.done:
                state = self.reset(rng)
                rng, _ = jax.random.split(rng)
            else:
                state = self.step(state, action)

        observations = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *observations)
        actions = jnp.stack(actions)
        rewards = jnp.stack(rewards)
        done = jnp.stack(done)
        action_log_probs = jnp.stack(action_log_probs)

        return EnvState(
            obs=observations,
            reward=rewards,
            done=done,
            info={
                "actions": actions,
                "action_log_probs": action_log_probs,
            },
        )
    
    def run_visualization(self) -> None:
        """Run the environment in visualization mode."""


    @property
    def observation_size(self) -> int:
        """Return the observation space size."""
        return self.observation_space.shape[0]

    @property
    def action_size(self) -> int:
        """Return the action space size."""
        return self.action_space.shape[0]
