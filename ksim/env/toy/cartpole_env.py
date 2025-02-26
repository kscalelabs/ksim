"""CartPole environment."""

from typing import Any, Callable, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ksim.env.base_env import BaseEnv, EnvState


class CartPoleEnv(BaseEnv):
    """CartPole environment wrapper to match the EnvState interface."""

    def __init__(self, render_mode: str | None = None) -> None:
        """Initialize the CartPole environment.

        Args:
            render_mode: The render mode for the environment. Options: 'human', 'rgb_array', None
        """
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, rng: PRNGKeyArray) -> EnvState:
        """Reset the environment."""
        # TODO: probably want to use RNG properly
        obs, info = self.env.reset()

        return EnvState(
            obs={"observations": jnp.array(obs)[None, :]},
            reward=jnp.array(1.0)[None],
            done=jnp.array(False)[None],
            info={"rng": rng, **info},
        )

    def step(self, prev_state: EnvState, action: Array, rng: PRNGKeyArray) -> EnvState:
        """Step the environment.

        NOTE: for simplicity, this environment is stateful and doesn't actually use prev_state in
        a functional manner.
        """
        try:
            obs, reward, terminated, truncated, info = self.env.step(action.item())
            done = bool(terminated or truncated)  # Convert to Python bool
        except AttributeError as e:
            if "bool8" in str(e):
                # Handle the numpy bool8 error
                obs, reward, terminated, truncated, info = self.env.step(action.item())
                # Ensure terminated and truncated are Python booleans
                terminated = bool(terminated)
                truncated = bool(truncated)
                done = terminated or truncated
            else:
                raise

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
        """Rollout the model for a given number of steps."""
        assert num_envs == 1, "CartPoleEnv only supports a single environment"
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
            # NOTE: need to be careful about when the reward updates... this works for survival
            # related tasks, but not those that directly depend on the action.

            if state.done:
                state = self.reset(rng)
                rng, _ = jax.random.split(rng)
            else:
                rng, step_rng = jax.random.split(rng)
                state = self.step(state, action, step_rng)

        observations = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *observations)
        actions = jnp.stack(actions)
        rewards = jnp.stack(rewards)
        done = jnp.stack(done)
        action_log_probs = jnp.stack(action_log_probs)

        return EnvState(
            obs=observations,
            reward=rewards,
            done=done,
            info={"rng": rng, "actions": actions, "action_log_probs": action_log_probs},
        )

    @property
    def observation_size(self) -> int:
        return 4

    @property
    def action_size(self) -> int:
        return 1
