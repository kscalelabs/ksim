"""CartPole environment."""

from typing import Callable

import gym
import jax
import jax.numpy as jnp
from brax.envs.base import State as BraxState
from jaxtyping import Array, PRNGKeyArray

from ksim.env.base_env import BaseEnv


class CartPoleEnv(BaseEnv):
    """CartPole environment wrapper to match the BraxState interface."""

    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, rng: PRNGKeyArray) -> BraxState:
        """Reset the environment.

        Args:
            rng: PRNG key.

        Returns:
            BraxState: The initial state of the environment.
        """
        # TODO: probably want to use RNG properly
        obs, info = self.env.reset()

        return BraxState(
            pipeline_state=None,  # CartPole doesn't use pipeline state
            obs={"observations": jnp.array(obs)[None, :]},
            reward=jnp.array(1.0)[None],
            done=jnp.array(False)[None],
            info={"rng": rng, **info},
        )

    def step(self, prev_state: BraxState, action: Array) -> BraxState:
        """Step the environment.

        NOTE: for simplicity, this environment is stateful and doesn't actually use prev_state in
        a functional manner.

        Args:
            prev_state: The previous state of the environment.
            action: The action to take.

        Returns:
            BraxState: The next state of the environment.
        """
        obs, reward, terminated, truncated, info = self.env.step(action.item())
        done = bool(terminated or truncated)
        return BraxState(
            pipeline_state=None,
            obs={"observations": jnp.array(obs)[None, :]},
            reward=jnp.array(reward)[None],
            done=jnp.array(done)[None],
            info={"rng": prev_state.info["rng"], **info},
        )

    def unroll_trajectories(
        self,
        action_fn: Callable[[Array], Array],
        rng: PRNGKeyArray,
        max_trajectory_steps: int,
    ) -> BraxState:
        """Rollout the model for a given number of steps.
        Args:
            model: The model.
            params: The parameters (really a variable dictionary).
            rng: The random key.

        Returns:
            A BraxState containing trajectories with shape (num_steps, ...) in leaves.
        """
        observations = []
        actions = []
        rewards = []
        done = []
        action_log_probs = []

        state = self.reset(rng)
        rng, _ = jax.random.split(rng)
        for _ in range(max_trajectory_steps):
            logits = action_fn(state.obs["observations"])
            assert isinstance(logits, Array)
            sampled_actions = jax.random.categorical(rng, logits)
            # Calculate log probabilities from logits
            log_probs = jax.nn.log_softmax(logits)
            action_log_prob = log_probs[jnp.arange(logits.shape[0]), sampled_actions]

            observations.append(state.obs)
            done.append(state.done)
            actions.append(sampled_actions)
            action_log_probs.append(action_log_prob)
            rewards.append(state.reward)
            # NOTE: need to be careful about when the reward updates... this works for survival
            # related tasks, but not those that directly depend on the action.

            if state.done:
                state = self.reset(rng)
                rng, _ = jax.random.split(rng)
            else:
                state = self.step(state, sampled_actions)

        observations = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *observations)
        actions = jnp.stack(actions)
        rewards = jnp.stack(rewards)
        done = jnp.stack(done)
        action_log_probs = jnp.stack(action_log_probs)

        return BraxState(
            pipeline_state=None,
            obs=observations,
            reward=rewards,
            done=done,
            metrics={},
            info={"rng": rng, "actions": actions, "action_log_probs": action_log_probs},
        )

    @property
    def observation_size(self) -> int:
        return 4

    @property
    def action_size(self) -> int:
        return 1

    @property
    def num_envs(self) -> int:
        return 1
