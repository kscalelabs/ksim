"""CartPole environment."""

from pathlib import Path
from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.base_env import BaseEnv, EnvState
from ksim.model.formulations import ActorCriticModel


class CartPoleEnv(BaseEnv):
    """CartPole environment wrapper."""

    def __init__(self, render_mode: str | None = None) -> None:
        """Initialize the CartPole environment.

        Args:
            render_mode: The render mode for the environment. Options: 'human', 'rgb_array', None
        """
        self.env = gym.make("CartPole-v1", render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def get_dummy_env_state(self, rng: PRNGKeyArray) -> EnvState:
        """Get a dummy environment state for compilation purposes."""
        return EnvState(
            obs=FrozenDict({"observations": jnp.zeros((1, self.observation_size))}),
            reward=jnp.zeros((1,)),
            done=jnp.array([False]),
            command=FrozenDict({}),
            action=jnp.zeros((self.action_size,)),
            action_log_prob=jnp.array(0.0),
            timestep=jnp.array(0.0),
            rng=rng,
        )

    def reset(self, model: ActorCriticModel, params: PyTree, rng: PRNGKeyArray) -> EnvState:
        """Reset the environment and sample an initial action from the model."""
        obs, info = self.env.reset()
        obs_dict = FrozenDict({"observations": jnp.array(obs)[None, :]})
        command = FrozenDict({})
        action, log_prob = model.apply(params, obs_dict, command, rng, method="actor_sample_and_log_prob")
        return EnvState(
            obs=obs_dict,
            reward=jnp.array(1.0)[None],
            done=jnp.array([False]),
            command=command,
            action=action,
            action_log_prob=log_prob,
            timestep=jnp.array(0.0),
            rng=rng,
        )

    def step(
        self,
        model: ActorCriticModel,
        params: PyTree,
        prev_state: EnvState,
        rng: PRNGKeyArray,
        action_log_prob: Array,
    ) -> EnvState:
        """Take a step in the environment.

        NOTE: for simplicity, this environment is stateful and doesn't actually use prev_state in
        a functional manner.
        """
        # Use the action stored in the previous state to step the gym environment.
        try:
            obs, reward, terminated, truncated, info = self.env.step(prev_state.action.item())
            done = bool(terminated or truncated)  # Convert to Python bool
        except AttributeError as e:
            if "bool8" in str(e):
                # Handle the numpy bool8 error
                obs, reward, terminated, truncated, info = self.env.step(prev_state.action.item())
                # Ensure terminated and truncated are Python booleans
                terminated = bool(terminated)
                truncated = bool(truncated)
                done = terminated or truncated
            else:
                raise

        new_obs = FrozenDict({"observations": jnp.array(obs)[None, :]})
        new_command = FrozenDict({})
        new_action, new_log_prob = model.apply(params, new_obs, new_command, rng, method="actor_sample_and_log_prob")
        assert isinstance(new_log_prob, Array)
        return EnvState(
            obs=new_obs,
            reward=jnp.array(reward)[None],
            done=jnp.array([done]),
            command=new_command,
            action=new_action,
            action_log_prob=new_log_prob,
            timestep=prev_state.timestep + 1.0,
            rng=prev_state.rng,
        )

    def unroll_trajectories(
        self,
        model: ActorCriticModel,
        params: PyTree,
        rng: PRNGKeyArray,
        num_steps: int,
        num_envs: int,
    ) -> EnvState:
        """Rollout the model for a given number of steps."""
        assert num_envs == 1, "CartPoleEnv only supports a single environment"
        observations = []
        actions = []
        rewards = []
        done = []
        action_log_probs = []

        state = self.reset(model, params, rng)
        rng, _ = jax.random.split(rng)
        for _ in range(num_steps):
            observations.append(state.obs)
            done.append(state.done)
            actions.append(state.action)
            rewards.append(state.reward)
            action_log_probs.append(state.action_log_prob)
            # NOTE: need to be careful about when the reward updates... this works for survival
            # related tasks, but not those that directly depend on the action.

            if state.done[0]:
                state = self.reset(model, params, rng)
                rng, _ = jax.random.split(rng)
            else:
                rng, step_rng = jax.random.split(rng)
                state = self.step(model, params, state, step_rng, state.action_log_prob)

        observations = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *observations)
        actions = jnp.stack(actions)
        rewards = jnp.stack(rewards)
        done = jnp.stack(done)
        action_log_probs = jnp.stack(action_log_probs)

        return EnvState(
            obs=observations,
            reward=rewards,
            done=done,
            command=FrozenDict({}),
            action=actions,
            action_log_prob=action_log_probs,
            timestep=jnp.arange(num_steps),
            rng=rng,
        )

    def unroll_trajectories_and_render(
        self,
        model: ActorCriticModel,
        params: PyTree,
        rng: PRNGKeyArray,
        num_steps: int,
        render_dir: Path,
        width: int = 640,
        height: int = 480,
        **kwargs: Any,
    ) -> tuple[list[np.ndarray], EnvState]:
        raise NotImplementedError("CartPoleEnv does not support trajectory saving yet.")

    @property
    def observation_size(self) -> int:
        return 4

    @property
    def action_size(self) -> int:
        return 1
