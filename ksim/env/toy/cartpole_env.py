"""CartPole environment."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from jaxtyping import PRNGKeyArray, PyTree

from ksim.env.base_env import BaseEnv, EnvState
from ksim.env.types import PhysicsData
from ksim.model.formulations import ActorCriticAgent


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

    ########################
    # Implementing the API #
    ########################
    def get_dummy_env_state(self, rng: PRNGKeyArray) -> EnvState:
        """Get a dummy environment state for compilation purposes."""
        return EnvState(
            obs=FrozenDict({"observations": jnp.zeros((1, self.observation_size))}),
            reward=jnp.zeros((1,)),
            done=jnp.array([False]),
            command=FrozenDict({}),
            action=jnp.zeros((self.action_size,)),
            timestep=jnp.array(0.0),
        )

    def reset(self, model: ActorCriticAgent, variables: PyTree, rng: PRNGKeyArray) -> EnvState:
        """Reset the environment and sample an initial action from the model."""
        env_state_0, gym_obs_1 = self.reset_and_give_obs(model, variables, rng)
        return env_state_0

    def step(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
        prev_env_state: EnvState,
        rng: PRNGKeyArray,
        *,
        current_gym_obs: np.ndarray,
    ) -> EnvState:
        """Take a step in the environment."""
        current_env_state, _ = self.step_given_gym_obs(
            model, variables, prev_env_state, rng, current_gym_obs=current_gym_obs
        )
        return current_env_state

    def unroll_trajectories(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
        rng: PRNGKeyArray,
        num_steps: int,
        num_envs: int,
        return_data: bool = False,
    ) -> tuple[EnvState, PhysicsData]:
        """Rollout the model for a given number of steps."""
        assert num_envs == 1, "CartPoleEnv only supports a single environment"
        observations = []
        actions = []
        rewards = []
        done = []

        prev_state, current_obs = self.reset_and_give_obs(
            model=model,
            variables=variables,
            rng=rng,
        )
        for _ in range(num_steps):
            rng = jax.random.split(rng)[0]
            if prev_state.done[0]:
                prev_state, current_obs = self.reset_and_give_obs(
                    model=model,
                    variables=variables,
                    rng=rng,
                )
            else:
                prev_state, current_obs = self.step_given_gym_obs(
                    model=model,
                    variables=variables,
                    prev_env_state=prev_state,
                    rng=rng,
                    current_gym_obs=current_obs,
                )

            observations.append(prev_state.obs)
            done.append(prev_state.done)
            actions.append(prev_state.action)
            rewards.append(prev_state.reward)

        observations = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *observations)
        actions = jnp.stack(actions)
        rewards = jnp.stack(rewards)
        done = jnp.stack(done)

        return EnvState(
            obs=observations,
            reward=rewards,
            done=done,
            command=FrozenDict({}),
            action=actions,
            timestep=jnp.arange(num_steps)[None],
        )

    @property
    def observation_size(self) -> int:
        return 4

    @property
    def action_size(self) -> int:
        return 1

    ####################
    # Helper functions #
    ####################
    def reset_and_give_obs(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
        rng: PRNGKeyArray,
    ) -> tuple[EnvState, np.ndarray]:
        """Reset the environment and return the observation."""
        gym_obs_0, _ = self.env.reset()
        obs_0 = FrozenDict({"observations": jnp.array(gym_obs_0)[None, :]})
        command = FrozenDict({})
        action_0, _ = model.apply(variables, obs_0, command, rng, method="actor_sample_and_log_prob")
        gym_obs_1 = self.env.step(action_0.item())[0]
        env_state_0 = EnvState(
            obs=obs_0,
            reward=jnp.array(1.0)[None],
            done=jnp.array([False]),
            command=command,
            action=action_0,
            timestep=jnp.array(0.0)[None],
        )
        return env_state_0, gym_obs_1

    def step_given_gym_obs(
        self,
        model: ActorCriticAgent,
        variables: PyTree,
        prev_env_state: EnvState,
        rng: PRNGKeyArray,
        *,
        current_gym_obs: np.ndarray,  # following same pattern as mjx.Env
    ) -> tuple[EnvState, np.ndarray]:
        """Take a step in the environment.

        NOTE: for simplicity, this environment is stateful and doesn't actually use prev_state in
        a functional manner.
        """

        obs = FrozenDict({"observations": jnp.array(current_gym_obs)[None, :]})
        command = FrozenDict({})
        action, _ = model.apply(variables, obs, command, rng, method="actor_sample_and_log_prob")

        gym_obs, gym_reward, gym_terminated, gym_truncated, _ = self.env.step(action.item())
        done = bool(gym_terminated or gym_truncated)

        current_env_state = EnvState(
            obs=obs,
            reward=jnp.array(gym_reward)[None],
            done=jnp.array([done]),
            command=command,
            action=action,
            timestep=prev_env_state.timestep + 1.0,
        )

        return current_env_state, gym_obs

    def render_trajectory(self):
        raise NotImplementedError("Not implemented for this environment.")
