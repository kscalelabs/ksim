"""CartPole environment."""

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree

from ksim.env.base_env import BaseEnv, BaseEnvConfig, EnvState
from ksim.env.types import PhysicsData, PhysicsModel
from ksim.model.base import ActorCriticAgent


class CartPoleEnv(BaseEnv):
    """CartPole environment wrapper.

    TODO: Incorporate simulation frequency (dt) and policy frequency (ctrl_dt) parameters
    from BaseEnvConfig into this environment.
    """

    def __init__(self, config: BaseEnvConfig) -> None:
        """Initialize the CartPole environment."""
        super().__init__(config)
        self.env = gym.make("CartPole-v1", render_mode=None)  # render mode handled separately
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Empty rewards and terminations lists since CartPole handles these internally
        self.rewards = []
        self.terminations = []

    ########################
    # Implementing the API #
    ########################
    def get_init_physics_data(self, num_envs: int) -> PhysicsData:
        """CartPole doesn't use physics data."""
        return None

    def get_init_physics_model(self) -> None:
        """CartPole doesn't use physics model."""
        return None

    def get_dummy_env_states(self, num_envs: int) -> EnvState:
        """Get dummy environment states for compilation."""
        return EnvState(
            obs=FrozenDict({"observations": jnp.zeros((num_envs, self.observation_size))}),
            reward=jnp.zeros((num_envs,)),
            done=jnp.zeros((num_envs,), dtype=bool),
            command=FrozenDict({}),
            action=jnp.zeros((num_envs, self.action_size)),
            timestep=jnp.zeros((num_envs,)),
            termination_components=FrozenDict({}),
            reward_components=FrozenDict({}),
        )

    def reset(
        self,
        model: ActorCriticAgent,
        variables: PyTree[Array],
        rng: PRNGKeyArray,
        physics_model_L: PhysicsModel,
    ) -> tuple[EnvState, PhysicsData | None]:
        """Reset the environment."""
        env_state_0, _ = self.reset_and_give_obs(model, variables, rng)
        return env_state_0, None

    def step(
        self,
        model: ActorCriticAgent,
        variables: PyTree[Array],
        env_state_L_t_minus_1: EnvState,
        rng: PRNGKeyArray,
        physics_data_L_t: PhysicsData,
        physics_model_L: PhysicsModel,
    ) -> tuple[EnvState, PhysicsData | None]:
        """Take a step in the environment."""
        current_env_state, _ = self.step_given_gym_obs(model, variables, env_state_L_t_minus_1, rng)
        return current_env_state, None

    def unroll_trajectories(
        self,
        model: ActorCriticAgent,
        variables: PyTree[Array],
        rng: PRNGKeyArray,
        num_steps: int,
        num_envs: int,
        env_state_EL_t_minus_1: EnvState | None = None,
        physics_data_EL_t: PhysicsData | None = None,
        physics_model_L: PhysicsModel | None = None,
        return_intermediate_data: bool = False,
    ) -> tuple[EnvState, PhysicsData, Array]:
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

        return (
            EnvState(
                obs=observations,
                reward=rewards,
                done=done,
                command=FrozenDict({}),
                action=actions,
                timestep=jnp.arange(num_steps)[None],
                termination_components=FrozenDict({}),
                reward_components=FrozenDict({}),
            ),
            None,
            jnp.array(False),
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
        variables: PyTree[Array],
        rng: PRNGKeyArray,
    ) -> tuple[EnvState, np.ndarray]:
        """Reset the environment and return the observation."""
        gym_obs_0, _ = self.env.reset()
        obs_0: FrozenDict[str, jax.Array] = FrozenDict(
            {"observations": jnp.array(gym_obs_0)[None, :]}
        )
        command: FrozenDict[str, jax.Array] = FrozenDict({})
        action_0, _ = model.apply_actor_sample_and_log_prob(
            variables=variables,
            obs=obs_0,
            cmd=command,
            prev_action=None,
            prev_model_input=None,
            recurrent_state=None,
            rng=rng,
        )
        gym_obs_1 = self.env.step(action_0.item())[0]
        env_state_0 = EnvState(
            obs=obs_0,
            reward=jnp.array(1.0)[None],
            done=jnp.array([False]),
            command=command,
            action=action_0,
            timestep=jnp.array(0.0)[None],
            termination_components=FrozenDict({}),
            reward_components=FrozenDict({}),
        )
        return env_state_0, gym_obs_1

    def step_given_gym_obs(
        self,
        model: ActorCriticAgent,
        variables: PyTree[Array],
        prev_env_state: EnvState,
        rng: PRNGKeyArray,
        *,
        current_gym_obs: np.ndarray | None = None,  # following same pattern as mjx.Env
    ) -> tuple[EnvState, np.ndarray]:
        """Take a step in the environment.

        NOTE: for simplicity, this environment is stateful and doesn't actually use prev_state in
        a functional manner.
        """
        obs: FrozenDict[str, jax.Array] = FrozenDict(
            {"observations": jnp.array(current_gym_obs)[None, :]}
        )
        command: FrozenDict[str, jax.Array] = FrozenDict({})
        action, _ = model.apply_actor_sample_and_log_prob(
            variables=variables,
            obs=obs,
            cmd=command,
            prev_action=prev_env_state.action,
            prev_model_input=None,
            recurrent_state=None,
            rng=rng,
        )

        gym_obs, gym_reward, gym_terminated, gym_truncated, _ = self.env.step(action.item())
        done = bool(gym_terminated or gym_truncated)

        current_env_state = EnvState(
            obs=obs,
            reward=jnp.array(gym_reward)[None],
            done=jnp.array([done]),
            command=command,
            action=action,
            timestep=prev_env_state.timestep + 1.0,
            termination_components=FrozenDict({}),
            reward_components=FrozenDict({}),
        )

        return current_env_state, gym_obs

    def render_trajectory(
        self,
        model: ActorCriticAgent,
        variables: PyTree[Array],
        rng: PRNGKeyArray,
        *,
        num_steps: int,
        width: int = 640,
        height: int = 480,
        camera: int | None = None,
    ) -> tuple[list[np.ndarray], EnvState]:
        raise NotImplementedError("Not implemented for this environment.")
