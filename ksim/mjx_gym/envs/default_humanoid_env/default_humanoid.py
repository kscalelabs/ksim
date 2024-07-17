"""Defines the default humanoid environment."""

from typing import NotRequired, TypedDict, Unpack

import jax
import jax.numpy as jp
import mujoco
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from brax.mjx.base import State as mjxState
from etils import epath

from ksim.mjx_gym.envs.default_humanoid_env.rewards import (
    DEFAULT_REWARD_PARAMS,
    RewardParams,
    get_reward_fn,
)


class EnvKwargs(TypedDict):
    sys: base.System
    backend: NotRequired[str]
    n_frames: NotRequired[int]
    debug: NotRequired[bool]


class DefaultHumanoidEnv(PipelineEnv):
    """An environment for humanoid body position, velocities, and angles.

    Note: This environment is based on the default humanoid environment in the Brax library.
    https://github.com/google/brax/blob/main/brax/envs/humanoid.py

    However, this environment is designed to work with modular reward functions, allowing for quicker experimentation.
    """

    def __init__(
        self,
        reward_params: RewardParams = DEFAULT_REWARD_PARAMS,
        terminate_when_unhealthy: bool = True,
        reset_noise_scale: float = 1e-2,
        exclude_current_positions_from_observation: bool = True,
        log_reward_breakdown: bool = True,
        **kwargs: Unpack[EnvKwargs],
    ) -> None:
        path = epath.Path(epath.resource_path("mujoco")) / ("mjx/test_data/humanoid")
        mj_model = mujoco.MjModel.from_xml_path((path / "humanoid.xml").as_posix())
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        mj_model.opt.iterations = 6
        mj_model.opt.ls_iterations = 6

        sys = mjcf.load_model(mj_model)

        physics_steps_per_control_step = 4  # Should find way to perturb this value in the future
        kwargs["n_frames"] = kwargs.get("n_frames", physics_steps_per_control_step)
        kwargs["backend"] = "mjx"

        super().__init__(sys, **kwargs)

        self._reward_params = reward_params
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation
        self._log_reward_breakdown = log_reward_breakdown

        self.reward_fn = get_reward_fn(self._reward_params, self.dt, include_reward_breakdown=True)

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state.

        Args:
            rng: Random number generator seed.

        Returns:
            The initial state of the environment.
        """
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)

        mjx_state = self.pipeline_init(qpos, qvel)
        assert isinstance(mjx_state, mjxState), f"mjx_state is of type {type(mjx_state)}"

        obs = self._get_obs(mjx_state, jp.zeros(self.sys.nu))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
        }
        for key in self._reward_params.keys():
            metrics[key] = zero

        return State(mjx_state, obs, reward, done, metrics)

    def step(self, state: State, action: jp.ndarray) -> State:
        """Runs one timestep of the environment's dynamics.

        Args:
            state: The current state of the environment.
            action: The action to take.

        Returns:
            A tuple of the next state, the reward, whether the episode has ended, and additional information.
        """
        mjx_state = state.pipeline_state
        assert mjx_state, "state.pipeline_state was recorded as None"
        # TODO: determine whether to raise an error or reset the environment

        next_mjx_state = self.pipeline_step(mjx_state, action)

        assert isinstance(next_mjx_state, mjxState), f"next_mjx_state is of type {type(next_mjx_state)}"
        assert isinstance(mjx_state, mjxState), f"mjx_state is of type {type(mjx_state)}"
        # mlutz: from what I've seen, .pipeline_state and .pipeline_step(...)
        # actually return an brax.mjx.base.State object however, the type
        # hinting suggests that it should return a brax.base.State object
        # brax.mjx.base.State inherits from brax.base.State but also inherits
        # from mjx.Data, which is needed for some rewards

        obs = self._get_obs(mjx_state, action)
        reward, is_healthy, reward_breakdown = self.reward_fn(mjx_state, action, next_mjx_state)

        if self._terminate_when_unhealthy:
            done = 1.0 - is_healthy
        else:
            done = jp.array(0)

        state.metrics.update(
            x_position=next_mjx_state.subtree_com[1][0],
            y_position=next_mjx_state.subtree_com[1][1],
            distance_from_origin=jp.linalg.norm(next_mjx_state.subtree_com[1]),
            x_velocity=(next_mjx_state.subtree_com[1][0] - mjx_state.subtree_com[1][0]) / self.dt,
            y_velocity=(next_mjx_state.subtree_com[1][1] - mjx_state.subtree_com[1][1]) / self.dt,
        )

        if self._log_reward_breakdown:
            for key, val in reward_breakdown.items():
                state.metrics[key] = val

        # TODO: fix the type hinting...
        return state.replace(
            pipeline_state=next_mjx_state,
            obs=obs,
            reward=reward,
            done=done,
        )

    def _get_obs(self, data: mjxState, action: jp.ndarray) -> jp.ndarray:
        """Observes humanoid body position, velocities, and angles.

        Args:
            data: The current state of the environment.
            action: The current action.

        Returns:
            Observations of the environment.
        """
        position = data.qpos
        if self._exclude_current_positions_from_observation:
            position = position[2:]

        # external_contact_forces are excluded
        return jp.concatenate(
            [
                position,
                data.qvel,
                data.cinert[1:].ravel(),
                data.cvel[1:].ravel(),
                data.qfrc_actuator,
            ]
        )
