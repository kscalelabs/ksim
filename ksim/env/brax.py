"""Defines the default humanoid environment."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Generic, TypeVar, cast

import jax
import jax.numpy as jnp
import mujoco
import tqdm
import xax
from brax.envs.base import PipelineEnv
from brax.envs.base import State as BraxState
from brax.io import mjcf
from brax.mjx.base import State as MjxState
from kscale import K
from omegaconf import MISSING

from ksim.observation.base import Observation
from ksim.resets.base import Reset
from ksim.rewards.base import Reward
from ksim.terminations.base import Termination

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def get_model_path(model_name: str, cache: bool = True) -> str | Path:
    async with K() as api:
        urdf_dir = await api.download_and_extract_urdf(model_name, cache=cache)

    try:
        urdf_path = next(urdf_dir.glob("*.mjcf"))
    except StopIteration:
        raise ValueError(f"No MJCF file found for {model_name} (in {urdf_dir})")

    return urdf_path


def _unique_dict(things: list[tuple[str, T]], name: str) -> dict[str, T]:
    return_dict = {k: v for k, v in things}
    if len(return_dict) != len(things):
        raise ValueError(f"Found duplicate {name} names!")
    return return_dict


@dataclass
class KScaleEnvConfig:
    # Model configuration options.
    model_name: str = xax.field(
        value=MISSING,
        help="The name of the model to use.",
    )

    # Environment configuration options.
    dt: float = xax.field(
        value=0.02,
        help="Simulation time step.",
    )
    frames_per_env_step: int = xax.field(
        value=1,
        help="The number of frames to simulate per environment step.",
    )
    debug_env: bool = xax.field(
        value=False,
        help="Whether to enable debug mode for the environment.",
    )
    solver: str = xax.field(
        value="CG",
        help="MuJoCo solver type ('CG' or 'Newton').",
    )

    # Solver configuration options.
    solver_iterations: int = xax.field(
        value=6,
        help="Number of main solver iterations.",
    )
    solver_ls_iterations: int = xax.field(
        value=6,
        help="Number of line search iterations.",
    )

    # Simulation artifact options.
    ignore_cached_urdf: bool = xax.field(
        value=False,
        help="Whether to ignore the cached URDF.",
    )


class KScaleEnv(PipelineEnv):
    """Defines a generic environment for interacting with K-Scale models."""

    def __init__(
        self,
        config: KScaleEnvConfig,
        terminations: Collection[Termination],
        resets: Collection[Reset],
        rewards: Collection[Reward],
        observations: Collection[Observation],
    ) -> None:
        self.config = config
        self.terminations = _unique_dict([(term.termination_name, term) for term in terminations], "termination")
        self.resets = _unique_dict([(reset.reset_name, reset) for reset in resets], "reset")
        self.rewards = _unique_dict([(reward.reward_name, reward) for reward in rewards], "reward")
        self.observations = _unique_dict([(obs.observation_name, obs) for obs in observations], "observation")

        # Downloads the model from the K-Scale API and loads it into MuJoCo.
        model_path = str(
            asyncio.run(
                get_model_path(
                    model_name=self.config.model_name,
                    cache=not self.config.ignore_cached_urdf,
                )
            )
        )

        logger.info("Initializing model from %s", model_path)

        # Configure model parameters
        match self.config.solver.lower():
            case "cg":
                solver = mujoco.mjtSolver.mjSOL_CG
            case "newton":
                solver = mujoco.mjtSolver.mjSOL_NEWTON
            case "sparse":
                solver = mujoco.mjtSolver.mjSOL_SPARSE
            case _:
                raise ValueError(f"Invalid solver: {self.config.solver}")

        logger.info("Loading model %s", model_path)
        sys = mjcf.load(model_path)
        sys = sys.tree_replace(
            {
                "opt.timestep": self.config.dt,
                "opt.solver": solver,
                "opt.iterations": self.config.solver_iterations,
                "opt.ls_iterations": self.config.solver_ls_iterations,
            }
        )

        logger.info("Initializing pipeline")
        super().__init__(
            sys=sys,
            backend="mjx",
            n_frames=self.config.frames_per_env_step,
            debug=self.config.debug_env,
        )

    def _pipeline_state_to_state(self, pipeline_state: MjxState) -> BraxState:
        obs = self.get_observation(pipeline_state)
        reward = self.get_reward(pipeline_state)
        done = self.get_done(pipeline_state)
        return BraxState(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
        )

    def reset(self, rng: jnp.ndarray) -> BraxState:
        q = jnp.zeros(self.sys.q_size())
        qd = jnp.zeros(self.sys.qd_size())
        pipeline_state = cast(MjxState, self.pipeline_init(q, qd))
        return self._pipeline_state_to_state(pipeline_state)

    def step(self, state: BraxState, action: jnp.ndarray) -> BraxState:
        pipeline_state = cast(MjxState, self.pipeline_step(state.pipeline_state, action))
        return self._pipeline_state_to_state(pipeline_state)

    def get_observation(self, pipeline_state: MjxState) -> dict[str, jnp.ndarray]:
        observations = {}
        for observation_name, observation in self.observations.items():
            observations[observation_name] = observation(pipeline_state)
        return observations

    def get_reward(self, pipeline_state: MjxState) -> jnp.ndarray:
        return jnp.zeros(())

    def get_done(self, pipeline_state: MjxState) -> jnp.ndarray:
        return jnp.zeros((), dtype=bool)

    def test_run(self, num_steps: int, render: bool = True, seed: int = 0) -> None:
        logger.info("Running test run for %d steps", num_steps)

        reset = jax.jit(self.reset)
        step = jax.jit(self.step)

        rng = jax.random.PRNGKey(seed)
        state: BraxState = reset(rng)
        trajectory = [state.pipeline_state]

        # Run simulation
        logger.info("Got initial state")
        for _ in tqdm.trange(num_steps):
            # Generate random action (replace with your control policy)
            rng, subkey = jax.random.split(rng)
            action = jax.random.uniform(subkey, shape=(self.action_size,), minval=-1, maxval=1)

            # Step environment
            state = step(state, action)
            trajectory.append(state.pipeline_state)

            if state.done.all():
                break

        # Render if requested
        if render:
            breakpoint()
            frames = self.render(trajectory)
