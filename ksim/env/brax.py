"""Defines the default humanoid environment."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Collection

import jax
import jax.numpy as jnp
import mujoco
import tqdm
import xax
from brax.envs.base import PipelineEnv
from brax.io import mjcf
from kscale import K
from omegaconf import MISSING

from ksim.observation.base import Observation
from ksim.resets.base import Reset
from ksim.rewards.base import Reward
from ksim.state.base import State
from ksim.terminations.base import Termination

logger = logging.getLogger(__name__)


async def get_model_path(model_name: str, cache: bool = True) -> str | Path:
    async with K() as api:
        urdf_dir = await api.download_and_extract_urdf(model_name, cache=cache)

    try:
        urdf_path = next(urdf_dir.glob("*.mjcf"))
    except StopIteration:
        raise ValueError(f"No MJCF file found for {model_name} (in {urdf_dir})")

    return urdf_path


@dataclass
class KScaleEnvConfig:
    # Model configuration options.
    model_name: str = xax.field(
        value=MISSING,
        help="The name of the model to use.",
    )

    # Environment configuration options.
    backend: str = xax.field(
        value="mjx",
        help="The physics backend to use.",
    )
    gravity: tuple[float, float, float] = xax.field(
        value=(0.0, 0.0, -9.81),
        help="Gravity vector.",
    )
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
    solver_tolerance: float = xax.field(
        value=1e-5,
        help="Solver tolerance.",
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
        terminations: Collection[Termination[State]],
        resets: Collection[Reset[State]],
        rewards: Collection[Reward[State]],
        observations: Collection[Observation[State]],
    ) -> None:
        self.config = config
        self.terminations = terminations
        self.resets = resets
        self.rewards = rewards

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
        mj_model = mujoco.MjModel.from_xml_path(model_path)

        # Configure model parameters
        match self.config.solver.lower():
            case "cg":
                mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
            case "newton":
                mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
            case "sparse":
                mj_model.opt.solver = mujoco.mjtSolver.mjSOL_SPARSE
            case _:
                raise ValueError(f"Invalid solver: {self.config.solver}")

        mj_model.opt.iterations = self.config.solver_iterations
        mj_model.opt.ls_iterations = self.config.solver_ls_iterations
        mj_model.opt.tolerance = self.config.solver_tolerance
        mj_model.opt.gravity = self.config.gravity
        mj_model.opt.timestep = self.config.dt

        logger.info("Loading model %s", model_path)
        sys = mjcf.load_model(mj_model)

        logger.info("Initializing pipeline")
        super().__init__(
            sys=sys,
            backend=self.config.backend,
            n_frames=self.config.frames_per_env_step,
            debug=self.config.debug_env,
        )

    def _pipeline_state_to_state(self, pipeline_state: State) -> State:
        obs = self.get_observation(pipeline_state)
        reward = self.get_reward(pipeline_state)
        done = self.get_done(pipeline_state)
        return State(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)

    def reset(self, rng: jnp.ndarray) -> State:
        q = jnp.zeros(self.sys.q_size())
        qd = jnp.zeros(self.sys.qd_size())
        pipeline_state = self.pipeline_init(q, qd)
        return self._pipeline_state_to_state(pipeline_state)

    def step(self, state: State, action: jnp.ndarray) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        return self._pipeline_state_to_state(pipeline_state)

    def get_observation(self, pipeline_state: State) -> jnp.ndarray:
        return pipeline_state

    def get_reward(self, pipeline_state: State) -> jnp.ndarray:
        return jnp.zeros(())

    def get_done(self, pipeline_state: State) -> jnp.ndarray:
        return jnp.zeros((), dtype=bool)

    def test_run(self, num_steps: int, render: bool = True) -> None:
        logger.info("Jitting reset")
        reset = jax.jit(self.reset)

        logger.info("Jitting step")
        step = jax.jit(self.step)

        # Initialize environment
        rng = jax.random.PRNGKey(0)
        logger.info("Running test run")
        state = reset(rng)

        # Store trajectory for rendering
        logger.info("Storing initial state")
        trajectory = [state.pipeline_state]

        # Run simulation
        for _ in tqdm.trange(num_steps):
            # Generate random action (replace with your control policy)
            rng, subkey = jax.random.split(rng)
            action = jax.random.uniform(subkey, shape=(self.action_size,), minval=-1, maxval=1)

            # Step environment
            state = step(state, action)
            trajectory.append(state.pipeline_state)

            logger.info("Stepping environment")

            if state.done.all():
                break

        # Render if requested
        if render:
            breakpoint()
            frames = self.render(trajectory)
