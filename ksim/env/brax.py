"""Defines the default humanoid environment."""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Unpack

import jax
import jax.numpy as jp
import mujoco
import xax
from brax.envs.base import PipelineEnv
from brax.io import mjcf
from brax.mjx.base import State as MjxState
from kscale import K

logger = logging.getLogger(__name__)


@dataclass
class Config:
    backend: str = xax.field(
        default="mjx",
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
        default=1,
        help="The number of frames to simulate per environment step.",
    )
    debug_env: bool = xax.field(
        default=False,
        help="Whether to enable debug mode for the environment.",
    )
    solver: str = xax.field(
        default="CG",
        help="MuJoCo solver type ('CG' or 'Newton').",
    )

    # Solver configuration options.
    solver_iterations: int = xax.field(
        default=6,
        help="Number of main solver iterations.",
    )
    solver_ls_iterations: int = xax.field(
        default=6,
        help="Number of line search iterations.",
    )
    solver_tolerance: float = xax.field(
        default=1e-5,
        help="Solver tolerance.",
    )


class KScaleEnv(PipelineEnv):
    """Defines a generic environment for interacting with K-Scale models."""

    def __init__(self, config: Config) -> None:
        self.config = config

        model_path = asyncio.run(self.get_model_path())

        mj_model = mujoco.MjModel.from_xml_path(model_path)

        # Configure model parameters
        match self.config.solver:
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

        sys = mjcf.load_model(mj_model)

        super().__init__(
            sys=sys,
            backend=self.config.backend,
            n_frames=self.config.frames_per_env_step,
            debug=self.config.debug_env,
        )

    async def get_model_path(self) -> str | Path:
        async with K() as api:
            urdf_dir = await api.download_and_extract_urdf(
                self.config.model_name,
                cache=not self.config.ignore_cached_urdf,
            )

        try:
            urdf_path = next(urdf_dir.glob("*.mjcf"))
        except StopIteration:
            raise ValueError(f"No MJCF file found for {self.config.model_name} (in {urdf_dir})")

        return urdf_path

    def reset(self, rng: jp.ndarray) -> MjxState:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.qpos0 + jax.random.uniform(rng1, (self.sys.nq,), minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, (self.sys.nv,), minval=low, maxval=hi)
        mjx_state = self.pipeline_init(qpos, qvel)
        assert isinstance(mjx_state, MjxState), f"mjx_state is of type {type(mjx_state)}"
        return mjx_state

    def step(self, state: MjxState, action: jp.ndarray) -> MjxState:
        mjx_state = state.pipeline_state
        assert mjx_state is not None, "state.pipeline_state was recorded as None"
        next_mjx_state = self.pipeline_step(mjx_state, action)
        assert isinstance(next_mjx_state, MjxState), f"next_mjx_state is of type {type(next_mjx_state)}"
        assert isinstance(mjx_state, MjxState), f"mjx_state is of type {type(mjx_state)}"
        return next_mjx_state
