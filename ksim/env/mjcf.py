"""Base environment class for interacting with MJCF models."""

import asyncio
import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
import mujoco
import xax
from mujoco import mjx

from ksim.action.mjcf import MjcfAction
from ksim.env.base import Environment, EnvironmentConfig
from ksim.state.mjcf import MjcfState
from ksim.terminations.base import Termination
from ksim.utils.mujoco import get_qpos_ids, get_qvel_ids, init, step

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class MjcfEnvironmentConfig(EnvironmentConfig):
    """Master configuration class for the base environment."""

    # Environment configuration options.
    gravity: tuple[float, float, float] = xax.field(
        value=(0.0, 0.0, -9.81),
        help="Gravity vector.",
    )
    dt: float = xax.field(
        value=0.02,
        help="Simulation time step.",
    )
    num_substeps: int = xax.field(
        value=1,
        help="Number of substeps to take per step.",
    )

    # Action configuration options.
    action_scale: float = xax.field(
        value=1.0,
        help="Action scaling factor.",
    )
    action_range: tuple[float, float] | None = xax.field(
        value=(-1.0, 1.0),
        help="Action range.",
    )

    # Robot configuration options.
    base_init_pos: tuple[float, float, float] = xax.field(
        value=(0.0, 0.0, 0.0),
        help="Initial base position.",
    )
    base_init_quat: tuple[float, float, float, float] = xax.field(
        value=(1.0, 0.0, 0.0, 0.0),
        help="Initial base quaternion.",
    )
    default_qpos: list[float] | None = xax.field(
        value=None,
        help="Default joint positions.",
    )
    default_qvel: list[float] | None = xax.field(
        value=None,
        help="Default joint velocities.",
    )

    # Solver configuration options.
    solver: str = xax.field(
        value="cg",
        help="Solver to use.",
    )
    solver_iterations: int = xax.field(
        value=6,
        help="Solver iterations.",
    )
    solver_tolerance: float = xax.field(
        value=1e-5,
        help="Solver tolerance.",
    )

    # Scene configuration options.
    enable_viewer: bool = False
    camera_pos: tuple[float, float, float] = (2.0, 0.0, 2.5)
    camera_lookat: tuple[float, float, float] = (0.0, 0.0, 0.5)
    camera_fov: float = 40


Tconfig = TypeVar("Tconfig", bound=MjcfEnvironmentConfig)
Tstate = TypeVar("Tstate", bound=MjcfState)
Taction = TypeVar("Taction", bound=MjcfAction)


class MjcfEnvironment(Environment[Tconfig, Tstate, Taction], ABC, Generic[Tconfig, Tstate, Taction]):
    def __init__(self, config: Tconfig, terminations: list[Termination]) -> None:
        super().__init__(config)
        self.terminations = terminations

        # Load and compile the MJCF model
        self.mj_model = self._load_mj_model()
        self.mjx_model = mjx.put_model(self.mj_model)

        # Get joint info
        self.joint_names = [
            self.mj_model.names[adr:].split(b"\x00", 1)[0].decode("utf-8") for adr in self.mj_model.name_jntadr
        ]

        # Get joint IDs for qpos and qvel
        self.qpos_ids = get_qpos_ids(self.mj_model, self.joint_names)
        self.qvel_ids = get_qvel_ids(self.mj_model, self.joint_names)

    def _load_mj_model(self) -> mujoco.MjModel:
        """Load the MuJoCo model."""
        model_path = asyncio.run(self.get_model_path())

        try:
            model = mujoco.MjModel.from_xml_path(str(model_path))

            # Configure model parameters
            match self.config.solver:
                case "cg":
                    model.opt.solver = mujoco.mjtSolver.mjSOL_CG
                case "newton":
                    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
                case "sparse":
                    model.opt.solver = mujoco.mjtSolver.mjSOL_SPARSE
                case _:
                    raise ValueError(f"Invalid solver: {self.config.solver}")

            model.opt.iterations = self.config.solver_iterations
            model.opt.ls_iterations = self.config.solver_iterations
            model.opt.tolerance = self.config.solver_tolerance
            model.opt.gravity = self.config.gravity
            model.opt.timestep = self.config.dt

            return model

        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}") from e

    @property
    def num_actions(self) -> int:
        return len(self.joint_names)

    @property
    def dt(self) -> float:
        return self.config.dt

    def get_initial_state(self) -> Tstate:
        # Initialize state arrays with zeros/defaults
        qpos = jnp.zeros(self.mjx_model.nq)
        qvel = jnp.zeros(self.mjx_model.nv)

        # Set default positions if provided
        if self.config.default_qpos is not None:
            qpos = qpos.at[self.qpos_ids].set(jnp.array(self.config.default_qpos))
        if self.config.default_qvel is not None:
            qvel = qvel.at[self.qvel_ids].set(jnp.array(self.config.default_qvel))

        # Set initial base position and orientation
        qpos = qpos.at[0:3].set(jnp.array(self.config.base_init_pos))
        qpos = qpos.at[3:7].set(jnp.array(self.config.base_init_quat))

        # Initialize MJX data
        data = init(
            self.mjx_model,
            qpos_nj=qpos,
            qvel_nj=qvel,
            ctrl_nj=jnp.zeros(self.mjx_model.nu),
        )

        return MjcfState(
            data=data,
            obs=self._get_obs(data),
            done=jnp.array(False),
            metrics={},
            info={},
        )

    def get_model_path(self) -> Path:
        """Return path to the MJCF/URDF model file."""
        raise NotImplementedError("Override get_model_path to return the path to your model file.")

    def step(self, state: Tstate, actions: Taction) -> Tstate:
        # Process actions
        scaled_actions = self._process_actions(actions)

        # Step physics
        next_data = step(
            self.mjx_model,
            state.data,
            scaled_actions,
            num_substeps=self.config.num_substeps,
        )

        # Get observation and reward
        obs = self._get_obs(next_data)

        # Check termination
        done = self.check_termination(next_data)

        return MjcfState(data=next_data, obs=obs, done=done)

    def _process_actions(self, actions: jax.Array) -> jax.Array:
        """Scale and clip actions."""
        if self.config.action_range is not None:
            actions = jnp.clip(actions, *self.config.action_range)
        return actions * self.config.action_scale

    def check_termination(self, data: mjx.Data) -> jax.Array:
        """Check termination conditions."""
        term_flags = [term(data) for term in self.terminations]
        if not term_flags:
            return jnp.zeros((), dtype=jnp.bool_)
        return jnp.any(jnp.stack(term_flags))
