"""Base environment class for interacting with MJCF models."""

import asyncio
import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, Generic, TypeVar

import jax
import jax.numpy as jnp
import mujoco
import xax
from mujoco import mjx

from ksim.action.mjcf import MjcfAction
from ksim.env.base import Environment, EnvironmentConfig
from ksim.resets.base import Reset
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
        help="Number of substeps to take per control step.",
    )
    seed: int = xax.field(
        value=1337,
        help="Random seed for the environment.",
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
        help="Default positions for each joint, in radians.",
    )
    default_qvel: list[float] | None = xax.field(
        value=None,
        help="Default velocities for each joint, in radians per second.",
    )
    default_ctrl: list[float] | None = xax.field(
        value=None,
        help="Default joint controls.",
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
    def __init__(
        self,
        config: Tconfig,
        terminations: Collection[Termination[MjcfState]],
        resets: Collection[Reset[MjcfState]],
        model: Callable[[MjcfState], Taction] | None = None,
    ) -> None:
        super().__init__(config, model=model)

        self.terminations = terminations
        self.resets = resets

        # Load the Mujoco model and get some common information about it.
        self._mj_model = self.load_mj_model()
        self._joint_names: list[str] = [
            self._mj_model.names[adr:].split(b"\x00", 1)[0].decode("utf-8") for adr in self._mj_model.name_jntadr
        ]
        self._qpos_ids = get_qpos_ids(self._mj_model, self._joint_names)
        self._qvel_ids = get_qvel_ids(self._mj_model, self._joint_names)
        self._mj_model = self.configure_mj_model(self._mj_model, self._joint_names)
        self._default_qpos_j = self.get_default_qpos(len(self._qpos_ids))
        self._default_qvel_j = self.get_default_qvel(len(self._qvel_ids))
        self._default_pose = self._default_qpos_j[7:]
        self._lower_bounds, self._upper_bounds = self.mj_model.jnt_range[1:].T  # Skip the root joint.

        # Moves the model into MJX.
        self._mjx_model = mjx.put_model(self._mj_model)

    def load_mj_model(self) -> mujoco.MjModel:
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

    def configure_mj_model(self, model: mujoco.MjModel, joint_names: list[str]) -> mujoco.MjModel:
        # We can provide additional configuration options by overriding this method.
        return model

    def get_default_qpos(self, num_joints: int) -> jnp.ndarray:
        qpos = jnp.zeros(num_joints)
        if self.config.default_qpos is not None:
            qpos = qpos.at[self._qpos_ids].set(jnp.array(self.config.default_qpos))
        qpos = qpos.at[0:3].set(jnp.array(self.config.base_init_pos))
        qpos = qpos.at[3:7].set(jnp.array(self.config.base_init_quat))
        return qpos

    def get_default_qvel(self, num_joints: int) -> jnp.ndarray:
        qvel = jnp.zeros(num_joints)
        if self.config.default_qvel is not None:
            qvel = qvel.at[self._qvel_ids].set(jnp.array(self.config.default_qvel))
        return qvel

    @property
    def num_actions(self) -> int:
        return self._mjx_model.nu

    @property
    def dt(self) -> float:
        return self.config.dt

    @property
    def joint_names(self) -> list[str]:
        return self._joint_names

    @property
    def qpos_ids(self) -> jnp.ndarray:
        return self._qpos_ids

    @property
    def qvel_ids(self) -> jnp.ndarray:
        return self._qvel_ids

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def num_substeps(self) -> int:
        """Defines the number of substeps to take per control step."""
        return self.config.num_substeps

    @property
    def ctrl_dt(self) -> float:
        """Defines the time step of the controller."""
        return self.config.dt * self.config.num_substeps

    @property
    def lower_bounds(self) -> jnp.ndarray:
        """Defines the lower bounds for each joint, in radians."""
        return self._lower_bounds

    @property
    def upper_bounds(self) -> jnp.ndarray:
        """Defines the upper bounds for each joint, in radians."""
        return self._upper_bounds

    def get_initial_state(self) -> Tstate:
        data = init(
            self._mjx_model,
            qpos_j=self._default_qpos_j,
            qvel_j=self._default_qvel_j,
            ctrl_j=self._default_qpos_j[7:],
        )

        return MjcfState(
            rng=jax.random.PRNGKey(self.config.seed),
            model=self._mjx_model,
            data=data,
            done=jnp.zeros((), dtype=jnp.bool_),
        )

    def get_model_path(self) -> Path:
        raise NotImplementedError("Override get_model_path to return the path to your model file.")

    def process_actions(self, actions: Taction) -> Taction:
        return actions

    def check_termination(self, state: Tstate) -> jnp.ndarray:
        """Check if the state is terminal.

        We provide a small abstraction here by defining termination condition
        classes that wrap some common functionality.

        Args:
            state: The current state.

        Returns:
            The termination flags array as (termination_conditions, num_envs).
            This can be used to determine which termination conditions are met.
        """
        term_flags = [term(state.data) for term in self.terminations]
        if not term_flags:
            return jnp.zeros((), dtype=jnp.bool_)
        return jnp.stack(term_flags, axis=0)

    def step(self, actions: Taction, state: Tstate) -> Tstate:
        scaled_actions = self.process_actions(actions)
        next_data = step(
            self._mjx_model,
            state.data,
            scaled_actions,
            num_substeps=self.config.num_substeps,
        )
        done = self.check_termination(next_data)
        return MjcfState(model=self._mjx_model, data=next_data, done=done)

    def reset(self, state: Tstate) -> Tstate:
        for reset in self.resets:
            state = reset(state)
        data = init(
            self._mjx_model,
            qpos_j=self._default_qpos_j,
            qvel_j=self._default_qvel_j,
            ctrl_j=self._default_qpos_j[7:],
        )
        return state.replace(data=data)
