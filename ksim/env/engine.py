"""Base JAX centric environment class.

Designed to be easily extensible to any physics engine that separates
model and data.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree
from mujoco import mjx

from ksim.actuators import Actuators
from ksim.commands import Command
from ksim.env.data import PhysicsModel, PhysicsState
from ksim.observation import Observation
from ksim.randomization import Randomization
from ksim.resets import Reset
from ksim.rewards import Reward
from ksim.terminations import Termination

logger = logging.getLogger(__name__)

EngineType = Literal["mjx", "mujoco"]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EngineConstants:
    obs_generators: Collection[Observation]
    command_generators: Collection[Command]
    reward_generators: Collection[Reward]
    termination_generators: Collection[Termination]
    randomization_generators: Collection[Randomization]
    initial_carry: PyTree

    def __hash__(self) -> int:
        """Custom hash that excludes initial_carry which contains unhashable JAX arrays.

        This is used for determining when Jax should recompile a kernel at
        runtime. Since we only create EngineConstants once, we don't expect
        the user to change `initial_carry` at runtime, so we exclude it from
        the hash. However, this might cause issues with the Jax compiler if the
        user does change `initial_carry` at runtime.
        """
        return hash(
            (
                self.obs_generators,
                self.command_generators,
                self.reward_generators,
                self.termination_generators,
                self.randomization_generators,
            )
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EngineVariables:
    carry: PyTree
    commands: FrozenDict[str, Array]
    physics_state: PhysicsState
    rng: PRNGKeyArray


class PhysicsEngine(eqx.Module, ABC):
    """The role of an engine is simple: reset and step. Decoupled from data."""

    actuators: Actuators
    resets: Collection[Reset]
    phys_steps_per_ctrl_steps: int
    min_action_latency_step: int
    max_action_latency_step: int

    def __init__(
        self,
        resets: Collection[Reset],
        actuators: Actuators,
        phys_steps_per_ctrl_steps: int,
        min_action_latency_step: int,
        max_action_latency_step: int,
    ) -> None:
        """Initialize the MJX engine with resetting and actuators."""
        self.actuators = actuators
        self.resets = resets
        self.phys_steps_per_ctrl_steps = phys_steps_per_ctrl_steps
        self.min_action_latency_step = min_action_latency_step
        self.max_action_latency_step = max_action_latency_step

    @abstractmethod
    def reset(
        self,
        physics_model: PhysicsModel,
        rng: PRNGKeyArray,
    ) -> PhysicsState:
        """Reset the engine and return the physics model and data."""

    @abstractmethod
    def step(
        self,
        action: Array,
        physics_model: PhysicsModel,
        physics_state: PhysicsState,
        rng: PRNGKeyArray,
    ) -> PhysicsState:
        """Step the engine and return the updated physics data."""


class MjxEngine(PhysicsEngine):
    """Defines an engine for MJX models."""

    def reset(
        self,
        physics_model: mjx.Model,
        rng: PRNGKeyArray,
    ) -> PhysicsState:
        mjx_data = mjx.make_data(physics_model)

        for reset in self.resets:
            rng, reset_rng = jax.random.split(rng)
            mjx_data = reset(mjx_data, reset_rng)

        mjx_data = mjx.forward(physics_model, mjx_data)
        assert isinstance(mjx_data, mjx.Data)
        default_action = mjx_data.ctrl

        return PhysicsState(data=mjx_data, most_recent_action=default_action)

    def step(
        self,
        action: Array,
        physics_model: mjx.Model,
        physics_state: PhysicsState,
        rng: PRNGKeyArray,
    ) -> PhysicsState:
        mjx_data = physics_state.data
        mjx_data = mjx.forward(physics_model, mjx_data)
        phys_steps_per_ctrl_steps = self.phys_steps_per_ctrl_steps
        prev_action = physics_state.most_recent_action

        # We wait some random amount before actually applying the action.
        latency_steps = jax.random.randint(
            key=rng,
            shape=(),
            minval=self.min_action_latency_step,
            maxval=self.max_action_latency_step,
        )

        def move_physics(carry: tuple[mjx.Data, Array], _: None) -> tuple[tuple[mjx.Data, Array], None]:
            data, step_num = carry

            # Randomly apply the action with some latency.
            ctrl = jax.lax.select(
                step_num >= latency_steps,
                action,
                prev_action,
            )

            torques = self.actuators.get_ctrl(ctrl, data)
            data_with_ctrl = data.replace(ctrl=torques)
            # data_with_ctrl = mjx.forward(physics_model, data_with_ctrl)
            new_data = mjx.step(physics_model, data_with_ctrl)
            return (new_data, step_num + 1.0), None

        # Runs the model for N steps.
        (mjx_data, *_), _ = jax.lax.scan(
            move_physics,
            (mjx_data, jnp.array(0.0)),
            None,
            length=phys_steps_per_ctrl_steps,
        )

        return PhysicsState(data=mjx_data, most_recent_action=action)


class MujocoEngine(PhysicsEngine):
    """Defines an engine for MuJoCo models."""

    def reset(
        self,
        physics_model: mujoco.MjModel,
        rng: PRNGKeyArray,
    ) -> PhysicsState:
        mujoco_data = mujoco.MjData(physics_model)

        for reset in self.resets:
            rng, reset_rng = jax.random.split(rng)
            mujoco_data = reset(mujoco_data, reset_rng)

        mujoco.mj_forward(physics_model, mujoco_data)
        default_action = mujoco_data.ctrl

        return PhysicsState(data=mujoco_data, most_recent_action=default_action)

    def step(
        self,
        action: Array,
        physics_model: mujoco.MjModel,
        physics_state: PhysicsState,
        rng: PRNGKeyArray,
    ) -> PhysicsState:
        mujoco_data = physics_state.data

        if not isinstance(mujoco_data, mujoco.MjData):
            raise ValueError("Mujoco data is not a MjData")

        mujoco.mj_forward(physics_model, mujoco_data)
        phys_steps_per_ctrl_steps = self.phys_steps_per_ctrl_steps
        prev_action = physics_state.most_recent_action

        # We wait some random amount before actually applying the action.
        latency_steps = jax.random.randint(
            key=rng,
            shape=(),
            minval=self.min_action_latency_step,
            maxval=self.max_action_latency_step,
        )

        for step_num in range(phys_steps_per_ctrl_steps):
            ctrl = jax.lax.select(
                step_num >= latency_steps,
                action,
                prev_action,
            )

            torques = self.actuators.get_ctrl(ctrl, mujoco_data)
            mujoco_data.ctrl[:] = torques
            # mujoco.mj_forward(physics_model, mujoco_data)
            mujoco.mj_step(physics_model, mujoco_data)

        return PhysicsState(data=mujoco_data, most_recent_action=action)


def get_physics_engine(
    engine_type: EngineType,
    resets: Collection[Reset],
    actuators: Actuators,
    *,
    dt: float,
    ctrl_dt: float,
    min_action_latency: float,
    max_action_latency: float,
) -> PhysicsEngine:
    if min_action_latency > ctrl_dt:
        raise RuntimeError(f"`{min_action_latency=}` cannot be greater than `{ctrl_dt=}`")
    if max_action_latency > ctrl_dt:
        logger.warning("`max_action_latency=%f` is greater than `ctrl_dt=%f`", max_action_latency, ctrl_dt)
    if (ctrl_dt - (ctrl_dt // dt) * dt) > 1e-6:
        logger.warning("`ctrl_dt=%f` is not a multiple of `dt=%f`", ctrl_dt, dt)

    # Converts to steps.
    min_action_latency_step = max(round(min_action_latency / dt), 0)
    max_action_latency_step = max(round(max_action_latency / dt), min_action_latency_step)
    phys_steps_per_ctrl_steps = round(ctrl_dt / dt)

    match engine_type:
        case "mujoco":
            return MujocoEngine(
                resets=resets,
                actuators=actuators,
                min_action_latency_step=min_action_latency_step,
                max_action_latency_step=max_action_latency_step,
                phys_steps_per_ctrl_steps=phys_steps_per_ctrl_steps,
            )

        case "mjx":
            return MjxEngine(
                resets=resets,
                actuators=actuators,
                min_action_latency_step=min_action_latency_step,
                max_action_latency_step=max_action_latency_step,
                phys_steps_per_ctrl_steps=phys_steps_per_ctrl_steps,
            )

        case _:
            raise ValueError(f"Unsupported physics model type: {engine_type}")


def engine_type_from_physics_model(physics_model: PhysicsModel) -> EngineType:
    if isinstance(physics_model, mujoco.MjModel):
        return "mujoco"
    if isinstance(physics_model, mjx.Model):
        return "mjx"
    raise ValueError(f"Unsupported physics model type: {type(physics_model)}")
