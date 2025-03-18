"""Base JAX centric environment class.

Designed to be easily extensible to any physics engine that separates
model and data.
"""

import logging
from abc import ABC, abstractmethod
from typing import Collection

import jax
import jax.numpy as jnp
import mujoco
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx

from ksim.actuators import Actuators
from ksim.env.data import PhysicsModel, PhysicsState
from ksim.resets import Reset

logger = logging.getLogger(__name__)


class PhysicsEngine(ABC):
    """The role of an engine is simple: reset and step. Decoupled from data."""

    @abstractmethod
    def reset(self, rng: PRNGKeyArray) -> PhysicsState:
        """Reset the engine and return the physics model and data."""

    @abstractmethod
    def step(
        self,
        action: Array,
        physics_state: PhysicsState,
        rng: PRNGKeyArray,
    ) -> PhysicsState:
        """Step the engine and return the physics model and data."""


class MjxEngine(PhysicsEngine):
    """Defines an engine for MJX models."""

    def __init__(
        self,
        physics_model: mjx.Model,
        resets: Collection[Reset],
        actuators: Actuators,
        *,
        phys_steps_per_ctrl_steps: int,
        min_action_latency_step: int,
        max_action_latency_step: int,
    ) -> None:
        """Initialize the MJX engine with resetting and actuators."""
        self.physics_model = physics_model
        self.actuators = actuators
        self.resets = resets

        self.phys_steps_per_ctrl_steps = phys_steps_per_ctrl_steps
        self.min_action_latency_step = min_action_latency_step
        self.max_action_latency_step = max_action_latency_step

    def reset(self, rng: PRNGKeyArray) -> PhysicsState:
        mjx_model = self.physics_model
        mjx_data = mjx.make_data(mjx_model)

        for reset in self.resets:
            rng, reset_rng = jax.random.split(rng)
            mjx_data = reset(mjx_data, reset_rng)

        mjx_data = mjx.forward(mjx_model, mjx_data)
        assert isinstance(mjx_data, mjx.Data)
        default_action = mjx_data.ctrl

        return PhysicsState(
            model=mjx_model,
            data=mjx_data,
            most_recent_action=default_action,
        )

    def step(
        self,
        action: Array,
        state: PhysicsState,
        rng: PRNGKeyArray,
    ) -> PhysicsState:
        mjx_model = state.model
        mjx_data = state.data
        mjx_data = mjx.forward(mjx_model, mjx_data)
        phys_steps_per_ctrl_steps = self.phys_steps_per_ctrl_steps
        prev_action = state.most_recent_action

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
            data_with_ctrl = mjx.forward(mjx_model, data_with_ctrl)
            new_data = mjx.step(mjx_model, data_with_ctrl)
            return (new_data, step_num + 1.0), None

        # Runs the model for N steps.
        (mjx_data, *_), _ = jax.lax.scan(
            move_physics,
            (mjx_data, jnp.array(0.0)),
            None,
            length=phys_steps_per_ctrl_steps,
        )

        return PhysicsState(model=mjx_model, data=mjx_data, most_recent_action=action)


class MujocoEngine(PhysicsEngine):
    pass


def get_physics_engine(
    physics_model: PhysicsModel,
    resets: Collection[Reset],
    actuators: Actuators,
    *,
    dt: float,
    ctrl_dt: float,
    min_action_latency: float,
    max_action_latency: int,
) -> PhysicsEngine:
    if min_action_latency > ctrl_dt:
        raise RuntimeError(f"`{min_action_latency=}` cannot be greater than `{ctrl_dt=}`")
    if max_action_latency > ctrl_dt:
        logger.warning("`max_action_latency=%f` is greater than `ctrl_dt=%f`", max_action_latency, ctrl_dt)
    if (ctrl_dt - (ctrl_dt // dt) * dt) > 1e-6:
        logger.warning("`ctrl_dt=%f` is not a multiple of `dt=%f`", ctrl_dt, dt)

    # Converts to steps.
    min_action_latency_step = max(round(min_action_latency / dt), 0)
    max_action_latency_step = max(round(max_action_latency / dt), min_action_latency)
    phys_steps_per_ctrl_steps = round(ctrl_dt / dt)

    if isinstance(physics_model, mjx.Model):
        return MjxEngine(
            physics_model=physics_model,
            resets=resets,
            actuators=actuators,
            dt=dt,
            ctrl_dt=ctrl_dt,
            min_action_latency_step=min_action_latency_step,
            max_action_latency_step=max_action_latency_step,
            phys_steps_per_ctrl_steps=phys_steps_per_ctrl_steps,
        )

    if isinstance(physics_model, mujoco.MjModel):
        raise NotImplementedError("MujocoEngine is not implemented")

        return MujocoEngine(
            physics_model=physics_model,
            resets=resets,
            actuators=actuators,
            min_action_latency_step=min_action_latency_step,
            max_action_latency_step=max_action_latency_step,
            phys_steps_per_ctrl_steps=phys_steps_per_ctrl_steps,
        )

    raise ValueError(f"Unsupported physics model type: {type(physics_model)}")
