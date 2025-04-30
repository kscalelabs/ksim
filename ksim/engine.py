"""Base JAX centric environment class.

Designed to be easily extensible to any physics engine that separates
model and data.
"""

__all__ = [
    "PhysicsEngine",
    "MjxEngine",
    "MujocoEngine",
    "get_physics_engine",
    "engine_type_from_physics_model",
]

import logging
from abc import ABC, abstractmethod
from typing import Collection, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from mujoco import mjx

from ksim.actuators import Actuators, StatefulActuators
from ksim.events import Event
from ksim.resets import Reset
from ksim.types import PhysicsModel, PhysicsState

logger = logging.getLogger(__name__)

EngineType = Literal["mjx", "mujoco"]


class PhysicsEngine(eqx.Module, ABC):
    """The role of an engine is simple: reset and step. Decoupled from data."""

    actuators: Actuators
    resets: Collection[Reset]
    events: Collection[Event]
    phys_steps_per_ctrl_steps: int
    max_action_latency_step: float

    def __init__(
        self,
        resets: Collection[Reset],
        events: Collection[Event],
        actuators: Actuators,
        phys_steps_per_ctrl_steps: int,
        max_action_latency_step: float,
    ) -> None:
        """Initialize the MJX engine with resetting and actuators."""
        self.actuators = actuators
        self.resets = resets
        self.events = events
        self.phys_steps_per_ctrl_steps = phys_steps_per_ctrl_steps
        self.max_action_latency_step = max_action_latency_step

    @abstractmethod
    def reset(self, physics_model: PhysicsModel, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsState:
        """Reset the engine and return the physics data."""

    @abstractmethod
    def step(
        self,
        action: Array,
        physics_model: PhysicsModel,
        physics_state: PhysicsState,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> PhysicsState:
        """Step the engine and return the updated physics data."""

    def _reset_events(self, rng: PRNGKeyArray) -> xax.FrozenDict[str, PyTree]:
        event_states: dict[str, PyTree] = {}
        for event in self.events:
            rng, event_rng = jax.random.split(rng)
            event_states[event.event_name] = event.get_initial_event_state(event_rng)
        return xax.FrozenDict(event_states)


class MjxEngine(PhysicsEngine):
    """Defines an engine for MJX models."""

    @xax.jit(static_argnames=["self"])
    def reset(self, physics_model: mjx.Model, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsState:
        mjx_data = mjx.make_data(physics_model)

        for reset in self.resets:
            rng, reset_rng = jax.random.split(rng)
            mjx_data = reset(mjx_data, curriculum_level, reset_rng)

        mjx_data = mjx.forward(physics_model, mjx_data)
        assert isinstance(mjx_data, mjx.Data)
        default_action = self.actuators.get_default_action(mjx_data)

        # Gets the initial actuator state for stateful actuators.
        actuator_state = (
            self.actuators.get_initial_state(mjx_data, rng) if isinstance(self.actuators, StatefulActuators) else None
        )

        rng, latency_rng = jax.random.split(rng)

        return PhysicsState(
            data=mjx_data,
            most_recent_action=default_action,
            event_states=self._reset_events(rng),
            actuator_state=actuator_state,
            action_latency=jax.random.uniform(
                latency_rng,
                minval=0,
                maxval=self.max_action_latency_step * curriculum_level.astype(float),
            )
            .round()
            .astype(int),
        )

    @xax.jit(static_argnames=["self"])
    def _physics_step(self, physics_model: mjx.Model, data_with_ctrl: mjx.Data) -> mjx.Data:
        # Just performs the MJX step, but wraps it in it's own JIT which can be
        # cached to prevent heavy recompilation every time the rewards or
        # events change.
        return mjx.step(physics_model, data_with_ctrl)

    @xax.jit(static_argnames=["self"])
    def step(
        self,
        action: Array,
        physics_model: mjx.Model,
        physics_state: PhysicsState,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> PhysicsState:
        mjx_data = physics_state.data
        phys_steps_per_ctrl_steps = self.phys_steps_per_ctrl_steps
        prev_action = physics_state.most_recent_action

        def move_physics(
            carry: tuple[mjx.Data, Array, xax.FrozenDict[str, PyTree], PyTree],
            rng: PRNGKeyArray,
        ) -> tuple[tuple[mjx.Data, Array, xax.FrozenDict[str, PyTree], PyTree], None]:
            data, step_num, event_states, actuator_state = carry

            # Randomly apply the action with some latency.
            ctrl = jax.lax.select(step_num >= physics_state.action_latency, action, prev_action)

            # Apply the events.
            new_event_states = {}
            for event in self.events:
                rng, event_rng = jax.random.split(rng)
                data, new_event_state = event(
                    model=physics_model,
                    data=data,
                    event_state=event_states[event.event_name],
                    curriculum_level=curriculum_level,
                    rng=event_rng,
                )
                new_event_states[event.event_name] = new_event_state

            rng, ctrl_rng = jax.random.split(rng)

            if isinstance(self.actuators, StatefulActuators):
                torques, actuator_state = self.actuators.get_stateful_ctrl(
                    action=ctrl,
                    physics_data=data,
                    actuator_state=actuator_state,
                    rng=ctrl_rng,
                )
            else:
                torques = self.actuators.get_ctrl(action=ctrl, physics_data=data, rng=ctrl_rng)

            data_with_ctrl = data.replace(ctrl=torques)
            new_data = self._physics_step(physics_model, data_with_ctrl)
            return (new_data, step_num + 1.0, xax.FrozenDict(new_event_states), actuator_state), None

        # Runs the model for N steps.
        (mjx_data, *_, event_info, actuator_state_final), _ = jax.lax.scan(
            move_physics,
            (mjx_data, jnp.array(0.0), physics_state.event_states, physics_state.actuator_state),
            jax.random.split(rng, phys_steps_per_ctrl_steps),
        )

        return PhysicsState(
            data=mjx_data,
            most_recent_action=action,
            event_states=xax.FrozenDict(event_info),
            actuator_state=actuator_state_final,
            action_latency=physics_state.action_latency,
        )


class MujocoEngine(PhysicsEngine):
    """Defines an engine for MuJoCo models."""

    def reset(self, physics_model: mujoco.MjModel, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsState:
        mujoco_data = mujoco.MjData(physics_model)

        for reset in self.resets:
            rng, reset_rng = jax.random.split(rng)
            mujoco_data = reset(mujoco_data, curriculum_level, reset_rng)

        mujoco.mj_forward(physics_model, mujoco_data)
        default_action = self.actuators.get_default_action(mujoco_data)
        actuator_state = (
            self.actuators.get_initial_state(mujoco_data, rng)
            if isinstance(self.actuators, StatefulActuators)
            else None
        )

        rng, latency_rng = jax.random.split(rng)

        return PhysicsState(
            data=mujoco_data,
            most_recent_action=default_action,
            event_states=self._reset_events(rng),
            actuator_state=actuator_state,
            action_latency=jax.random.uniform(
                latency_rng,
                minval=0,
                maxval=self.max_action_latency_step * curriculum_level.astype(float),
            )
            .round()
            .astype(int),
        )

    def step(
        self,
        action: Array,
        physics_model: mujoco.MjModel,
        physics_state: PhysicsState,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> PhysicsState:
        mujoco_data = physics_state.data

        if not isinstance(mujoco_data, mujoco.MjData):
            raise ValueError("Mujoco data is not a MjData")

        mujoco.mj_forward(physics_model, mujoco_data)
        phys_steps_per_ctrl_steps = self.phys_steps_per_ctrl_steps
        prev_action = physics_state.most_recent_action

        event_states = physics_state.event_states
        actuator_state = physics_state.actuator_state

        for step_num in range(phys_steps_per_ctrl_steps):
            # Randomly apply the action with some latency.
            ctrl = jax.lax.select(step_num >= physics_state.action_latency, action, prev_action)

            # Apply the events.
            new_event_states = {}
            for event in self.events:
                mujoco_data, new_event_state = event(
                    model=physics_model,
                    data=mujoco_data,
                    event_state=event_states[event.event_name],
                    curriculum_level=curriculum_level,
                    rng=rng,
                )
                new_event_states[event.event_name] = new_event_state

            event_states = xax.FrozenDict(new_event_states)

            if isinstance(self.actuators, StatefulActuators):
                torques, actuator_state = self.actuators.get_stateful_ctrl(
                    action=ctrl,
                    physics_data=mujoco_data,
                    actuator_state=actuator_state,
                    rng=rng,
                )
            else:
                torques = self.actuators.get_ctrl(action=ctrl, physics_data=mujoco_data, rng=rng)

            mujoco_data.ctrl[:] = torques
            mujoco.mj_step(physics_model, mujoco_data)

        return PhysicsState(
            data=mujoco_data,
            most_recent_action=action,
            event_states=event_states,
            actuator_state=actuator_state,
            action_latency=physics_state.action_latency,
        )


def get_physics_engine(
    engine_type: EngineType,
    resets: Collection[Reset],
    events: Collection[Event],
    actuators: Actuators,
    *,
    dt: float,
    ctrl_dt: float,
    max_action_latency: float,
) -> PhysicsEngine:
    if max_action_latency > ctrl_dt:
        logger.warning("`max_action_latency=%f` is greater than `ctrl_dt=%f`", max_action_latency, ctrl_dt)
    if (remainder := (ctrl_dt - round(ctrl_dt / dt) * dt)) > 1e-6:
        logger.warning("`ctrl_dt=%f` is not a multiple of `dt=%f` (remainder=%f)", ctrl_dt, dt, remainder)

    # Converts to steps.
    max_action_latency_step = max(max_action_latency / dt, 0)
    phys_steps_per_ctrl_steps = round(ctrl_dt / dt)

    match engine_type:
        case "mujoco":
            return MujocoEngine(
                resets=resets,
                events=events,
                actuators=actuators,
                max_action_latency_step=max_action_latency_step,
                phys_steps_per_ctrl_steps=phys_steps_per_ctrl_steps,
            )

        case "mjx":
            return MjxEngine(
                resets=resets,
                events=events,
                actuators=actuators,
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
