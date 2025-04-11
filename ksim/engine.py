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
from ksim.types import PhysicsModel, PhysicsState, PlannerState

logger = logging.getLogger(__name__)

EngineType = Literal["mjx", "mujoco"]


class PhysicsEngine(eqx.Module, ABC):
    """The role of an engine is simple: reset and step. Decoupled from data."""

    actuators: Actuators
    resets: Collection[Reset]
    events: Collection[Event]
    phys_steps_per_ctrl_steps: int
    min_action_latency_step: int
    max_action_latency_step: int

    def __init__(
        self,
        resets: Collection[Reset],
        events: Collection[Event],
        actuators: Actuators,
        phys_steps_per_ctrl_steps: int,
        min_action_latency_step: int,
        max_action_latency_step: int,
    ) -> None:
        """Initialize the MJX engine with resetting and actuators."""
        self.actuators = actuators
        self.resets = resets
        self.events = events
        self.phys_steps_per_ctrl_steps = phys_steps_per_ctrl_steps
        self.min_action_latency_step = min_action_latency_step
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

    def reset(self, physics_model: mjx.Model, curriculum_level: Array, rng: PRNGKeyArray) -> PhysicsState:
        mjx_data = mjx.make_data(physics_model)

        for reset in self.resets:
            rng, reset_rng = jax.random.split(rng)
            mjx_data = reset(mjx_data, curriculum_level, reset_rng)

        mjx_data = mjx.forward(physics_model, mjx_data)
        assert isinstance(mjx_data, mjx.Data)
        default_action = self.actuators.get_default_action(mjx_data)

        initial_position = mjx_data.qpos[7:]
        initial_velocity = jnp.zeros_like(initial_position)

        # Gets the initial planner state for stateful actuators.
        planner_state = (
            self.actuators.get_default_state(initial_position, initial_velocity)
            if isinstance(self.actuators, StatefulActuators)
            else None
        )

        return PhysicsState(
            data=mjx_data,
            most_recent_action=default_action,
            event_states=self._reset_events(rng),
            planner_state=planner_state,
        )

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

        rng, latency_rng = jax.random.split(rng)

        # We wait some random amount before actually applying the action.
        latency_steps = jax.random.randint(
            key=latency_rng,
            shape=(),
            minval=self.min_action_latency_step,
            maxval=self.max_action_latency_step,
        )

        def move_physics(
            carry: tuple[mjx.Data, Array, xax.FrozenDict[str, PyTree], PlannerState],
            rng: PRNGKeyArray,
        ) -> tuple[tuple[mjx.Data, Array, xax.FrozenDict[str, PyTree], PlannerState], None]:
            data, step_num, event_states, planner_state = carry

            # Randomly apply the action with some latency.
            ctrl = jax.lax.select(step_num >= latency_steps, action, prev_action)

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
                torques, planner_state = self.actuators.get_stateful_ctrl(
                    action=ctrl,
                    physics_data=data,
                    planner_state=planner_state,
                    rng=ctrl_rng,
                )
            else:
                torques = self.actuators.get_ctrl(action=ctrl, physics_data=data, rng=ctrl_rng)

            data_with_ctrl = data.replace(ctrl=torques)
            new_data = mjx.step(physics_model, data_with_ctrl)
            return (new_data, step_num + 1.0, xax.FrozenDict(new_event_states), planner_state), None

        # Runs the model for N steps.
        (mjx_data, *_, event_info, planner_state_final), _ = jax.lax.scan(
            move_physics,
            (mjx_data, jnp.array(0.0), physics_state.event_states, physics_state.planner_state),
            jax.random.split(rng, phys_steps_per_ctrl_steps),
        )

        return PhysicsState(
            data=mjx_data,
            most_recent_action=action,
            event_states=xax.FrozenDict(event_info),
            planner_state=planner_state_final,
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
        initial_position = mujoco_data.qpos[7:]
        initial_velocity = jnp.zeros_like(initial_position)
        default_planner = PlannerState(position=initial_position, velocity=initial_velocity)

        return PhysicsState(
            data=mujoco_data,
            most_recent_action=default_action,
            event_states=self._reset_events(rng),
            planner_state=default_planner,
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

        latency_rng, action_rng = jax.random.split(rng)

        # We wait some random amount before actually applying the action.
        latency_steps = jax.random.randint(
            key=latency_rng,
            shape=(),
            minval=self.min_action_latency_step,
            maxval=self.max_action_latency_step,
        )

        event_states = physics_state.event_states
        planner_state = physics_state.planner_state

        for step_num in range(phys_steps_per_ctrl_steps):
            # Randomly apply the action with some latency.
            ctrl = jax.lax.select(step_num >= latency_steps, action, prev_action)

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
                torques, planner_state = self.actuators.get_stateful_ctrl(
                    action=ctrl,
                    physics_data=mujoco_data,
                    planner_state=planner_state,
                    rng=action_rng,
                )
            else:
                torques = self.actuators.get_ctrl(action=ctrl, physics_data=mujoco_data, rng=action_rng)

            mujoco_data.ctrl[:] = torques
            mujoco.mj_step(physics_model, mujoco_data)

        return PhysicsState(
            data=mujoco_data,
            most_recent_action=action,
            event_states=event_states,
            planner_state=planner_state,
        )


def get_physics_engine(
    engine_type: EngineType,
    resets: Collection[Reset],
    events: Collection[Event],
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
                events=events,
                actuators=actuators,
                min_action_latency_step=min_action_latency_step,
                max_action_latency_step=max_action_latency_step,
                phys_steps_per_ctrl_steps=phys_steps_per_ctrl_steps,
            )

        case "mjx":
            return MjxEngine(
                resets=resets,
                events=events,
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
