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
from ksim.debugging import JitLevel
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
    min_action_latency_step: float
    max_action_latency_step: float
    drop_action_prob: float
    phys_steps_per_actuator_step: int

    def __init__(
        self,
        resets: Collection[Reset],
        events: Collection[Event],
        actuators: Actuators,
        phys_steps_per_ctrl_steps: int,
        min_action_latency_step: float,
        max_action_latency_step: float,
        drop_action_prob: float,
        phys_steps_per_actuator_step: int,
    ) -> None:
        """Initialize the MJX engine with resetting and actuators."""
        self.actuators = actuators
        self.resets = resets
        self.events = events
        self.phys_steps_per_ctrl_steps = phys_steps_per_ctrl_steps
        self.min_action_latency_step = min_action_latency_step
        self.max_action_latency_step = max_action_latency_step
        self.drop_action_prob = drop_action_prob
        self.phys_steps_per_actuator_step = phys_steps_per_actuator_step

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

        # Zeros out some non-zeroed quantities.
        mjx_data = mjx_data.replace(
            qvel=jnp.zeros_like(mjx_data.qvel),
            qacc=jnp.zeros_like(mjx_data.qacc),
        )

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
                minval=self.min_action_latency_step * curriculum_level,
                maxval=self.max_action_latency_step * curriculum_level,
            ),
        )

    @xax.jit(static_argnames=["self"], jit_level=JitLevel.MJX)
    def _physics_step(self, physics_model: mjx.Model, data: mjx.Data) -> mjx.Data:
        # Just performs the MJX step, but wraps it in it's own JIT which can be
        # cached to prevent heavy recompilation every time the rewards or
        # events change.
        return mjx.step(physics_model, data)

    @xax.jit(static_argnames=["self"], jit_level=JitLevel.ENGINE)
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

        # Randomly drops some actions.
        rng, drop_rng = jax.random.split(rng)
        drop_action = jax.random.bernoulli(drop_rng, self.drop_action_prob * curriculum_level, shape=action.shape)
        action = jnp.where(drop_action, prev_action, action)

        def move_physics(
            carry: tuple[mjx.Data, Array, xax.FrozenDict[str, PyTree], PyTree, Array],
            rng: PRNGKeyArray,
        ) -> tuple[tuple[mjx.Data, Array, xax.FrozenDict[str, PyTree], PyTree, Array], None]:
            data, step_num, event_states, actuator_state, last_torques = carry

            # Randomly apply the action with some latency.
            prct = jnp.clip(step_num - physics_state.action_latency, 0.0, 1.0)
            ctrl = prev_action * (1.0 - prct) + action * prct

            should_update = jnp.mod(step_num, self.phys_steps_per_actuator_step) == 0

            def update_actuators(prev_actuator_state: PyTree) -> tuple[Array, PyTree]:
                if isinstance(self.actuators, StatefulActuators):
                    torques, actuator_state = self.actuators.get_stateful_ctrl(
                        action=ctrl,
                        physics_data=data,
                        actuator_state=prev_actuator_state,
                        rng=rng,
                    )
                else:
                    torques = self.actuators.get_ctrl(action=ctrl, physics_data=data, rng=rng)
                    actuator_state = prev_actuator_state
                return torques, actuator_state

            torques, actuator_state = jax.lax.cond(
                should_update,
                lambda: update_actuators(actuator_state),
                lambda: (last_torques, actuator_state),
            )

            data = data.replace(ctrl=torques)

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

            new_data = self._physics_step(physics_model, data)
            return (new_data, step_num + 1.0, xax.FrozenDict(new_event_states), actuator_state, torques), None

        # Runs the model for N steps.
        (mjx_data, *_, event_info, actuator_state_final, _), _ = xax.scan(
            move_physics,
            # Note that we pass in 0 for initial last computed torques but step 0 will always update.
            (
                mjx_data,
                jnp.array(0.0),
                physics_state.event_states,
                physics_state.actuator_state,
                jnp.zeros_like(physics_state.data.ctrl),
            ),
            jax.random.split(rng, phys_steps_per_ctrl_steps),
            jit_level=JitLevel.ENGINE,
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
        mj_data = mujoco.MjData(physics_model)

        for reset in self.resets:
            rng, reset_rng = jax.random.split(rng)
            mj_data = reset(mj_data, curriculum_level, reset_rng)

        mujoco.mj_forward(physics_model, mj_data)

        # Zeros out some non-zeroed quantities.
        mj_data.qvel[:] = 0.0
        mj_data.qacc[:] = 0.0

        default_action = self.actuators.get_default_action(mj_data)
        actuator_state = (
            self.actuators.get_initial_state(mj_data, rng) if isinstance(self.actuators, StatefulActuators) else None
        )

        rng, latency_rng = jax.random.split(rng)

        return PhysicsState(
            data=mj_data,
            most_recent_action=default_action,
            event_states=self._reset_events(rng),
            actuator_state=actuator_state,
            action_latency=jax.random.uniform(
                latency_rng,
                minval=self.min_action_latency_step * curriculum_level,
                maxval=self.max_action_latency_step * curriculum_level,
            ),
        )

    def step(
        self,
        action: Array,
        physics_model: mujoco.MjModel,
        physics_state: PhysicsState,
        curriculum_level: Array,
        rng: PRNGKeyArray,
    ) -> PhysicsState:
        data = physics_state.data

        if not isinstance(data, mujoco.MjData):
            raise ValueError("Mujoco data is not a MjData")

        mujoco.mj_forward(physics_model, data)
        phys_steps_per_ctrl_steps = self.phys_steps_per_ctrl_steps
        prev_action = physics_state.most_recent_action

        # Randomly drops some actions.
        rng, drop_rng = jax.random.split(rng)
        drop_action = jax.random.bernoulli(drop_rng, self.drop_action_prob * curriculum_level, shape=action.shape)
        action = jnp.where(drop_action, prev_action, action)

        event_states = physics_state.event_states
        actuator_state = physics_state.actuator_state

        last_torques = jnp.zeros_like(action)

        for step_num in range(phys_steps_per_ctrl_steps):
            # Randomly apply the action with some latency.
            prct = jnp.clip(step_num - physics_state.action_latency, 0.0, 1.0)
            ctrl = prev_action * (1.0 - prct) + action * prct

            if (step_num % self.phys_steps_per_actuator_step) == 0:
                if isinstance(self.actuators, StatefulActuators):
                    torques, actuator_state = self.actuators.get_stateful_ctrl(
                        action=ctrl,
                        physics_data=data,
                        actuator_state=actuator_state,
                        rng=rng,
                    )
                else:
                    torques = self.actuators.get_ctrl(action=ctrl, physics_data=data, rng=rng)
                last_torques = torques
            else:
                torques = last_torques

            data.ctrl[:] = torques

            # Apply the events.
            new_event_states = {}
            for event in self.events:
                data, new_event_state = event(
                    model=physics_model,
                    data=data,
                    event_state=event_states[event.event_name],
                    curriculum_level=curriculum_level,
                    rng=rng,
                )
                new_event_states[event.event_name] = new_event_state
            event_states = xax.FrozenDict(new_event_states)

            mujoco.mj_step(physics_model, data)

        return PhysicsState(
            data=data,
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
    action_latency_range: tuple[float, float],
    drop_action_prob: float,
    actuator_update_dt: float | None,
) -> PhysicsEngine:
    min_action_latency, max_action_latency = action_latency_range
    if min_action_latency < 0:
        raise ValueError("`min_action_latency` must be non-negative")
    if min_action_latency > max_action_latency:
        raise ValueError("`min_action_latency` must be less than or equal to `max_action_latency`")
    if max_action_latency > ctrl_dt:
        logger.warning("`max_action_latency=%f` is greater than `ctrl_dt=%f`", max_action_latency, ctrl_dt)
    if (remainder := (ctrl_dt - round(ctrl_dt / dt) * dt)) > 1e-6:
        logger.warning("`ctrl_dt=%f` is not a multiple of `dt=%f` (remainder=%f)", ctrl_dt, dt, remainder)
    if drop_action_prob < 0 or drop_action_prob >= 1:
        raise ValueError("`drop_action_prob` must be between 0 and 1")

    # Converts to steps.
    min_action_latency_step = min_action_latency / dt
    max_action_latency_step = max_action_latency / dt
    phys_steps_per_ctrl_steps = round(ctrl_dt / dt)

    if actuator_update_dt is not None:
        phys_steps_per_actuator_step = max(1, round(actuator_update_dt / dt))
    else:
        phys_steps_per_actuator_step = 1

    match engine_type:
        case "mujoco":
            return MujocoEngine(
                resets=resets,
                events=events,
                actuators=actuators,
                min_action_latency_step=min_action_latency_step,
                max_action_latency_step=max_action_latency_step,
                phys_steps_per_ctrl_steps=phys_steps_per_ctrl_steps,
                drop_action_prob=drop_action_prob,
                phys_steps_per_actuator_step=phys_steps_per_actuator_step,
            )

        case "mjx":
            return MjxEngine(
                resets=resets,
                events=events,
                actuators=actuators,
                min_action_latency_step=min_action_latency_step,
                max_action_latency_step=max_action_latency_step,
                phys_steps_per_ctrl_steps=phys_steps_per_ctrl_steps,
                drop_action_prob=drop_action_prob,
                phys_steps_per_actuator_step=phys_steps_per_actuator_step,
            )

        case _:
            raise ValueError(f"Unsupported physics model type: {engine_type}")


def engine_type_from_physics_model(physics_model: PhysicsModel) -> EngineType:
    if isinstance(physics_model, mujoco.MjModel):
        return "mujoco"
    if isinstance(physics_model, mjx.Model):
        return "mjx"
    raise ValueError(f"Unsupported physics model type: {type(physics_model)}")
