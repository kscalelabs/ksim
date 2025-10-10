"""Base JAX centric environment class.

Designed to be easily extensible to any physics engine that separates
model and data.
"""

__all__ = [
    "EngineConfig",
    "PhysicsEngine",
    "MjxEngine",
    "MujocoEngine",
    "get_physics_engine",
    "engine_type_from_physics_model",
]

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Collection, Literal, Mapping, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import xax
from jaxtyping import Array, PRNGKeyArray, PyTree
from mujoco import mjx
from omegaconf import MISSING

from ksim.actuators import Actuators, StatefulActuators
from ksim.debugging import JitLevel
from ksim.events import Event
from ksim.resets import Reset
from ksim.types import PhysicsModel, PhysicsState

logger = logging.getLogger(__name__)

EngineType = Literal["mjx", "mujoco"]


@jax.tree_util.register_dataclass
@dataclass
class EngineConfig:
    action_latency_range: tuple[float, float] = xax.field(
        value=(0.0, 0.0),
        help="The range of action latencies to use.",
    )
    drop_action_prob: float = xax.field(
        value=0.0,
        help="The probability of dropping an action.",
    )
    actuator_update_dt: float | None = xax.field(
        value=None,
        help="The time step of the actuator update.",
    )
    ctrl_dt: float = xax.field(
        value=MISSING,
        help="The time step of the control loop.",
    )
    dt: float = xax.field(
        value=MISSING,
        help="The time step of the physics loop.",
    )

    def __post_init__(self) -> None:
        if self.dt is MISSING:
            raise ValueError("`dt` is required")
        if self.ctrl_dt is MISSING:
            raise ValueError("`ctrl_dt` is required")

        min_action_latency, max_action_latency = self.action_latency_range
        if min_action_latency < 0:
            raise ValueError("`min_action_latency` must be non-negative")
        if min_action_latency > max_action_latency:
            raise ValueError("`min_action_latency` must be less than or equal to `max_action_latency`")
        if max_action_latency > self.ctrl_dt:
            logger.warning("`max_action_latency=%f` is greater than `ctrl_dt=%f`", max_action_latency, self.ctrl_dt)
        if (remainder := (self.ctrl_dt - round(self.ctrl_dt / self.dt) * self.dt)) > 1e-6:
            logger.warning("`ctrl_dt=%f` is not a multiple of `dt=%f` (remainder=%f)", self.ctrl_dt, self.dt, remainder)

        if self.drop_action_prob < 0 or self.drop_action_prob >= 1:
            raise ValueError("`drop_action_prob` must be between 0 and 1")

    @property
    def min_action_latency_step(self) -> float:
        return self.action_latency_range[0] / self.dt

    @property
    def max_action_latency_step(self) -> float:
        return self.action_latency_range[1] / self.dt

    @property
    def phys_steps_per_ctrl_steps(self) -> int:
        return round(self.ctrl_dt / self.dt)

    @property
    def phys_steps_per_actuator_step(self) -> int:
        if self.actuator_update_dt is not None:
            return max(1, round(self.actuator_update_dt / self.dt))
        return 1


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EngineData:
    dt: float
    ctrl_dt: float
    drop_action_prob: float
    actuator_update_dt: float | None
    min_action_latency_step: float
    max_action_latency_step: float
    phys_steps_per_ctrl_steps: int
    phys_steps_per_actuator_step: int

    @classmethod
    def from_config(cls, config: EngineConfig) -> Self:
        # We need to build the static EngineData object from the config in
        # order to satisfy PyTree requirements.
        return cls(
            dt=config.dt,
            ctrl_dt=config.ctrl_dt,
            drop_action_prob=config.drop_action_prob,
            actuator_update_dt=config.actuator_update_dt,
            min_action_latency_step=config.min_action_latency_step,
            max_action_latency_step=config.max_action_latency_step,
            phys_steps_per_ctrl_steps=config.phys_steps_per_ctrl_steps,
            phys_steps_per_actuator_step=config.phys_steps_per_actuator_step,
        )


class PhysicsEngine(eqx.Module, ABC):
    """The role of an engine is simple: reset and step. Decoupled from data."""

    actuators: Actuators = eqx.field()
    resets: tuple[Reset, ...] = eqx.field()
    events: xax.FrozenDict[str, Event] = eqx.field()
    data: EngineData = eqx.field()

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
        for name, event in self.events.items():
            rng, event_rng = jax.random.split(rng)
            event_states[name] = event.get_initial_event_state(event_rng)
        return xax.freeze_dict(event_states)


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
                minval=self.data.min_action_latency_step * curriculum_level,
                maxval=self.data.max_action_latency_step * curriculum_level,
            ),
        )

    @xax.jit(
        static_argnames=["self"],
        donate_argnames=["data"],
        jit_level=JitLevel.MJX,
    )
    def _physics_step(self, physics_model: mjx.Model, data: mjx.Data) -> mjx.Data:
        # Just performs the MJX step, but wraps it in it's own JIT which can be
        # cached to prevent heavy recompilation every time the rewards or
        # events change.
        return mjx.step(physics_model, data)

    @xax.jit(
        static_argnames=["self"],
        donate_argnames=["physics_state", "rng"],
        jit_level=JitLevel.ENGINE,
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
        phys_steps_per_ctrl_steps = self.data.phys_steps_per_ctrl_steps
        prev_action = physics_state.most_recent_action

        # Randomly drops some actions.
        rng, drop_rng = jax.random.split(rng)
        drop_action = jax.random.bernoulli(drop_rng, self.data.drop_action_prob * curriculum_level, shape=action.shape)
        action = jnp.where(drop_action, prev_action, action)

        def move_physics(
            carry: tuple[mjx.Data, Array, PRNGKeyArray, xax.FrozenDict[str, PyTree], PyTree, Array],
            _: None,
        ) -> tuple[tuple[mjx.Data, Array, PRNGKeyArray, xax.FrozenDict[str, PyTree], PyTree, Array], None]:
            data, step_num, rng, event_states, actuator_state, last_torques = carry

            # Randomly apply the action with some latency.
            prct = jnp.clip(step_num - physics_state.action_latency, 0.0, 1.0)
            ctrl = prev_action * (1.0 - prct) + action * prct

            should_update = jnp.mod(step_num, self.data.phys_steps_per_actuator_step) == 0
            rng, rng_update = jax.random.split(rng)

            def update_actuators(prev_actuator_state: PyTree) -> tuple[Array, PyTree]:
                if isinstance(self.actuators, StatefulActuators):
                    torques, actuator_state = self.actuators.get_stateful_ctrl(
                        action=ctrl,
                        physics_data=data,
                        actuator_state=prev_actuator_state,
                        rng=rng_update,
                    )
                else:
                    torques = self.actuators.get_ctrl(
                        action=ctrl,
                        physics_data=data,
                        curriculum_level=curriculum_level,
                        rng=rng_update,
                    )
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
            for name, event in self.events.items():
                rng, rng_event = jax.random.split(rng)
                data, new_event_state = event(
                    model=physics_model,
                    data=data,
                    event_state=event_states[name],
                    curriculum_level=curriculum_level,
                    rng=rng_event,
                )
                new_event_states[name] = new_event_state

            new_data = self._physics_step(physics_model, data)
            return (new_data, step_num + 1.0, rng, xax.freeze_dict(new_event_states), actuator_state, torques), None

        # Runs the model for N steps.
        (mjx_data, *_, event_info, actuator_state_final, _), _ = xax.scan(
            move_physics,
            # Note that we pass in 0 for initial last computed torques but step 0 will always update.
            (
                mjx_data,
                jnp.array(0.0),
                rng,
                physics_state.event_states,
                physics_state.actuator_state,
                jnp.zeros_like(physics_state.data.ctrl),
            ),
            length=phys_steps_per_ctrl_steps,
            jit_level=JitLevel.ENGINE,
        )

        return PhysicsState(
            data=mjx_data,
            most_recent_action=action,
            event_states=xax.freeze_dict(event_info),
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
                minval=self.data.min_action_latency_step * curriculum_level,
                maxval=self.data.max_action_latency_step * curriculum_level,
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
        phys_steps_per_ctrl_steps = self.data.phys_steps_per_ctrl_steps
        prev_action = physics_state.most_recent_action

        # Randomly drops some actions.
        rng, drop_rng = jax.random.split(rng)
        drop_action = jax.random.bernoulli(drop_rng, self.data.drop_action_prob * curriculum_level, shape=action.shape)
        action = jnp.where(drop_action, prev_action, action)

        event_states = physics_state.event_states
        actuator_state = physics_state.actuator_state

        last_torques = jnp.zeros_like(action)

        for step_num in range(phys_steps_per_ctrl_steps):
            # Randomly apply the action with some latency.
            prct = jnp.clip(step_num - physics_state.action_latency, 0.0, 1.0)
            ctrl = prev_action * (1.0 - prct) + action * prct

            if (step_num % self.data.phys_steps_per_actuator_step) == 0:
                if isinstance(self.actuators, StatefulActuators):
                    torques, actuator_state = self.actuators.get_stateful_ctrl(
                        action=ctrl,
                        physics_data=data,
                        actuator_state=actuator_state,
                        rng=rng,
                    )
                else:
                    torques = self.actuators.get_ctrl(
                        action=ctrl,
                        physics_data=data,
                        curriculum_level=curriculum_level,
                        rng=rng,
                    )
                last_torques = torques
            else:
                torques = last_torques

            data.ctrl[:] = torques

            # Apply the events.
            new_event_states = {}
            for name, event in self.events.items():
                data, new_event_state = event(
                    model=physics_model,
                    data=data,
                    event_state=event_states[name],
                    curriculum_level=curriculum_level,
                    rng=rng,
                )
                new_event_states[name] = new_event_state
            event_states = xax.freeze_dict(new_event_states)

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
    events: Mapping[str, Event],
    actuators: Actuators,
    config: EngineConfig,
) -> PhysicsEngine:
    data = EngineData.from_config(config)

    match engine_type:
        case "mujoco":
            return MujocoEngine(
                resets=tuple(resets),
                events=xax.freeze_dict(events),
                actuators=actuators,
                data=data,
            )

        case "mjx":
            return MjxEngine(
                resets=tuple(resets),
                events=xax.freeze_dict(events),
                actuators=actuators,
                data=data,
            )

        case _:
            raise ValueError(f"Unsupported physics model type: {engine_type}")


def engine_type_from_physics_model(physics_model: PhysicsModel) -> EngineType:
    if isinstance(physics_model, mujoco.MjModel):
        return "mujoco"
    if isinstance(physics_model, mjx.Model):
        return "mjx"
    raise ValueError(f"Unsupported physics model type: {type(physics_model)}")
