"""MJX environment class."""

from typing import Collection

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx

from ksim.actuators import Actuators
from ksim.env.base_engine import PhysicsEngine
from ksim.env.data import PhysicsModel, PhysicsState
from ksim.resets import Reset


class MjxEngine(PhysicsEngine):
    """MJX engine class.

    NOTE: resetting and actuator logic live here, not during unrolling. This is
    - Actuators necessarily must get applied during unrolling.
    - Resets are not necessarily applied during unrolling.
    """

    def __init__(
        self,
        default_physics_model: PhysicsModel,
        resetters: Collection[Reset],
        actuators: Actuators,
        *,
        dt: float,
        ctrl_dt: float,
        min_action_latency_step: int,
        max_action_latency_step: int,
    ) -> None:
        """Initialize the MJX engine with resetting and actuators."""
        assert isinstance(default_physics_model, mjx.Model)
        self.default_mjx_model = default_physics_model
        self.actuators = actuators
        self.resetters = resetters
        self.ctrl_dt = ctrl_dt
        self.phys_steps_per_ctrl_steps = int(dt / ctrl_dt)
        self.min_action_latency_step = min_action_latency_step
        self.max_action_latency_step = max_action_latency_step

    def reset(self, rng: PRNGKeyArray) -> PhysicsState:
        """Reset the engine and return the physics model and data."""
        mjx_model = self.default_mjx_model
        mjx_data = mjx.make_data(mjx_model)

        # probably don't need to scan, fixed and small
        for resetter in self.resetters:
            rng = jax.random.split(rng, 1)[0]
            mjx_data = resetter(mjx_data, rng)

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
        """Step the engine and return the physics model and data."""
        mjx_model = state.model
        mjx_data = state.data
        mjx_data = mjx.forward(mjx_model, mjx_data)
        phys_steps_per_ctrl_steps = self.phys_steps_per_ctrl_steps
        prev_action = state.most_recent_action

        # TODO: probably incldue the model + data domain randomizer here...

        # NOTE: latency is untested...
        latency_steps = jax.random.randint(
            key=rng,
            shape=(),
            minval=self.min_action_latency_step,
            maxval=self.max_action_latency_step,
        )

        def move_physics(carry: tuple[mjx.Data, Array], _: None) -> tuple[tuple[mjx.Data, Array], None]:
            data, step_num = carry
            ctrl = jax.lax.select(
                step_num >= latency_steps,
                action,
                prev_action,
            )

            torques = self.actuators.get_ctrl(ctrl, data)
            data_with_ctrl = data.replace(ctrl=torques)
            data_with_ctrl = mjx.forward(mjx_model, data_with_ctrl)  # TODO: investigate if we can remove this
            new_data = mjx.step(mjx_model, data_with_ctrl)
            return (new_data, step_num + 1.0), None

        mjx_data = jax.lax.scan(move_physics, (mjx_data, jnp.array(0.0)), None, length=phys_steps_per_ctrl_steps)[0][0]

        return PhysicsState(model=mjx_model, data=mjx_data, most_recent_action=action)
