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
        jax.debug.print("[400] MjxEngine.__init__: starting")
        assert isinstance(default_physics_model, mjx.Model)
        self.default_mjx_model = default_physics_model
        self.actuators = actuators
        self.resetters = resetters
        assert ctrl_dt % dt == 0, "ctrl_dt must be a multiple of dt"
        self.ctrl_dt = ctrl_dt
        self.phys_steps_per_ctrl_steps = int(ctrl_dt / dt)
        self.min_action_latency_step = min_action_latency_step
        self.max_action_latency_step = max_action_latency_step
        jax.debug.print("[401] MjxEngine.__init__: completed")

    def reset(self, rng: PRNGKeyArray) -> PhysicsState:
        """Reset the engine and return the physics model and data."""
        jax.debug.print("[500] MjxEngine.reset: starting")
        try:
            mjx_model = self.default_mjx_model
            mjx_data = mjx.make_data(mjx_model)
            jax.debug.print("[501] MjxEngine.reset: mjx_data created")

            # probably don't need to scan, fixed and small
            for resetter in self.resetters:
                rng = jax.random.split(rng, 1)[0]
                mjx_data = resetter(mjx_data, rng)
            jax.debug.print("[502] MjxEngine.reset: resetters applied")

            mjx_data = mjx.forward(mjx_model, mjx_data)
            jax.debug.print("[503] MjxEngine.reset: mjx.forward completed")
            assert isinstance(mjx_data, mjx.Data)
            default_action = mjx_data.ctrl

            return PhysicsState(
                model=mjx_model,
                data=mjx_data,
                most_recent_action=default_action,
            )
        except Exception as e:
            jax.debug.print("[ERR500] MjxEngine.reset: exception: {error}", error=str(e))
            raise

    def step(
        self,
        action: Array,
        state: PhysicsState,
        rng: PRNGKeyArray,
    ) -> PhysicsState:
        """Step the engine and return the physics model and data."""
        jax.debug.print("[600] MjxEngine.step: starting")
        try:
            mjx_model = state.model
            mjx_data = state.data
            
            jax.debug.print("[601] MjxEngine.step: calling mjx.forward")
            mjx_data = mjx.forward(mjx_model, mjx_data)
            jax.debug.print("[602] MjxEngine.step: mjx.forward completed")
            
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
            jax.debug.print("[603] MjxEngine.step: generated latency_steps={steps}", steps=latency_steps)

            def move_physics(carry: tuple[mjx.Data, Array], _: None) -> tuple[tuple[mjx.Data, Array], None]:
                data, step_num = carry
                jax.debug.print("[610] move_physics: step_num={step_num}", step_num=step_num)
                
                ctrl = jax.lax.select(
                    step_num >= latency_steps,
                    action,
                    prev_action,
                )
                jax.debug.print("[611] move_physics: ctrl shape={shape}, min={min}, max={max}", 
                             shape=ctrl.shape, min=jnp.min(ctrl), max=jnp.max(ctrl))

                try:
                    torques = self.actuators.get_ctrl(ctrl, data)
                    jax.debug.print("[612] move_physics: torques shape={shape}, min={min}, max={max}", 
                                 shape=torques.shape, min=jnp.min(torques), max=jnp.max(torques))
                except Exception as e:
                    jax.debug.print("[ERR610] move_physics: exception in get_ctrl: {error}", error=str(e))
                    raise
                
                try:
                    data_with_ctrl = data.replace(ctrl=torques)
                    jax.debug.print("[613] move_physics: data_with_ctrl created")
                except Exception as e:
                    jax.debug.print("[ERR611] move_physics: exception in data.replace: {error}", error=str(e))
                    raise
                
                try:
                    data_with_ctrl = mjx.forward(mjx_model, data_with_ctrl)
                    jax.debug.print("[614] move_physics: mjx.forward completed")
                except Exception as e:
                    jax.debug.print("[ERR612] move_physics: exception in mjx.forward: {error}", error=str(e))
                    raise
                
                try:
                    new_data = mjx.step(mjx_model, data_with_ctrl)
                    jax.debug.print("[615] move_physics: mjx.step completed")
                except Exception as e:
                    jax.debug.print("[ERR613] move_physics: exception in mjx.step: {error}", error=str(e))
                    raise
                
                return (new_data, step_num + 1.0), None

            jax.debug.print("[604] MjxEngine.step: about to execute scan over move_physics")
            mjx_data = jax.lax.scan(move_physics, (mjx_data, jnp.array(0.0)), None, length=phys_steps_per_ctrl_steps)[0][0]
            jax.debug.print("[605] MjxEngine.step: scan completed successfully")

            return PhysicsState(model=mjx_model, data=mjx_data, most_recent_action=action)
        except Exception as e:
            jax.debug.print("[ERR600] MjxEngine.step: exception: {error}", error=str(e))
            raise
