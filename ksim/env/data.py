"""Base Types for Environments."""

from dataclasses import dataclass

import jax
import mujoco
from flax.core import FrozenDict
from jaxtyping import Array
from mujoco import mjx

PhysicsData = mjx.Data | mujoco.MjData
PhysicsModel = mjx.Model | mujoco.MjModel


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhysicsState:
    """Everything you need for the engine to take an action and step physics."""

    most_recent_action: Array  # since ctrl_dt > dt, we need this to simulate actuator logic
    model: PhysicsModel  # really important to think of this as a pointer of pointers... can be shared
    data: PhysicsData
    
    def __post_init__(self):
        # Add debug prints to show what's in the physics state
        try:
            jax.debug.print("PhysicsState created: model={model_type}, data={data_type}", 
                         model_type=type(self.model).__name__, 
                         data_type=type(self.data).__name__)
            
            if hasattr(self.data, 'qpos'):
                jax.debug.print("PhysicsState data: qpos shape={qpos_shape}, qvel shape={qvel_shape}, ctrl shape={ctrl_shape}",
                             qpos_shape=self.data.qpos.shape if hasattr(self.data, 'qpos') else None,
                             qvel_shape=self.data.qvel.shape if hasattr(self.data, 'qvel') else None, 
                             ctrl_shape=self.data.ctrl.shape if hasattr(self.data, 'ctrl') else None)
                
                # Check for extreme values which could cause issues
                if hasattr(self.data, 'qpos'):
                    qpos_max = jax.numpy.max(self.data.qpos) if self.data.qpos.size > 0 else None
                    qpos_min = jax.numpy.min(self.data.qpos) if self.data.qpos.size > 0 else None
                    jax.debug.print("PhysicsState data values: qpos min={qpos_min}, max={qpos_max}",
                                 qpos_min=qpos_min, qpos_max=qpos_max)
        except Exception as e:
            jax.debug.print("Error in PhysicsState.__post_init__: {error}", error=str(e))
            # Don't raise as this is diagnostic code


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Transition:
    obs: FrozenDict[str, Array]  # <- state
    command: FrozenDict[str, Array]  # <- prev command
    action: Array  # <- obs, command
    reward: Array  # <- state, action, next state
    done: Array  # <- state, action, next state
    timestep: Array  # <- state

    termination_components: FrozenDict[str, Array]  # The specific reasons the episode terminated.
    reward_components: FrozenDict[str, Array]  # The individual reward components, scaled.
