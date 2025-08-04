"""Defines some useful filters."""

__all__ = [
    "clip_acceleration",
    "ClipAccelerationParams",
]

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array

from ksim.debugging import JitLevel


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ClipAccelerationParams:
    position: Array
    velocity: Array

    @classmethod
    def initialize(cls, num_joints: int) -> Self:
        return cls(
            position=jnp.zeros(num_joints),
            velocity=jnp.zeros(num_joints),
        )

    @classmethod
    def initialize_from(cls, init_position: Array) -> Self:
        return cls(
            position=init_position,
            velocity=jnp.zeros_like(init_position),
        )


@xax.jit(
    static_argnames=["max_acceleration", "ctrl_dt"],
    donate_argnames=["params"],
    jit_level=JitLevel.HELPER_FUNCTIONS,
)
def clip_acceleration(
    action: Array,
    params: ClipAccelerationParams,
    max_acceleration: float,
    ctrl_dt: float,
) -> ClipAccelerationParams:
    """Clips the maximum acceleration of an action.

    Args:
        action: The action to filter, which is inferred to be the target
            position for each joint, with shape (*, num_joints).
        params: The parameters of the filter.
        max_acceleration: The maximum acceleration of the action.
        ctrl_dt: The time step of the control loop.

    Returns:
        The new parameters of the filter.
    """
    cur_velocity = (action - params.position) / ctrl_dt
    cur_acceleration = (cur_velocity - params.velocity) / ctrl_dt
    clipped_acceleration = jnp.clip(cur_acceleration, -max_acceleration, max_acceleration)
    new_velocity = params.velocity + clipped_acceleration * ctrl_dt
    new_position = params.position + new_velocity * ctrl_dt
    return ClipAccelerationParams(new_position, new_velocity)
