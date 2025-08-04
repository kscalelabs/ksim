"""Defines some useful filters."""

__all__ = [
    "clip_acceleration",
    "ClipAccelerationParams",
    "integrate_velocity",
    "IntegrateVelocityParams",
    "integrate_acceleration",
    "IntegrateAccelerationParams",
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
    static_argnames=["ctrl_dt"],
    donate_argnames=["params"],
    jit_level=JitLevel.HELPER_FUNCTIONS,
)
def clip_acceleration(
    target_position: Array,
    params: ClipAccelerationParams,
    max_acceleration: Array,
    ctrl_dt: float,
) -> ClipAccelerationParams:
    """Clips the maximum acceleration of an action.

    Args:
        target_position: The position to filter, with shape (*, num_joints).
        params: The parameters of the filter.
        max_acceleration: The maximum acceleration of the action.
        ctrl_dt: The time step of the control loop.

    Returns:
        The new parameters of the filter.
    """
    cur_velocity = (target_position - params.position) / ctrl_dt
    cur_acceleration = (cur_velocity - params.velocity) / ctrl_dt
    clipped_acceleration = jnp.clip(cur_acceleration, -max_acceleration, max_acceleration)
    new_velocity = params.velocity + clipped_acceleration * ctrl_dt
    new_position = params.position + new_velocity * ctrl_dt
    return ClipAccelerationParams(new_position, new_velocity)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IntegrateVelocityParams:
    position: Array

    @classmethod
    def initialize(cls, num_joints: int) -> Self:
        return cls(
            position=jnp.zeros(num_joints),
        )

    @classmethod
    def initialize_from(cls, init_position: Array) -> Self:
        return cls(
            position=init_position,
        )


@xax.jit(
    static_argnames=["ctrl_dt", "backprop"],
    donate_argnames=["params"],
    jit_level=JitLevel.HELPER_FUNCTIONS,
)
def integrate_velocity(
    velocity: Array,
    params: IntegrateVelocityParams,
    ctrl_dt: float,
    backprop: bool = False,
) -> IntegrateVelocityParams:
    """Integrates a velocity to a position.

    Args:
        velocity: The velocity to integrate, with shape (*, num_joints).
        params: The parameters of the filter.
        ctrl_dt: The time step of the control loop.
        backprop: Whether to backprop through the integration.

    Returns:
        The new parameters.
    """
    prev_position = params.position
    if not backprop:
        prev_position = jax.lax.stop_gradient(prev_position)
    cur_position = prev_position + velocity * ctrl_dt
    return IntegrateVelocityParams(cur_position)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IntegrateAccelerationParams:
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
    static_argnames=["ctrl_dt", "backprop"],
    donate_argnames=["params"],
    jit_level=JitLevel.HELPER_FUNCTIONS,
)
def integrate_acceleration(
    acceleration: Array,
    params: IntegrateAccelerationParams,
    ctrl_dt: float,
    max_velocity: float = 10.0,
    backprop: bool = False,
) -> IntegrateAccelerationParams:
    """Clips the maximum acceleration of an action.

    Args:
        acceleration: The acceleration to integrate, with shape (*, num_joints).
        params: The parameters of the filter.
        ctrl_dt: The time step of the control loop.
        max_velocity: The maximum velocity of the action.
        backprop: Whether to backprop through the integration.

    Returns:
        The new parameters.
    """
    prev_velocity, prev_position = params.velocity, params.position
    if not backprop:
        prev_velocity = jax.lax.stop_gradient(prev_velocity)
        prev_position = jax.lax.stop_gradient(prev_position)
    cur_velocity = jnp.clip(prev_velocity + acceleration * ctrl_dt, -max_velocity, max_velocity)
    cur_position = prev_position + cur_velocity * ctrl_dt
    return IntegrateAccelerationParams(cur_position, cur_velocity)
