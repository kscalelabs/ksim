"""Defines some useful filters."""

__all__ = [
    "clip_acceleration",
    "ClipAccelerationParams",
    "integrate_velocity",
    "IntegrateVelocityParams",
    "integrate_acceleration",
    "IntegrateAccelerationParams",
    "ClipPositions",
    "TanhPositions",
    "LowPassFilterParams",
    "lowpass_one_pole",
    "lowpass_one_pole_scan",
]

from dataclasses import dataclass
from typing import Self

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from jaxtyping import Array

from ksim.debugging import JitLevel
from ksim.types import PhysicsModel


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
    static_argnames=["ctrl_dt", "max_acceleration"],
    donate_argnames=["params"],
    jit_level=JitLevel.HELPER_FUNCTIONS,
)
def clip_acceleration(
    target_position: Array,
    params: ClipAccelerationParams,
    max_acceleration: float,
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
    backprop: bool = True,
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
    backprop: bool = True,
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


class ClipPositions(eqx.Module):
    min_ranges: tuple[float, ...] = eqx.field()
    max_ranges: tuple[float, ...] = eqx.field()
    num_joints: int = eqx.field()

    def __init__(self, ranges: list[tuple[float, float]]) -> None:
        super().__init__()

        min_ranges, max_ranges = zip(*ranges, strict=True)
        self.min_ranges = tuple(min_ranges)
        self.max_ranges = tuple(max_ranges)
        self.num_joints = len(ranges)

    @classmethod
    def from_physics_model(cls, physics_model: PhysicsModel) -> Self:
        ranges = physics_model.jnt_range
        ranges_list = [(minv, maxv) for minv, maxv in ranges.tolist()[1:]]
        return cls(ranges_list)

    def clip(self, positions: Array) -> Array:
        chex.assert_shape(positions, (..., self.num_joints))
        return jnp.clip(positions, jnp.array(self.min_ranges), jnp.array(self.max_ranges))


class TanhPositions(eqx.Module):
    min_ranges: tuple[float, ...] = eqx.field()
    max_ranges: tuple[float, ...] = eqx.field()
    num_joints: int = eqx.field()

    def __init__(self, ranges: list[tuple[float, float]]) -> None:
        super().__init__()

        min_ranges, max_ranges = zip(*ranges, strict=True)
        self.min_ranges = tuple(min_ranges)
        self.max_ranges = tuple(max_ranges)
        self.num_joints = len(ranges)

    @classmethod
    def from_physics_model(cls, physics_model: PhysicsModel) -> Self:
        ranges = physics_model.jnt_range
        ranges_list = [(minv, maxv) for minv, maxv in ranges.tolist()[1:]]
        return cls(ranges_list)

    def clip(self, positions: Array) -> Array:
        chex.assert_shape(positions, (..., self.num_joints))
        minv, maxv = jnp.array(self.min_ranges), jnp.array(self.max_ranges)
        return (jnp.tanh(positions) * (maxv - minv)) + minv

    def get_bias(self, zeros: Array) -> Array:
        minv, maxv = jnp.array(self.min_ranges), jnp.array(self.max_ranges)
        return jnp.atanh((zeros - minv) / (maxv - minv))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LowPassFilterParams:
    y_last: Array

    @classmethod
    def initialize(cls, num_joints: int) -> Self:
        return cls(
            y_last=jnp.zeros(num_joints),
        )


def lowpass_one_pole(
    x: Array,
    ctrl_dt: float,
    fc: float | Array,
    params: LowPassFilterParams,
) -> tuple[Array, LowPassFilterParams]:
    rc = 1.0 / (2.0 * jnp.pi * fc)
    alpha = ctrl_dt / (rc + ctrl_dt)
    y_next = params.y_last + alpha * (x - params.y_last)
    return y_next, LowPassFilterParams(y_last=y_next)


@xax.jit(
    static_argnames=["ctrl_dt"],
    donate_argnames=["params"],
    jit_level=JitLevel.HELPER_FUNCTIONS,
)
def lowpass_one_pole_scan(
    x: Array,
    ctrl_dt: float,
    fc: float | Array,
    params: LowPassFilterParams,
) -> tuple[Array, LowPassFilterParams]:
    """Applies a one-pole low-pass filter to a signal.

    Args:
        x: The signal to filter, with shape (*, num_joints).
        ctrl_dt: The time step of the control loop.
        fc: The cutoff frequency.
        params: The parameters of the filter.

    Returns:
        The filtered signal, with shape (*, num_joints).
    """

    def step(y_prev: LowPassFilterParams, xi: Array) -> tuple[LowPassFilterParams, Array]:
        y_next, params = lowpass_one_pole(xi, ctrl_dt, fc, y_prev)
        return params, y_next

    y_last, y = xax.scan(step, params, x, jit_level=JitLevel.HELPER_FUNCTIONS)
    return y, y_last
