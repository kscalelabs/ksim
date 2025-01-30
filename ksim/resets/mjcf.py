"""Defines some useful resets for MJCF environments."""

import jax
import jax.numpy as jnp
from mujoco.mjx._src import math

from ksim.resets.base import Reset
from ksim.state.base import State


class XYPositionReset(Reset[State]):
    def __init__(self, x_range: tuple[float, float], y_range: tuple[float, float]) -> None:
        super().__init__()

        self.x_range = x_range
        self.y_range = y_range

    def __call__(self, state: State) -> State:
        rng, key = jax.random.split(state.rng)
        dx = jax.random.uniform(key, (1,), minval=self.x_range[0], maxval=self.x_range[1])
        dy = jax.random.uniform(key, (1,), minval=self.y_range[0], maxval=self.y_range[1])
        qpos_j = state.data.qpos
        qpos_j = qpos_j.at[0:1].set(qpos_j[0:1] + dx)
        qpos_j = qpos_j.at[1:2].set(qpos_j[1:2] + dy)
        return State(
            rng=rng,
            model=state.model,
            data=state.data.replace(qpos=qpos_j),
            done=state.done,
        )


class RandomYawReset(Reset[State]):
    def __init__(self, yaw_range: tuple[float, float] = (-jnp.pi, jnp.pi)) -> None:
        super().__init__()

        self.yaw_range = yaw_range

    def __call__(self, state: State) -> State:
        rng, key = jax.random.split(state.rng)
        yaw = jax.random.uniform(key, (1,), minval=self.yaw_range[0], maxval=self.yaw_range[1])
        quat = math.axis_angle_to_quat(jnp.array([0, 0, 1]), yaw)
        qpos = state.data.qpos
        new_quat = math.quat_mul(qpos[3:7], quat)
        qpos = qpos.at[3:7].set(new_quat)
        return State(
            rng=rng,
            model=state.model,
            data=state.data.replace(qpos=qpos),
            done=state.done,
        )


class VelocityReset(Reset[State]):
    def __init__(self, velocity_range: tuple[float, float] = (-0.5, 0.5)) -> None:
        super().__init__()
        self.velocity_range = velocity_range

    def __call__(self, state: State) -> State:
        rng, key = jax.random.split(state.rng)
        qvel = state.data.qvel
        qvel = qvel.at[0:6].set(
            jax.random.uniform(key, (6,), minval=self.velocity_range[0], maxval=self.velocity_range[1])
        )
        state.data.replace(qvel=qvel)
        return State(
            rng=rng,
            model=state.model,
            data=state.data.replace(qvel=qvel),
            done=state.done,
        )


class PerturbationReset(Reset[State]):
    def __init__(
        self,
        kick_wait_time_range: tuple[float, float],
        kick_duration_range: tuple[float, float],
        velocity_kick_range: tuple[float, float],
        dt: float,
    ) -> None:
        super().__init__()

        self.kick_wait_time_range = kick_wait_time_range
        self.kick_duration_range = kick_duration_range
        self.velocity_kick_range = velocity_kick_range
        self.dt = dt

    def __call__(self, state: State) -> State:
        rng, key1, key2, key3 = jax.random.split(state.rng, 4)

        # Time until next perturbation
        time_until_next_pert = jax.random.uniform(
            key1,
            minval=self.kick_wait_time_range[0],
            maxval=self.kick_wait_time_range[1],
        )
        steps_until_next_pert = jnp.round(time_until_next_pert / self.dt).astype(jnp.int32)

        # Perturbation duration
        pert_duration_seconds = jax.random.uniform(
            key2,
            minval=self.kick_duration_range[0],
            maxval=self.kick_duration_range[1],
        )
        pert_duration_steps = jnp.round(pert_duration_seconds / self.dt).astype(jnp.int32)

        # Perturbation magnitude
        pert_mag = jax.random.uniform(
            key3,
            minval=self.velocity_kick_range[0],
            maxval=self.velocity_kick_range[1],
        )

        return State(
            rng=rng,
            model=state.model,
            data=state.data.replace(
                steps_until_pert=steps_until_next_pert,
                pert_duration=pert_duration_steps,
                pert_magnitude=pert_mag,
            ),
            done=state.done,
        )


class CommandReset(Reset[State]):
    def __init__(self, command_range: tuple[float, float], dt: float, mean_time: float = 5.0) -> None:
        super().__init__()

        self.command_range = command_range
        self.dt = dt
        self.mean_time = mean_time

    def __call__(self, state: State) -> State:
        rng, key1, key2 = jax.random.split(state.rng, 3)

        # Time until next command
        time_until_next_cmd = jax.random.exponential(key1) * self.mean_time
        steps_until_next_cmd = jnp.round(time_until_next_cmd / self.dt).astype(jnp.int32)

        # Generate new command
        cmd = jax.random.uniform(key2, shape=(3,), minval=self.command_range[0], maxval=self.command_range[1])

        return state.replace(
            rng=rng,
            steps_until_command=steps_until_next_cmd,
            command=cmd,
        )
