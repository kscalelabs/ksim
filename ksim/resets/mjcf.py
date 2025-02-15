"""Defines some useful resets for MJCF environments."""

import jax

from ksim.resets.base import Reset, ResetData


class XYPositionReset(Reset):
    """Resets the position of the robot to a random position within a given range."""

    x_range: tuple[float, float]
    y_range: tuple[float, float]

    def __init__(self, x_range: tuple[float, float], y_range: tuple[float, float]) -> None:
        super().__init__()

        self.x_range = x_range
        self.y_range = y_range

    def __call__(self, data: ResetData) -> ResetData:
        rng, keyx, keyy = jax.random.split(data.rng, 3)
        dx = jax.random.uniform(keyx, (1,), minval=self.x_range[0], maxval=self.x_range[1])
        dy = jax.random.uniform(keyy, (1,), minval=self.y_range[0], maxval=self.y_range[1])
        qpos_j = data.state.q
        qpos_j = qpos_j.at[0:1].set(qpos_j[0:1] + dx)
        qpos_j = qpos_j.at[1:2].set(qpos_j[1:2] + dy)
        return ResetData(
            rng=rng,
            state=data.state.replace(q=qpos_j),
        )
