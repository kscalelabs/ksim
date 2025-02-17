"""Defines some useful resets for MJCF environments."""

import jax
import jax.numpy as jnp
import numpy as np

from ksim.resets.base import Reset, ResetBuilder, ResetData
from ksim.utils.data import BuilderData


class XYPositionReset(Reset):
    bounds: tuple[float, float, float, float]
    padding_prct: float
    hfield_data: jnp.ndarray

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        hfield_data: np.ndarray,
        padding_prct: float = 0.1,
    ) -> None:
        """Initialize the reset.

        Args:
            bounds: Bounds for where to position the robot.
            hfield_data: The height field data.
            padding_prct: The percentage of the bounds to pad.
        """
        super().__init__()

        self.bounds = bounds
        self.padding_prct = padding_prct
        self.hfield_data = jnp.array(hfield_data)

    def __call__(self, data: ResetData) -> ResetData:
        x, y, _, _ = self.bounds

        minx = x * -0.5 * (1.0 - self.padding_prct)
        maxx = x * 0.5 * (1.0 - self.padding_prct)
        miny = y * -0.5 * (1.0 - self.padding_prct)
        maxy = y * 0.5 * (1.0 - self.padding_prct)

        # Generate random position
        rng, keyx, keyy = jax.random.split(data.rng, 3)
        dx = jax.random.uniform(keyx, (1,), minval=minx, maxval=maxx)
        dy = jax.random.uniform(keyy, (1,), minval=miny, maxval=maxy)

        # Update position while maintaining small height above ground
        qpos_j = data.state.q
        qpos_j = qpos_j.at[0:1].set(dx)
        qpos_j = qpos_j.at[1:2].set(dy)

        # Make sure the Z position is above the ground.
        z = self.hfield_data[dx.astype(int), dy.astype(int)]
        qpos_j = qpos_j.at[2:3].set(z)

        return ResetData(
            rng=rng,
            state=data.state.replace(q=qpos_j),
        )


class XYPositionResetBuilder(ResetBuilder[XYPositionReset]):
    def __init__(self, padding_prct: float = 0.1) -> None:
        super().__init__()

        self.padding_prct = padding_prct

    def __call__(self, data: BuilderData) -> XYPositionReset:
        x, y, ztop, zbottom = data.model.hfield_size.flatten().tolist()
        nx, ny = int(data.model.hfield_nrow), int(data.model.hfield_ncol)
        hfield_data = data.model.hfield_data.reshape(nx, ny)
        return XYPositionReset(
            bounds=(x, y, ztop, zbottom),
            hfield_data=hfield_data,
            padding_prct=self.padding_prct,
        )
