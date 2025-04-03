"""Defines some additional useful Distrax distributions."""

__all__ = [
    "AsymmetricBijector",
    "UnitIntervalToRangeBijector",
    "DoubleUnitIntervalToRangeBijector",
]

import distrax
import jax.numpy as jnp
from distrax._src.utils import conversion
from jaxtyping import Array


class AsymmetricBijector(distrax.Bijector):
    """A bijector that makes a distribution asymmetric.

    This maps a distribution with support `[-max, max]` to a distribution with
    support `[-min, max]`. Alternativly, this can be parametrized by providing
    `scale = min / max`.

    `scale` must be broadcastable to the shape of the input.

    NOTE: This expects that all the `min` values are strictly negative and
    all the `max` values are strictly positive - you will likely encounter
    NaN values if this is not the case. Similarly, `scale` should be strictly
    positive, if using the `scale` parametrization instead.
    """

    def __init__(
        self,
        min: Array | None = None,
        max: Array | None = None,
        scale: Array | None = None,
    ) -> None:
        super().__init__(event_ndims_in=0)

        if scale is None:
            assert min is not None and max is not None, "If scale is not provided, min and max must be provided."
            scale = -min / max
        else:
            assert min is None and max is None, "If scale is provided, min and max must not be provided."

        self._scale = conversion.as_float_array(scale)

    @property
    def scale(self) -> Array:
        return self._scale

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        return jnp.where(x < 0, jnp.log(self._scale), 0.0)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        y = jnp.where(x < 0, x * self._scale, x)
        return y, self.forward_log_det_jacobian(x)

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        x = jnp.where(y < 0, y / self._scale, y)
        return x, -self.forward_log_det_jacobian(x)

    def same_as(self, other: distrax.Bijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return type(other) is AsymmetricBijector and jnp.array_equal(self._scale, other._scale).all().item()


class UnitIntervalToRangeBijector(distrax.Bijector):
    """A bijector that maps a distribution with support `[0, 1]` to a distribution with support `[-min, max]`."""

    def __init__(self, min: Array, max: Array) -> None:
        super().__init__(event_ndims_in=0)
        self._min = conversion.as_float_array(min)
        self._max = conversion.as_float_array(max)

    @property
    def min(self) -> Array:
        return self._min

    @property
    def max(self) -> Array:
        return self._max

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        return jnp.log(self._max - self._min)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        y = (self._max - self._min) * x + self._min
        log_det = self.forward_log_det_jacobian(x)
        return y, log_det

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        # Transform from [min,max] to [0,1]
        x = (y - self._min) / (self._max - self._min)
        log_det = -self.forward_log_det_jacobian(x)  # Inverse log det is negative of forward
        return x, log_det

    def same_as(self, other: distrax.Bijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return (
            type(other) is UnitIntervalToRangeBijector
            and jnp.array_equal(self._min, other._min).all().item()
            and jnp.array_equal(self._max, other._max).all().item()
        )


class DoubleUnitIntervalToRangeBijector(distrax.Bijector):
    """A bijector that maps a distribution with support `[-1, 1]` to a distribution with support `[-min, max]`."""

    def __init__(self, min: Array, max: Array) -> None:
        super().__init__(event_ndims_in=0)
        self._min = conversion.as_float_array(min)
        self._max = conversion.as_float_array(max)

    @property
    def min(self) -> Array:
        return self._min

    @property
    def max(self) -> Array:
        return self._max

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        return jnp.log((self._max - self._min) / 2)

    def forward_and_log_det(self, x: Array) -> tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        # Transform from [-1,1] to [min,max]
        y = (self._max - self._min) / 2 * x + (self._max + self._min) / 2
        log_det = self.forward_log_det_jacobian(x)
        return y, log_det

    def inverse_and_log_det(self, y: Array) -> tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        x = (2 * y - (self._max + self._min)) / (self._max - self._min)
        log_det = -self.forward_log_det_jacobian(x)
        return x, log_det

    def same_as(self, other: distrax.Bijector) -> bool:
        """Returns True if this bijector is guaranteed to be the same as `other`."""
        return (
            type(other) is DoubleUnitIntervalToRangeBijector
            and jnp.array_equal(self._min, other._min).all().item()
            and jnp.array_equal(self._max, other._max).all().item()
        )
