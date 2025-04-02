"""Tests for observations in the ksim package."""

import attrs
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx

import ksim


@attrs.define(frozen=True)
class DummyObservation(ksim.Observation):
    """A dummy observation for testing."""

    def observe(self, state: ksim.ObservationState, rng: PRNGKeyArray) -> Array:
        """Get a dummy observation from the state."""
        return jnp.zeros(1)


class DummyMjxData(mjx.Data):
    """Mock mjx.Data for testing."""

    qpos: Array
    qvel: Array
    sensordata: Array


@pytest.fixture
def default_mjx_data() -> DummyMjxData:
    """Create a default mjx data fixture for testing."""
    return DummyMjxData(
        qpos=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        qvel=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        sensordata=jnp.array([10.0, 11.0, 12.0, 13.0, 14.0]),
    )


@pytest.fixture
def rng() -> PRNGKeyArray:
    """Create a default RNG key fixture for testing."""
    return jax.random.PRNGKey(0)
