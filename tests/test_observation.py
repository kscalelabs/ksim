"""Tests for observations in the ksim package."""

from pathlib import Path

import attrs
import jax
import jax.numpy as jnp
import mujoco
import pytest
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx

import ksim

_TOL = 1e-4


@attrs.define(frozen=True)
class DummyObservation(ksim.Observation):
    """A dummy observation for testing."""

    def observe(self, state: ksim.PhysicsData, rng: PRNGKeyArray) -> Array:
        """Get a dummy observation from the state."""
        return jnp.zeros(1)

    def add_noise(self, observation: Array, rng: PRNGKeyArray) -> Array:
        """Add noise to the observation."""
        return observation


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


@pytest.fixture
def humanoid_model() -> mujoco.MjModel:
    """Create a humanoid model fixture for testing."""
    mjcf_path = (Path(__file__).parent / "fixed_assets" / "default_humanoid_test.mjcf").resolve().as_posix()
    return mujoco.MjModel.from_xml_path(mjcf_path)
