"""Tests for reset builders in the ksim package."""

import attrs
import jax
import jax.numpy as jnp
import mujoco
import pytest
import xax
from jaxtyping import Array

import ksim


@attrs.define(frozen=True)
class DummyReset(ksim.Reset):
    """Dummy reset for testing."""

    def __call__(self, data: ksim.PhysicsData, curriculum_level: Array, rng: jax.Array) -> dict[str, Array]:
        return {"qpos": jnp.zeros((3,))}


class DummyMjxData:
    """Mock mjx.Data for testing."""

    def __init__(self) -> None:
        self.qpos = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        self.model = DummyModel()

    def replace(self, **kwargs: dict[str, Array]) -> "DummyMjxData":
        """Mimics the behavior of mjx.Data.replace."""
        new_data = DummyMjxData()
        for key, value in kwargs.items():
            setattr(new_data, key, value)
        return new_data


class DummyModel:
    """Mock mjx.Model for testing."""

    def __init__(self) -> None:
        self.nq = 7
        self.hfield_nrow = 10
        self.hfield_ncol = 10
        self.hfield_size = jnp.array([[1.0, 1.0, 0.1, 0.1]])
        self.hfield_data = jnp.zeros((10, 10))


def test_reset_name() -> None:
    """Test that reset names are correctly generated."""
    reset = DummyReset()
    assert reset.get_name() == "dummy_reset"
    assert reset.reset_name == "dummy_reset"


class TestXYPositionResetBuilder:
    """Tests for the XYPositionResetBuilder class."""

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(0)

    def test_xy_position_reset(self, humanoid_model: mujoco.MjModel, rng: jax.Array) -> None:
        """Test that the XYPositionReset resets the XY position."""
        reset = ksim.get_xy_position_reset(humanoid_model)
        data = DummyMjxData()
        curriculum_level = jnp.array(0.0)
        result = reset(data, curriculum_level, rng)

        # Check that the result is a DummyMjxData object
        assert isinstance(result, DummyMjxData)


@pytest.mark.parametrize(
    "reset",
    [
        ksim.HFieldXYPositionReset(
            bounds=(0.0, 0.0, 0.0, 0.0),
            padded_bounds=(0.0, 0.0, 0.0, 0.0),
            x_range=1.0,
            y_range=1.0,
            hfield_data=xax.HashableArray(jnp.zeros((10, 10))),
        ),
        ksim.PlaneXYPositionReset(
            bounds=(0.0, 0.0, 0.0), padded_bounds=(0.0, 0.0, 0.0, 0.0), x_range=1.0, y_range=1.0, robot_base_height=0.0
        ),
    ],
)
def test_reset_hashable(reset: ksim.Reset) -> None:
    """Test that all resets are hashable."""
    assert hash(reset) is not None
