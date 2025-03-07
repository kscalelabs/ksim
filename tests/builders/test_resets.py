"""Tests for reset builders in the ksim package."""

import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array
from mujoco import mjx

from ksim.builders.resets import (
    Reset,
    XYPositionResetBuilder,
)
from ksim.utils.data import BuilderData, MujocoMappings


class DummyReset(Reset):
    """Dummy reset for testing."""

    def __call__(self, data: mjx.Data, rng: jax.Array) -> dict[str, Array]:
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
    def builder_data(self) -> BuilderData:
        """Return a builder data object."""
        mappings = MujocoMappings(
            sensor_name_to_idx_range={},
            qpos_name_to_idx_range={},
            qvelacc_name_to_idx_range={},
            ctrl_name_to_idx={},
            geom_name_to_idx={},
            body_name_to_idx={},
            floor_geom_idx=None,
        )
        return BuilderData(
            model=DummyModel(),
            dt=0.004,
            ctrl_dt=0.02,
            mujoco_mappings=mappings,
        )

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(0)

    def test_xy_position_reset_builder(self, builder_data: BuilderData) -> None:
        """Test that the XYPositionResetBuilder creates a reset function."""
        builder = XYPositionResetBuilder()
        reset = builder(builder_data)
        assert reset.reset_name == "xyposition_reset"

    def test_xy_position_reset(self, builder_data: BuilderData, rng: jax.Array) -> None:
        """Test that the XYPositionReset resets the XY position."""
        builder = XYPositionResetBuilder()
        reset = builder(builder_data)
        data = DummyMjxData()
        result = reset(data, rng)

        # Check that the result is a DummyMjxData object
        assert isinstance(result, DummyMjxData)

        # Check that the qpos has the right shape
        assert result.qpos.shape == (7,)

        # Check that the XY position is exactly what we expect with RNG key 0
        expected_dx = jnp.array([-0.8868711])
        expected_dy = jnp.array([0.72440904])
        assert jnp.allclose(result.qpos[0:1], expected_dx, rtol=1e-5)
        assert jnp.allclose(result.qpos[1:2], expected_dy, rtol=1e-5)

        # Check that the rest of qpos is unchanged
        assert jnp.allclose(result.qpos[2:], data.qpos[2:])
