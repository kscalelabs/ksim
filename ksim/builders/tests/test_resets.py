import unittest
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ksim.builders.resets import Reset, XYPositionReset, XYPositionResetBuilder
from ksim.utils.data import BuilderData


class DummyReset(Reset):
    """Dummy reset class for testing base functionality."""

    def __call__(self, data: Any, rng: PRNGKeyArray) -> Any:
        return data


class DummyMjxData:
    """Mock mjx.Data for testing."""

    def __init__(self):
        self._qpos = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def qpos(self) -> Array:
        return self._qpos

    @qpos.setter
    def qpos(self, value):
        self._qpos = value

    def replace(self, **kwargs):
        self._qpos = kwargs.get("qpos", self._qpos)
        return self


class BaseResetTest(chex.TestCase):
    def test_reset_name(self):
        reset = DummyReset()
        self.assertEqual(reset.get_name(), "dummy_reset")
        self.assertEqual(reset.reset_name, "dummy_reset")


class XYPositionResetTest(chex.TestCase):
    def setUp(self):
        self.data = DummyMjxData()
        self.rng = jax.random.PRNGKey(0)
        self.hfield_data = jnp.ones((10, 10))
        self.bounds = (5.0, 6.0, 2.0, 0.0)  # x, y, ztop, zbottom

    def test_xy_position_reset(self):
        reset = XYPositionReset(bounds=self.bounds, hfield_data=self.hfield_data)
        result = reset(self.data, self.rng)

        # Check that position was updated
        self.assertFalse(jnp.allclose(result.qpos[:3], jnp.zeros(3)))

        # Check that positions are within bounds (considering padding)
        x, y, ztop, _ = self.bounds
        padding = reset.padding_prct
        self.assertTrue(-x * (1.0 - padding) <= result.qpos[0] <= x * (1.0 - padding))
        self.assertTrue(-y * (1.0 - padding) <= result.qpos[1] <= y * (1.0 - padding))

        # Z should be hfield height (1.0) plus ztop
        # Need to check approximately since we're dealing with casted indices
        self.assertGreater(result.qpos[2], 0.0)

    def test_padding_affects_range(self):
        # Test with different padding values
        reset1 = XYPositionReset(bounds=self.bounds, hfield_data=self.hfield_data, padding_prct=0.0)
        reset2 = XYPositionReset(bounds=self.bounds, hfield_data=self.hfield_data, padding_prct=0.5)

        # Generate many resets to check the range
        samples = 100

        def sample_positions(reset, seed):
            positions = []
            for i in range(samples):
                key = jax.random.fold_in(seed, i)
                data_copy = DummyMjxData()
                result = reset(data_copy, key)
                positions.append((result.qpos[0], result.qpos[1]))
            return positions

        positions1 = sample_positions(reset1, self.rng)
        positions2 = sample_positions(reset2, self.rng)

        # Convert to arrays for easier analysis
        x1 = jnp.array([p[0] for p in positions1])
        y1 = jnp.array([p[1] for p in positions1])
        x2 = jnp.array([p[0] for p in positions2])
        y2 = jnp.array([p[1] for p in positions2])

        # The range of positions should be larger with less padding
        self.assertGreater(jnp.max(jnp.abs(x1)), jnp.max(jnp.abs(x2)))  # type: ignore
        self.assertGreater(jnp.max(jnp.abs(y1)), jnp.max(jnp.abs(y2)))  # type: ignore

    def test_different_keys_give_different_results(self):
        reset = XYPositionReset(bounds=self.bounds, hfield_data=self.hfield_data)

        # Generate two resets with different keys
        key1 = jax.random.PRNGKey(0)
        key2 = jax.random.PRNGKey(1)

        data1 = DummyMjxData()
        data2 = DummyMjxData()

        result1 = reset(data1, key1)  # type: ignore
        result2 = reset(data2, key2)  # type: ignore

        # Results should be different
        self.assertFalse(jnp.array_equal(result1.qpos, result2.qpos))


class XYPositionResetBuilderTest(chex.TestCase):
    def setUp(self):
        class DummyModel:
            hfield_size = jnp.array([5.0, 6.0, 2.0, 0.0])
            hfield_nrow = 10
            hfield_ncol = 12
            hfield_data = jnp.ones(10 * 12)

        self.builder_data = BuilderData(
            model=DummyModel(),
            dt=0.004,
            ctrl_dt=0.02,
            mujoco_mappings=None,  # type: ignore
        )

    def test_builder_creates_correct_reset(self):
        builder = XYPositionResetBuilder(padding_prct=0.2)
        reset = builder(self.builder_data)

        self.assertIsInstance(reset, XYPositionReset)
        self.assertEqual(reset.padding_prct, 0.2)
        self.assertEqual(reset.bounds, (5.0, 6.0, 2.0, 0.0))
        chex.assert_shape(reset.hfield_data, (10, 12))

    def test_default_padding(self):
        builder = XYPositionResetBuilder()
        reset = builder(self.builder_data)

        self.assertEqual(reset.padding_prct, 0.1)  # Default value


if __name__ == "__main__":
    unittest.main()
