"""Tests for observation builders in the ksim package."""

import unittest
from typing import Literal

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx

from ksim.builders.observation import (
    BaseAngularVelocityObservation,
    BaseLinearVelocityObservation,
    BaseOrientationObservation,
    BasePositionObservation,
    JointPositionObservation,
    JointVelocityObservation,
    Observation,
    SensorObservation,
    SensorObservationBuilder,
)
from ksim.utils.data import BuilderData, MujocoMappings

_TOL = 1e-4


class DummyObservation(Observation):
    def __call__(self, state: mjx.Data, rng: PRNGKeyArray) -> Array:
        return jnp.zeros((3,))


class DummyMjxData:
    """Mock mjx.Data for testing."""

    def __init__(self) -> None:
        self._qpos = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        self._qvel = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self._sensordata = jnp.array([10.0, 11.0, 12.0, 13.0, 14.0])

    @property
    def qpos(self) -> Array:
        return self._qpos

    @property
    def qvel(self) -> Array:
        return self._qvel

    @property
    def sensordata(self) -> Array:
        return self._sensordata

    def replace(self, **kwargs: dict[str, Array]) -> "DummyMjxData":
        """Mimics the behavior of mjx.Data.replace."""
        new_data = DummyMjxData()
        for key, value in kwargs.items():
            setattr(new_data, f"_{key}", value)
        return new_data


class BaseObservationTest(chex.TestCase):
    def test_observation_name(self) -> None:
        obs = DummyObservation(noise=0.0, noise_type="gaussian")
        self.assertEqual(obs.get_name(), "dummy_observation")
        self.assertEqual(obs.observation_name, "dummy_observation")

    def test_add_gaussian_noise(self) -> None:
        rng = jax.random.PRNGKey(0)
        obs = DummyObservation(noise=1.0, noise_type="gaussian")
        clean_data = jnp.zeros((3,))
        noisy_data = obs.add_noise(clean_data, rng)

        # The noise should make the data non-zero
        self.assertFalse(jnp.array_equal(clean_data, noisy_data))

    def test_add_uniform_noise(self) -> None:
        rng = jax.random.PRNGKey(0)
        obs = DummyObservation(noise=1.0, noise_type="uniform")
        clean_data = jnp.zeros((3,))
        noisy_data = obs.add_noise(clean_data, rng)

        # The noise should make the data non-zero
        self.assertFalse(jnp.array_equal(clean_data, noisy_data))

        # Uniform noise should be bounded
        self.assertTrue(jnp.all(noisy_data >= -1.0))
        self.assertTrue(jnp.all(noisy_data <= 1.0))

    def test_gaussian_noise_statistics(self) -> None:
        # Test statistical properties of Gaussian noise
        rng = jax.random.PRNGKey(0)
        obs = DummyObservation(noise=2.0, noise_type="gaussian")
        clean_data = jnp.zeros((1000,))

        # Generate many noisy samples
        noisy_data = jax.vmap(lambda i: obs.add_noise(clean_data, jax.random.fold_in(rng, i)))(
            jnp.arange(100)
        )

        # Check statistical properties (mean close to 0, std dev close to noise value)
        mean = jnp.mean(noisy_data)
        std = jnp.std(noisy_data)

        self.assertLess(jnp.abs(mean), 0.1)
        self.assertLess(jnp.abs(std - 2.0), 0.2)

    def test_uniform_noise_statistics(self) -> None:
        # Test statistical properties of uniform noise
        rng = jax.random.PRNGKey(0)
        obs = DummyObservation(noise=3.0, noise_type="uniform")
        clean_data = jnp.zeros((1000,))

        # Generate many noisy samples
        noisy_data = jax.vmap(lambda i: obs.add_noise(clean_data, jax.random.fold_in(rng, i)))(
            jnp.arange(100)
        )

        # Check that all values are within bounds
        self.assertTrue(jnp.all(noisy_data >= -3.0))
        self.assertTrue(jnp.all(noisy_data <= 3.0))

        # Check that distribution is roughly uniform
        hist = jnp.histogram(noisy_data, bins=10, range=(-3.0, 3.0))[0]
        expected_count = len(noisy_data) / 10
        # Allow some variance but each bin should be reasonably populated
        self.assertTrue(jnp.all(hist > expected_count * 0.7))

    def test_invalid_noise_type(self) -> None:
        # Test that an invalid noise type raises an error
        rng = jax.random.PRNGKey(0)
        # Using a string literal that's not a valid noise type
        obs = DummyObservation(noise=1.0, noise_type="invalid_type")  # type: ignore[arg-type]
        clean_data = jnp.zeros((3,))

        with self.assertRaises(ValueError):
            obs.add_noise(clean_data, rng)

    def test_zero_noise(self) -> None:
        # Test that zero noise doesn't change the data
        rng = jax.random.PRNGKey(0)
        obs = DummyObservation(noise=0.0, noise_type="gaussian")
        clean_data = jnp.ones((3,))
        noisy_data = obs.add_noise(clean_data, rng)

        chex.assert_trees_all_close(clean_data, noisy_data)


class MjxObservationTest(chex.TestCase):
    def setUp(self) -> None:
        self.data = DummyMjxData()
        self.rng = jax.random.PRNGKey(0)

    def test_base_position_observation(self) -> None:
        obs = BasePositionObservation()
        result = obs(self.data, self.rng)
        chex.assert_shape(result, (3,))
        chex.assert_trees_all_close(
            result,
            jnp.array([1.0, 2.0, 3.0]),
            atol=_TOL,
        )

    def test_base_orientation_observation(self) -> None:
        obs = BaseOrientationObservation()
        result = obs(self.data, self.rng)
        chex.assert_shape(result, (4,))
        chex.assert_trees_all_close(
            result,
            jnp.array([4.0, 5.0, 6.0, 7.0]),
            atol=_TOL,
        )

    def test_base_linear_velocity_observation(self) -> None:
        obs = BaseLinearVelocityObservation()
        result = obs(self.data, self.rng)
        chex.assert_shape(result, (3,))
        chex.assert_trees_all_close(
            result,
            jnp.array([0.1, 0.2, 0.3]),
            atol=_TOL,
        )

    def test_base_angular_velocity_observation(self) -> None:
        obs = BaseAngularVelocityObservation()
        result = obs(self.data, self.rng)
        chex.assert_shape(result, (3,))
        chex.assert_trees_all_close(
            result,
            jnp.array([0.4, 0.5, 0.6]),
            atol=_TOL,
        )

    def test_joint_position_observation(self) -> None:
        obs = JointPositionObservation()
        result = obs(self.data, self.rng)
        chex.assert_shape(result, (2,))
        chex.assert_trees_all_close(
            result,
            jnp.array([8.0, 9.0]),
            atol=_TOL,
        )

    def test_joint_velocity_observation(self) -> None:
        obs = JointVelocityObservation()
        result = obs(self.data, self.rng)
        chex.assert_shape(result, (3,))
        chex.assert_trees_all_close(
            result,
            jnp.array([0.7, 0.8, 0.9]),
            atol=_TOL,
        )

    def test_base_position_observation_with_noise(self) -> None:
        obs = BasePositionObservation(noise=1.0, noise_type="gaussian")
        clean_result = obs(self.data, self.rng)

        # Run with different key to get different noise
        different_rng = jax.random.fold_in(self.rng, 1)
        noisy_result = obs(self.data, different_rng)

        # Results should be different due to noise
        self.assertFalse(jnp.array_equal(clean_result, noisy_result))

        # But the base data should be the same before noise
        expected_base = jnp.array([1.0, 2.0, 3.0])
        # Extract noise from first result
        noise = clean_result - expected_base
        self.assertFalse(jnp.allclose(noise, jnp.zeros_like(noise)))


class SensorObservationTest(chex.TestCase):
    def setUp(self) -> None:
        self.data = DummyMjxData()
        self.rng = jax.random.PRNGKey(0)

    def test_sensor_observation(self) -> None:
        obs = SensorObservation(
            sensor_name="test_sensor",
            sensor_idx_range=(1, 4),
        )
        result = obs(self.data, self.rng)
        chex.assert_shape(result, (3,))
        chex.assert_trees_all_close(
            result,
            jnp.array([11.0, 12.0, 13.0]),
            atol=_TOL,
        )

    def test_sensor_observation_with_noise(self) -> None:
        obs = SensorObservation(
            sensor_name="test_sensor",
            sensor_idx_range=(1, 4),
            noise=0.5,
            noise_type="gaussian",
        )
        result = obs(self.data, self.rng)
        noisy_result = obs.add_noise(result, self.rng)

        # The noise should make the data different
        self.assertFalse(jnp.array_equal(result, noisy_result))

    def test_sensor_observation_name(self) -> None:
        obs = SensorObservation(
            sensor_name="test_sensor",
            sensor_idx_range=(1, 4),
        )
        self.assertEqual(obs.get_name(), "test_sensor_sensor_observation")

    def test_noise_consistency(self) -> None:
        # Test that the same RNG key produces the same noise
        obs = SensorObservation(
            sensor_name="test_sensor",
            sensor_idx_range=(1, 4),
            noise=0.5,
            noise_type="gaussian",
        )

        # Get two noisy observations with the same key
        result1 = obs(self.data, self.rng)
        result2 = obs(self.data, self.rng)

        # They should be identical since we used the same key
        chex.assert_trees_all_close(result1, result2)

        # Now with different keys, they should be different
        different_rng = jax.random.fold_in(self.rng, 1)
        result3 = obs(self.data, different_rng)

        # Results should be different
        self.assertFalse(jnp.array_equal(result1, result3))


class SensorObservationBuilderTest(chex.TestCase):
    def setUp(self) -> None:
        # Create a BuilderData with sensor mappings
        self.mappings = MujocoMappings(
            sensor_name_to_idx_range={
                "imu_sensor": (0, 3),
                "force_sensor": (3, 6),
                "touch_sensor": (6, 10),
            },
            qpos_name_to_idx_range={},
            qvelacc_name_to_idx_range={},
            ctrl_name_to_idx={},
            geom_idx_to_body_name={},
        )
        self.builder_data = BuilderData(
            model=None,
            dt=0.004,
            ctrl_dt=0.02,
            mujoco_mappings=self.mappings,
        )

    def test_builder_creates_correct_observation(self) -> None:
        builder = SensorObservationBuilder(
            sensor_name="imu_sensor",
            noise=0.1,
            noise_type="uniform",
        )
        obs = builder(self.builder_data)

        self.assertIsInstance(obs, SensorObservation)
        self.assertEqual(obs.sensor_name, "imu_sensor")
        self.assertEqual(obs.sensor_idx_range, (0, 3))
        self.assertEqual(obs.noise, 0.1)
        self.assertEqual(obs.noise_type, "uniform")

    def test_builder_raises_error_for_invalid_sensor(self) -> None:
        builder = SensorObservationBuilder(sensor_name="nonexistent_sensor")

        with self.assertRaises(ValueError):
            builder(self.builder_data)

    def test_builder_noise_parameters(self) -> None:
        # Test that noise parameters are correctly passed through the builder
        noise_value = 2.5
        noise_type: Literal["gaussian", "uniform"] = "uniform"

        builder = SensorObservationBuilder(
            sensor_name="imu_sensor",
            noise=noise_value,
            noise_type=noise_type,
        )
        obs = builder(self.builder_data)

        # Check that the observation has the correct noise parameters
        self.assertEqual(obs.noise, noise_value)
        self.assertEqual(obs.noise_type, noise_type)


if __name__ == "__main__":
    unittest.main()
