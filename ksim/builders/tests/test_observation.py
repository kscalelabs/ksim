import unittest
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array

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
    def __call__(self, state: Any) -> Array:
        return jnp.zeros((3,))


class DummyMjxData:
    """Mock mjx.Data for testing."""

    @property
    def qpos(self) -> Array:
        return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    @property
    def qvel(self) -> Array:
        return jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    @property
    def sensordata(self) -> Array:
        return jnp.array([10.0, 11.0, 12.0, 13.0, 14.0])


class BaseObservationTest(chex.TestCase):
    def test_observation_name(self):
        obs = DummyObservation(noise=0.0, noise_type="gaussian")
        self.assertEqual(obs.get_name(), "dummy_observation")
        self.assertEqual(obs.observation_name, "dummy_observation")

    def test_add_gaussian_noise(self):
        rng = jax.random.PRNGKey(0)
        obs = DummyObservation(noise=1.0, noise_type="gaussian")
        clean_data = jnp.zeros((3,))
        noisy_data = obs.add_noise(clean_data, rng)

        # The noise should make the data non-zero
        self.assertFalse(jnp.array_equal(clean_data, noisy_data))

    def test_add_uniform_noise(self):
        rng = jax.random.PRNGKey(0)
        obs = DummyObservation(noise=1.0, noise_type="uniform")
        clean_data = jnp.zeros((3,))
        noisy_data = obs.add_noise(clean_data, rng)

        # The noise should make the data non-zero
        self.assertFalse(jnp.array_equal(clean_data, noisy_data))

        # Uniform noise should be bounded
        self.assertTrue(jnp.all(noisy_data >= -1.0))
        self.assertTrue(jnp.all(noisy_data <= 1.0))

    def test_gaussian_noise_statistics(self):
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

        self.assertLess(jnp.abs(mean), 0.1)  # type: ignore
        self.assertLess(jnp.abs(std - 2.0), 0.2)  # type: ignore

    def test_uniform_noise_statistics(self):
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

    def test_invalid_noise_type(self):
        # Test that an invalid noise type raises an error
        rng = jax.random.PRNGKey(0)
        obs = DummyObservation(noise=1.0, noise_type="invalid_type")  # type: ignore
        clean_data = jnp.zeros((3,))

        with self.assertRaises(ValueError):
            obs.add_noise(clean_data, rng)

    def test_zero_noise(self):
        # Test that zero noise doesn't change the data
        rng = jax.random.PRNGKey(0)
        obs = DummyObservation(noise=0.0, noise_type="gaussian")
        clean_data = jnp.ones((3,))
        noisy_data = obs.add_noise(clean_data, rng)

        chex.assert_trees_all_close(clean_data, noisy_data)


class MjxObservationTest(chex.TestCase):
    def setUp(self):
        self.data = DummyMjxData()
        self.rng = jax.random.PRNGKey(0)

    def test_base_position_observation(self):
        obs = BasePositionObservation()
        result = obs(self.data, self.rng)  # type: ignore
        chex.assert_shape(result, (3,))
        chex.assert_trees_all_close(
            result,
            jnp.array([1.0, 2.0, 3.0]),
            atol=_TOL,
        )

    def test_base_orientation_observation(self):
        obs = BaseOrientationObservation()
        result = obs(self.data, self.rng)  # type: ignore
        chex.assert_shape(result, (4,))
        chex.assert_trees_all_close(
            result,
            jnp.array([4.0, 5.0, 6.0, 7.0]),
            atol=_TOL,
        )

    def test_base_linear_velocity_observation(self):
        obs = BaseLinearVelocityObservation()
        result = obs(self.data, self.rng)  # type: ignore
        chex.assert_shape(result, (3,))
        chex.assert_trees_all_close(
            result,
            jnp.array([0.1, 0.2, 0.3]),
            atol=_TOL,
        )

    def test_base_angular_velocity_observation(self):
        obs = BaseAngularVelocityObservation()
        result = obs(self.data, self.rng)  # type: ignore
        chex.assert_shape(result, (3,))
        chex.assert_trees_all_close(
            result,
            jnp.array([0.4, 0.5, 0.6]),
            atol=_TOL,
        )

    def test_joint_position_observation(self):
        obs = JointPositionObservation()
        result = obs(self.data, self.rng)  # type: ignore
        chex.assert_shape(result, (2,))
        chex.assert_trees_all_close(
            result,
            jnp.array([8.0, 9.0]),
            atol=_TOL,
        )

    def test_joint_velocity_observation(self):
        obs = JointVelocityObservation()
        result = obs(self.data, self.rng)  # type: ignore
        chex.assert_shape(result, (3,))
        chex.assert_trees_all_close(
            result,
            jnp.array([0.7, 0.8, 0.9]),
            atol=_TOL,
        )

    def test_base_position_observation_with_noise(self):
        obs = BasePositionObservation(noise=1.0, noise_type="gaussian")
        clean_result = obs(self.data, self.rng)  # type: ignore

        # Run with different key to get different noise
        different_rng = jax.random.fold_in(self.rng, 1)
        noisy_result = obs(self.data, different_rng)  # type: ignore

        # Results should be different due to noise
        self.assertFalse(jnp.array_equal(clean_result, noisy_result))

        # But the base data should be the same before noise
        expected_base = jnp.array([1.0, 2.0, 3.0])
        # Extract noise from first result
        noise = clean_result - expected_base
        self.assertFalse(jnp.allclose(noise, jnp.zeros_like(noise)))


class SensorObservationTest(chex.TestCase):
    def setUp(self):
        self.data = DummyMjxData()
        self.rng = jax.random.PRNGKey(0)

    def test_sensor_observation(self):
        obs = SensorObservation(
            sensor_name="test_sensor",
            sensor_idx_range=(1, 4),
        )
        result = obs(self.data, self.rng)  # type: ignore
        chex.assert_shape(result, (3,))
        chex.assert_trees_all_close(
            result,
            jnp.array([11.0, 12.0, 13.0]),
            atol=_TOL,
        )

    def test_sensor_observation_with_noise(self):
        obs = SensorObservation(
            sensor_name="test_sensor",
            sensor_idx_range=(1, 4),
            noise=0.5,
            noise_type="gaussian",
        )
        result = obs(self.data, self.rng)  # type: ignore
        noisy_result = obs.add_noise(result, self.rng)  # type: ignore

        # The noise should make the data different
        self.assertFalse(jnp.array_equal(result, noisy_result))

    def test_sensor_observation_name(self):
        obs = SensorObservation(
            sensor_name="test_sensor",
            sensor_idx_range=(1, 4),
        )
        self.assertEqual(obs.get_name(), "test_sensor_sensor_observation")

    def test_noise_consistency(self):
        # Test that the same RNG key produces the same noise
        obs = SensorObservation(
            sensor_name="test_sensor",
            sensor_idx_range=(1, 4),
            noise=0.5,
            noise_type="gaussian",
        )

        # Get two noisy observations with the same key
        result1 = obs(self.data, self.rng)  # type: ignore
        result2 = obs(self.data, self.rng)  # type: ignore

        # They should be identical since we used the same key
        chex.assert_trees_all_close(result1, result2)

        # Now with different keys, they should be different
        different_rng = jax.random.fold_in(self.rng, 1)
        result3 = obs(self.data, different_rng)  # type: ignore

        # Results should be different
        self.assertFalse(jnp.array_equal(result1, result3))


class SensorObservationBuilderTest(chex.TestCase):
    def setUp(self):
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

    def test_builder_creates_correct_observation(self):
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

    def test_builder_raises_error_for_invalid_sensor(self):
        builder = SensorObservationBuilder(sensor_name="nonexistent_sensor")

        with self.assertRaises(ValueError):
            builder(self.builder_data)

    def test_builder_noise_parameters(self):
        # Test that noise parameters are correctly passed through the builder
        noise_value = 2.5
        noise_type = "uniform"

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
