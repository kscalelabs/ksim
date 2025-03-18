"""Tests for observation builders in the ksim package."""

import unittest
from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp
import mujoco
import pytest
from jaxtyping import Array, PRNGKeyArray

from ksim.env.data import PhysicsData
from ksim.observation import (
    BaseAngularVelocityObservation,
    BaseLinearVelocityObservation,
    BaseOrientationObservation,
    BasePositionObservation,
    JointPositionObservation,
    JointVelocityObservation,
    Observation,
    SensorObservation,
)

_TOL = 1e-4


class DummyObservation(Observation):
    def observe(self, state: PhysicsData, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(1)


@jax.tree_util.register_dataclass
@dataclass
class DummyMjxData:
    """Mock mjx.Data for testing."""

    qpos: Array
    qvel: Array
    sensordata: Array


@pytest.fixture
def default_mjx_data() -> DummyMjxData:
    return DummyMjxData(
        qpos=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        qvel=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        sensordata=jnp.array([10.0, 11.0, 12.0, 13.0, 14.0]),
    )


def test_base_position_observation(default_mjx_data: DummyMjxData) -> None:
    rng = jax.random.PRNGKey(0)
    obs = BasePositionObservation()
    result = obs(default_mjx_data, rng)
    chex.assert_shape(result, (3,))
    chex.assert_trees_all_close(
        result,
        jnp.array([1.0, 2.0, 3.0]),
        atol=_TOL,
    )


def test_base_orientation_observation(default_mjx_data: DummyMjxData) -> None:
    rng = jax.random.PRNGKey(0)
    obs = BaseOrientationObservation()
    result = obs(default_mjx_data, rng)
    chex.assert_shape(result, (4,))
    chex.assert_trees_all_close(
        result,
        jnp.array([4.0, 5.0, 6.0, 7.0]),
        atol=_TOL,
    )


def test_base_linear_velocity_observation(default_mjx_data: DummyMjxData) -> None:
    rng = jax.random.PRNGKey(0)
    obs = BaseLinearVelocityObservation()
    result = obs(default_mjx_data, rng)
    chex.assert_shape(result, (3,))
    chex.assert_trees_all_close(
        result,
        jnp.array([0.1, 0.2, 0.3]),
        atol=_TOL,
    )


def test_base_angular_velocity_observation(default_mjx_data: DummyMjxData) -> None:
    rng = jax.random.PRNGKey(0)
    obs = BaseAngularVelocityObservation()
    result = obs(default_mjx_data, rng)
    chex.assert_shape(result, (3,))
    chex.assert_trees_all_close(
        result,
        jnp.array([0.4, 0.5, 0.6]),
        atol=_TOL,
    )


def test_joint_position_observation(default_mjx_data: DummyMjxData) -> None:
    rng = jax.random.PRNGKey(0)
    obs = JointPositionObservation()
    result = obs(default_mjx_data, rng)
    chex.assert_shape(result, (2,))
    chex.assert_trees_all_close(
        result,
        jnp.array([8.0, 9.0]),
        atol=_TOL,
    )


def test_joint_velocity_observation(default_mjx_data: DummyMjxData) -> None:
    rng = jax.random.PRNGKey(0)
    obs = JointVelocityObservation()
    result = obs(default_mjx_data, rng)
    chex.assert_shape(result, (3,))
    chex.assert_trees_all_close(
        result,
        jnp.array([0.7, 0.8, 0.9]),
        atol=_TOL,
    )


def test_base_position_observation_with_noise(default_mjx_data: DummyMjxData) -> None:
    rng = jax.random.PRNGKey(0)
    obs = BasePositionObservation(noise=1.0)
    clean_result = obs(default_mjx_data, rng)

    # Run with different key to get different noise
    different_rng = jax.random.fold_in(rng, 1)
    noisy_result = obs(default_mjx_data, different_rng)

    # Results should be different due to noise
    assert not jnp.array_equal(clean_result, noisy_result)

    # But the base data should be the same before noise
    expected_base = jnp.array([1.0, 2.0, 3.0])
    # Extract noise from first result
    noise = clean_result - expected_base
    assert not jnp.allclose(noise, jnp.zeros_like(noise))


def test_sensor_observation(default_mjx_data: DummyMjxData) -> None:
    rng = jax.random.PRNGKey(0)
    obs = SensorObservation(
        sensor_name="test_sensor",
        sensor_idx_range=(1, 4),
    )
    result = obs(default_mjx_data, rng)
    chex.assert_shape(result, (3,))
    chex.assert_trees_all_close(
        result,
        jnp.array([11.0, 12.0, 13.0]),
        atol=_TOL,
    )


def test_sensor_observation_with_noise(default_mjx_data: DummyMjxData) -> None:
    rng = jax.random.PRNGKey(0)
    obs = SensorObservation(
        sensor_name="test_sensor",
        sensor_idx_range=(1, 4),
        noise=0.5,
        noise_type="gaussian",
    )
    result = obs(default_mjx_data, rng)
    noisy_result = obs.add_noise(result, rng)

    # The noise should make the data different
    assert not jnp.array_equal(result, noisy_result)


def test_noise_consistency(default_mjx_data: DummyMjxData) -> None:
    rng = jax.random.PRNGKey(0)

    # Test that the same RNG key produces the same noise
    obs = SensorObservation(
        sensor_name="test_sensor",
        sensor_idx_range=(1, 4),
        noise=0.5,
        noise_type="gaussian",
    )

    # Get two noisy observations with the same key
    result1 = obs(default_mjx_data, rng)
    result2 = obs(default_mjx_data, rng)

    # They should be identical since we used the same key
    chex.assert_trees_all_close(result1, result2)

    # Now with different keys, they should be different
    different_rng = jax.random.fold_in(rng, 1)
    result3 = obs(default_mjx_data, different_rng)

    # Results should be different
    assert not jnp.array_equal(result1, result3)


def test_builder_creates_correct_observation(humanoid_model: mujoco.MjModel) -> None:
    obs = SensorObservation.create(
        humanoid_model,
        sensor_name="imu_acc",
        noise=0.1,
        noise_type="uniform",
    )

    assert isinstance(obs, SensorObservation)
    assert obs.sensor_name == "imu_acc"
    assert obs.sensor_idx_range == (0, 3)
    assert obs.noise == 0.1
    assert obs.noise_type == "uniform"


def test_builder_raises_error_for_invalid_sensor(humanoid_model: mujoco.MjModel) -> None:
    with pytest.raises(ValueError):
        SensorObservation.create(
            humanoid_model,
            sensor_name="nonexistent_sensor",
            noise=0.1,
            noise_type="uniform",
        )


if __name__ == "__main__":
    unittest.main()
