"""Tests for command builders in the ksim package."""

import attrs
import chex
import jax
import jax.numpy as jnp
import mujoco
import pytest
from jaxtyping import Array

import ksim

_TOL = 1e-4


@attrs.define(frozen=True)
class DummyCommand(ksim.Command):
    def initial_command(self, physics_data: ksim.PhysicsData, rng: jax.Array) -> jnp.ndarray:
        return jnp.zeros((1,))

    def __call__(self, prev_command: Array, physics_data: ksim.PhysicsData, rng: jax.Array) -> jnp.ndarray:
        return jnp.zeros((1,))


def get_simple_model() -> mujoco.MjModel:
    """Get a simple model for testing."""
    xml = """
    <mujoco>
        <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0" />
            <body name="torso" pos="0 0 0.5">
                <joint name="free_joint" type="free"/>
                <geom name="torso_geom" type="sphere" size="0.1" />
                <body name="body1" pos="0.2 0 0">
                    <joint name="joint1" type="hinge" axis="0 0 1" damping="0.1" armature="0.1" frictionloss="0.1" />
                    <geom name="body1_geom" type="capsule" size="0.05" fromto="0 0 0 0.2 0 0" />
                    <body name="body2" pos="0.2 0 0">
                        <joint name="joint2" type="hinge" axis="0 1 0"
                        damping="0.1" armature="0.1" frictionloss="0.1" />
                        <geom name="body2_geom" type="capsule" size="0.05" fromto="0 0 0 0.2 0 0" />
                    </body>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """
    mj_model = mujoco.MjModel.from_xml_string(xml)
    return mujoco.MjData(mj_model)


class TestLinearVelocityCommand:
    """Tests for the LinearVelocityCommand class."""

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def physics_data(self) -> ksim.PhysicsData:
        """Get a simple model for testing."""
        return get_simple_model()

    def test_command_shape(self, rng: jax.Array, physics_data: ksim.PhysicsData) -> None:
        """Test that the command returns the correct shape."""
        cmd = ksim.LinearVelocityCommand(x_range=(-1.0, 1.0), y_range=(-2.0, 2.0))
        initial_command = cmd.initial_command(physics_data, rng)
        result = cmd(initial_command, physics_data, rng)
        chex.assert_shape(result, (2,))

    def test_command_bounds(self, rng: jax.Array, physics_data: ksim.PhysicsData) -> None:
        """Test that the command values are within the expected bounds."""
        x_scale, y_scale = 2.0, 3.0
        cmd = ksim.LinearVelocityCommand(x_range=(-x_scale, x_scale), y_range=(-y_scale, y_scale))
        command = cmd.initial_command(physics_data, rng)

        # Run multiple times to test bounds probabilistically
        for i in range(100):
            key = jax.random.fold_in(rng, i)
            command = cmd(command, physics_data, key)

            assert command[0] >= -x_scale
            assert command[0] <= x_scale
            assert command[1] >= -y_scale
            assert command[1] <= y_scale

    def test_zero_probability(self, rng: jax.Array, physics_data: ksim.PhysicsData) -> None:
        """Test that the command returns zeros when zero_prob is 1.0."""
        cmd = ksim.LinearVelocityCommand(x_range=(-1.0, 1.0), y_range=(-2.0, 2.0), x_zero_prob=1.0, y_zero_prob=1.0)
        command = cmd.initial_command(physics_data, rng)
        result = cmd(command, physics_data, rng)
        chex.assert_trees_all_close(
            result,
            jnp.zeros_like(result),
            atol=_TOL,
        )

    def test_update_mechanism(self, rng: jax.Array, physics_data: ksim.PhysicsData) -> None:
        """Test that the command update mechanism works correctly."""
        cmd = ksim.LinearVelocityCommand(x_range=(-1.0, 1.0), y_range=(-2.0, 2.0))
        command = cmd.initial_command(physics_data, rng)

        next_command = cmd(command, physics_data, rng)
        assert jnp.array_equal(next_command, command)


class TestAngularVelocityCommand:
    """Tests for the AngularVelocityCommand class."""

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def physics_data(self) -> ksim.PhysicsData:
        """Get a simple model for testing."""
        return get_simple_model()

    def test_command_shape(self, rng: jax.Array, physics_data: ksim.PhysicsData) -> None:
        """Test that the command returns the correct shape."""
        cmd = ksim.AngularVelocityCommand(scale=1.0)
        initial_command = cmd.initial_command(physics_data, rng)
        result = cmd(initial_command, physics_data, rng)
        chex.assert_shape(result, (1,))

    def test_command_bounds(self, rng: jax.Array, physics_data: ksim.PhysicsData) -> None:
        """Test that the command values are within the expected bounds."""
        scale = 2.0
        cmd = ksim.AngularVelocityCommand(scale=scale)

        # Run multiple times to test bounds probabilistically
        for i in range(100):
            key = jax.random.fold_in(rng, i)
            command = cmd.initial_command(physics_data, key)
            result = cmd(command, physics_data, key)

            assert result[0] >= -scale
            assert result[0] <= scale

    def test_zero_probability(self, rng: jax.Array, physics_data: ksim.PhysicsData) -> None:
        """Test that the command returns zeros when zero_prob is 1.0."""
        cmd = ksim.AngularVelocityCommand(scale=1.0, zero_prob=1.0)
        command = cmd.initial_command(physics_data, rng)
        result = cmd(command, physics_data, rng)
        chex.assert_trees_all_close(
            result,
            jnp.zeros_like(result),
            atol=_TOL,
        )

    def test_update_mechanism(self, rng: jax.Array, physics_data: ksim.PhysicsData) -> None:
        """Test that the command update mechanism works correctly."""
        cmd = ksim.AngularVelocityCommand(scale=1.0)
        command = cmd.initial_command(physics_data, rng)

        next_command = cmd(command, physics_data, rng)
        assert jnp.array_equal(next_command, command)
