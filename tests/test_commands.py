"""Tests for command builders in the ksim package."""

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from ksim.commands import AngularVelocityCommand, Command, LinearVelocityCommand

_TOL = 1e-4


class DummyCommand(Command):
    def initial_command(self, rng: jax.Array) -> jnp.ndarray:
        return jnp.zeros((1,))

    def __call__(self, rng: jax.Array, time: Array) -> jnp.ndarray:
        return jnp.zeros((1,))


class TestLinearVelocityCommand:
    """Tests for the LinearVelocityCommand class."""

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def time(self) -> jnp.ndarray:
        """Return a time value."""
        return jnp.array(0.0)

    def test_command_shape(self, rng: jax.Array, time: jnp.ndarray) -> None:
        """Test that the command returns the correct shape."""
        cmd = LinearVelocityCommand()
        initial_command = cmd.initial_command(rng)
        result = cmd(initial_command, time, rng)
        chex.assert_shape(result, (2,))

    def test_command_bounds(self, rng: jax.Array, time: jnp.ndarray) -> None:
        """Test that the command values are within the expected bounds."""
        x_scale, y_scale = 2.0, 3.0
        cmd = LinearVelocityCommand(x_scale=x_scale, y_scale=y_scale)
        command = cmd.initial_command(rng)

        # Run multiple times to test bounds probabilistically
        for i in range(100):
            key = jax.random.fold_in(rng, i)
            command = cmd(command, time, key)

            assert command[0] >= -x_scale
            assert command[0] <= x_scale
            assert command[1] >= -y_scale
            assert command[1] <= y_scale

    def test_zero_probability(self, rng: jax.Array, time: jnp.ndarray) -> None:
        """Test that the command returns zeros when zero_prob is 1.0."""
        cmd = LinearVelocityCommand(zero_prob=1.0)
        command = cmd.initial_command(rng)
        result = cmd(command, time, rng)
        chex.assert_trees_all_close(
            result,
            jnp.zeros_like(result),
            atol=_TOL,
        )

    def test_update_mechanism(self, rng: jax.Array) -> None:
        """Test that the command update mechanism works correctly."""
        cmd = LinearVelocityCommand(switch_prob=1.0)
        command = cmd.initial_command(rng)
        time = jnp.array(0.0)

        next_command = cmd(command, time, rng)
        assert not jnp.array_equal(next_command, command)


class TestAngularVelocityCommand:
    """Tests for the AngularVelocityCommand class."""

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def time(self) -> jnp.ndarray:
        """Return a time value."""
        return jnp.array(0.0)

    def test_command_shape(self, rng: jax.Array, time: jnp.ndarray) -> None:
        """Test that the command returns the correct shape."""
        cmd = AngularVelocityCommand()
        initial_command = cmd.initial_command(rng)
        result = cmd(initial_command, time, rng)
        chex.assert_shape(result, (1,))

    def test_command_bounds(self, rng: jax.Array, time: jnp.ndarray) -> None:
        """Test that the command values are within the expected bounds."""
        scale = 2.0
        cmd = AngularVelocityCommand(scale=scale)

        # Run multiple times to test bounds probabilistically
        for i in range(100):
            key = jax.random.fold_in(rng, i)
            command = cmd.initial_command(key)
            result = cmd(command, time, key)

            assert result[0] >= -scale
            assert result[0] <= scale

    def test_zero_probability(self, rng: jax.Array, time: jnp.ndarray) -> None:
        """Test that the command returns zeros when zero_prob is 1.0."""
        cmd = AngularVelocityCommand(zero_prob=1.0)
        command = cmd.initial_command(rng)
        result = cmd(command, time, rng)
        chex.assert_trees_all_close(
            result,
            jnp.zeros_like(result),
            atol=_TOL,
        )

    def test_update_mechanism(self, rng: jax.Array) -> None:
        """Test that the command update mechanism works correctly."""
        cmd = AngularVelocityCommand(switch_prob=1.0)
        command = cmd.initial_command(rng)
        time = jnp.array(0.0)

        next_command = cmd(command, time, rng)
        assert not jnp.array_equal(next_command, command)
