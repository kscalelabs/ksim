"""Tests for command builders in the ksim package."""

import chex
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

from ksim.builders.commands import (
    AngularVelocityCommand,
    Command,
    LinearVelocityCommand,
)

_TOL = 1e-4


class DummyCommand(Command):
    def __call__(self, rng: jax.Array, time: Array) -> jnp.ndarray:
        return jnp.zeros((1,))


def test_command_name() -> None:
    """Test that command names are correctly generated."""
    cmd = DummyCommand()
    assert cmd.get_name() == "dummy_command"
    assert cmd.command_name == "dummy_command"


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

    def test_initialization(self) -> None:
        """Test that the command is initialized with the correct parameters."""
        cmd = LinearVelocityCommand(
            x_scale=1.0,
            y_scale=1.0,
            switch_prob=0.0,
            zero_prob=0.0,
        )
        assert cmd.x_scale == 1.0
        assert cmd.y_scale == 1.0
        assert cmd.switch_prob == 0.0
        assert cmd.zero_prob == 0.0

    def test_command_shape(self, rng: jax.Array, time: jnp.ndarray) -> None:
        """Test that the command returns the correct shape."""
        cmd = LinearVelocityCommand()
        result = cmd(rng, time)
        chex.assert_shape(result, (2,))

    def test_command_bounds(self, rng: jax.Array, time: jnp.ndarray) -> None:
        """Test that the command values are within the expected bounds."""
        x_scale, y_scale = 2.0, 3.0
        cmd = LinearVelocityCommand(x_scale=x_scale, y_scale=y_scale)

        # Run multiple times to test bounds probabilistically
        for i in range(100):
            key = jax.random.fold_in(rng, i)
            result = cmd(key, time)

            assert result[0] >= -x_scale
            assert result[0] <= x_scale
            assert result[1] >= -y_scale
            assert result[1] <= y_scale

    def test_zero_probability(self, rng: jax.Array, time: jnp.ndarray) -> None:
        """Test that the command returns zeros when zero_prob is 1.0."""
        cmd = LinearVelocityCommand(zero_prob=1.0)
        result = cmd(rng, time)
        chex.assert_trees_all_close(
            result,
            jnp.zeros_like(result),
            atol=_TOL,
        )

    def test_update_mechanism(self, rng: jax.Array) -> None:
        """Test that the command update mechanism works correctly."""
        cmd = LinearVelocityCommand(switch_prob=1.0)
        prev_command = jnp.array([1.0, 1.0])
        time = jnp.array(0.0)

        result = cmd.update(prev_command, rng, time)
        assert not jnp.array_equal(result, prev_command)


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

    def test_initialization(self) -> None:
        """Test that the command is initialized with the correct parameters."""
        cmd = AngularVelocityCommand(scale=1.0, switch_prob=0.0, zero_prob=0.0)
        assert cmd.scale == 1.0
        assert cmd.switch_prob == 0.0
        assert cmd.zero_prob == 0.0

    def test_command_shape(self, rng: jax.Array, time: jnp.ndarray) -> None:
        """Test that the command returns the correct shape."""
        cmd = AngularVelocityCommand()
        result = cmd(rng, time)
        chex.assert_shape(result, (1,))

    def test_command_bounds(self, rng: jax.Array, time: jnp.ndarray) -> None:
        """Test that the command values are within the expected bounds."""
        scale = 2.0
        cmd = AngularVelocityCommand(scale=scale)

        # Run multiple times to test bounds probabilistically
        for i in range(100):
            key = jax.random.fold_in(rng, i)
            result = cmd(key, time)

            assert result[0] >= -scale
            assert result[0] <= scale

    def test_zero_probability(self, rng: jax.Array, time: jnp.ndarray) -> None:
        """Test that the command returns zeros when zero_prob is 1.0."""
        cmd = AngularVelocityCommand(zero_prob=1.0)
        result = cmd(rng, time)
        chex.assert_trees_all_close(
            result,
            jnp.zeros_like(result),
            atol=_TOL,
        )

    def test_update_mechanism(self, rng: jax.Array) -> None:
        """Test that the command update mechanism works correctly."""
        cmd = AngularVelocityCommand(switch_prob=1.0)
        prev_command = jnp.array([1.0])
        time = jnp.array(0.0)

        result = cmd.update(prev_command, rng, time)
        assert not jnp.array_equal(result, prev_command)
