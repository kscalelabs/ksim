import unittest

import chex
import jax
import jax.numpy as jnp

from ksim.builders.commands import (
    AngularVelocityCommand,
    Command,
    LinearVelocityCommand,
)

_TOL = 1e-4


class DummyCommand(Command):
    def __call__(self, rng):
        return jnp.zeros((1,))


class BaseCommandTest(chex.TestCase):
    def test_command_name(self):
        cmd = DummyCommand()
        self.assertEqual(cmd.get_name(), "dummy_command")
        self.assertEqual(cmd.command_name, "dummy_command")


class LinearVelocityCommandTest(chex.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)

    def test_initialization(self):
        cmd = LinearVelocityCommand(
            x_scale=1.0,
            y_scale=1.0,
            switch_prob=0.0,
            zero_prob=0.0,
        )
        self.assertEqual(cmd.x_scale, 1.0)
        self.assertEqual(cmd.y_scale, 1.0)
        self.assertEqual(cmd.switch_prob, 0.0)
        self.assertEqual(cmd.zero_prob, 0.0)

    def test_command_shape(self):
        cmd = LinearVelocityCommand()
        result = cmd(self.rng)
        chex.assert_shape(result, (2,))

    def test_command_bounds(self):
        x_scale, y_scale = 2.0, 3.0
        cmd = LinearVelocityCommand(x_scale=x_scale, y_scale=y_scale)

        # Run multiple times to test bounds probabilistically
        for _ in range(100):
            self.rng, key = jax.random.split(self.rng)
            result = cmd(key)

            self.assertTrue(result[0] >= -x_scale)
            self.assertTrue(result[0] <= x_scale)
            self.assertTrue(result[1] >= -y_scale)
            self.assertTrue(result[1] <= y_scale)

    def test_zero_probability(self):
        cmd = LinearVelocityCommand(zero_prob=1.0)
        result = cmd(self.rng)
        chex.assert_trees_all_close(
            result,
            jnp.zeros_like(result),
            atol=_TOL,
        )

    def test_update_mechanism(self):
        cmd = LinearVelocityCommand(switch_prob=1.0)
        prev_command = jnp.array([1.0, 1.0])
        time = jnp.array(0.0)

        result = cmd.update(prev_command, self.rng, time)
        self.assertFalse(jnp.array_equal(result, prev_command))


class AngularVelocityCommandTest(chex.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)

    def test_initialization(self):
        cmd = AngularVelocityCommand(scale=1.0, switch_prob=0.0, zero_prob=0.0)
        self.assertEqual(cmd.scale, 1.0)
        self.assertEqual(cmd.switch_prob, 0.0)
        self.assertEqual(cmd.zero_prob, 0.0)

    def test_command_shape(self):
        cmd = AngularVelocityCommand()
        result = cmd(self.rng)
        chex.assert_shape(result, (1,))

    def test_command_bounds(self):
        scale = 2.0
        cmd = AngularVelocityCommand(scale=scale)

        # Run multiple times to test bounds probabilistically
        for _ in range(100):
            self.rng, key = jax.random.split(self.rng)
            result = cmd(key)

            self.assertTrue(result[0] >= -scale)
            self.assertTrue(result[0] <= scale)

    def test_zero_probability(self):
        cmd = AngularVelocityCommand(zero_prob=1.0)
        result = cmd(self.rng)
        chex.assert_trees_all_close(
            result,
            jnp.zeros_like(result),
            atol=_TOL,
        )

    def test_update_mechanism(self):
        cmd = AngularVelocityCommand(switch_prob=1.0)
        prev_command = jnp.array([1.0])
        time = jnp.array(0.0)

        result = cmd.update(prev_command, self.rng, time)
        self.assertFalse(jnp.array_equal(result, prev_command))


if __name__ == "__main__":
    unittest.main()
