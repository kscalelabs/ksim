"""PPO algorithm tests."""

import chex
import unittest
from ksim.task.ppo import PPOBatch, PPOTask
import jax.numpy as jnp
import jax
import xax
from dataclasses import dataclass
from ksim.task.rl import RLConfig

_TOL = 1e-4


@jax.tree_util.register_dataclass
@dataclass
class DummyConfig(RLConfig):
    gamma: float = xax.field(value=0.99)
    lam: float = xax.field(value=0.95)

    max_trajectory_seconds: float = xax.field(value=1.0)


class DummyPPOTask(PPOTask):
    def get_environment(self): ...

    def get_model(self): ...

    def get_model_obs_from_state(self, state): ...

    def viz_environment(self): ...


class ComputeAdvantagesTest(chex.TestCase):
    def setUp(self):
        self.task = DummyPPOTask(config=DummyConfig())

    def test_simple(self):
        batch = PPOBatch(
            observations=None,
            next_observations=None,
            actions=None,
            rewards=jnp.array([[1.0, 1.0], [1.0, 1.0]]),
            done=jnp.array([[0, 0], [1, 1]]),
            action_log_probs=None,
        )

        values = jnp.array([[1.0, 1.0], [0.0, 0.0]])

        expected_advantages = jnp.array([[0.9405, 0.9405], [1.0, 1.0]])
        computed_advantages = self.task._compute_advantages(values, batch)
        chex.assert_trees_all_close(
            computed_advantages,
            expected_advantages,
            atol=_TOL,
        )

    def test_one_episode_one_env(self):
        batch = PPOBatch(
            observations=None,
            next_observations=None,
            actions=None,
            rewards=jnp.array([[1.0], [1.0], [0.0]]),
            done=jnp.array([[0], [0], [1]]),
            action_log_probs=None,
        )

        values = jnp.array([[1.0], [1.0], [0.0]])
        expected_advantages = jnp.array([[0.99], [0.0], [0.0]])
        computed_advantages = self.task._compute_advantages(values, batch)
        chex.assert_trees_all_close(
            computed_advantages,
            expected_advantages,
            atol=_TOL,
        )

    def test_two_episodes_two_envs(self):
        batch = PPOBatch(
            observations=None,
            next_observations=None,
            actions=None,
            rewards=jnp.array([[1.0, 1.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
            done=jnp.array([[0, 0], [1, 1], [0, 0], [1, 1]]),
            action_log_probs=None,
        )

        values = jnp.array([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])

        expected_advantages = jnp.array(
            [
                [0.0, 0.9405],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )

        computed_advantages = self.task._compute_advantages(values, batch)
        chex.assert_trees_all_close(
            computed_advantages,
            expected_advantages,
            atol=_TOL,
        )


if __name__ == "__main__":
    unittest.main()
