"""Tests for the PPO loss."""

import unittest

import chex
import jax.numpy as jnp

from ksim.losses.ppo import PPOLoss

_TOL = 1e-4


class TestPppoBasic(chex.TestCase):
    def setUp(self) -> None:
        self.loss = PPOLoss(
            clip_param=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.001,
            clip_value_loss=True,
            normalize_advantage=True,
            gamma=0.99,
            lam=0.95,
            eps=1e-6,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_episode_zero_advantages_one_env(self) -> None:
        rewards = jnp.array([[0.5], [0.5], [0.5], [0.5]])
        dones = jnp.array([[False], [False], [False], [True]])
        # Manually computed discounted returns:
        # R3 = 0.5, R2 = 0.5 + 0.99*0.5 = 0.995, R1 = 0.5 + 0.99*0.995 ≈ 1.48505,
        # R0 = 0.5 + 0.99*1.48505 ≈ 1.9702.
        expected_value_targets = jnp.array([[1.9702], [1.48505], [0.995], [0.5]])
        # Use expected returns as dummy value estimates so that advantages should be zero.
        values = expected_value_targets
        advantages, value_targets = self.variant(self.loss.compute_advantages_and_value_targets)(values, rewards, dones)
        expected_advantages = jnp.zeros_like(expected_value_targets)
        chex.assert_trees_all_close(value_targets, expected_value_targets, atol=_TOL)
        chex.assert_trees_all_close(advantages, expected_advantages, atol=_TOL)

    @chex.variants(with_jit=True, without_jit=True)
    def test_two_episodes_zero_advantages_one_env(self) -> None:
        rewards = jnp.array([[0.5], [0.5], [0.5], [0.5], [0.5], [0.5], [0.5], [0.5]])
        dones = jnp.array([[False], [False], [False], [True], [False], [False], [False], [True]])
        # For each episode of 4 steps, the discounted returns are:
        # R3 = 0.5, R2 = 0.5 + 0.99*0.5 = 0.995, R1 = 0.5 + 0.99*0.995 ≈ 1.48505,
        # R0 = 0.5 + 0.99*1.48505 ≈ 1.9702.
        expected_value_targets = jnp.array(
            [
                [1.9702],
                [1.48505],
                [0.995],
                [0.5],
                [1.9702],
                [1.48505],
                [0.995],
                [0.5],
            ]
        )
        values = expected_value_targets  # Setting dummy estimates equal to expected returns.
        advantages, value_targets = self.variant(self.loss.compute_advantages_and_value_targets)(values, rewards, dones)
        expected_advantages = jnp.zeros_like(expected_value_targets)
        chex.assert_trees_all_close(value_targets, expected_value_targets, atol=_TOL)
        chex.assert_trees_all_close(advantages, expected_advantages, atol=_TOL)

    @chex.variants(with_jit=True, without_jit=True)
    def test_single_episode_zero_advantages_one_env_no_terminal(self) -> None:
        rewards = jnp.array([[0.5], [0.5], [0.5], [0.5]])
        dones = jnp.array([[False], [False], [False], [False]])
        # For a non-terminal episode, if we set the value estimates to the fixed point of:
        # v = reward + gamma * v  =>  v = 0.5 + 0.99*v, then v = 50.
        expected_value_targets = jnp.array([[50.0], [50.0], [50.0], [50.0]])
        values = expected_value_targets
        advantages, value_targets = self.variant(self.loss.compute_advantages_and_value_targets)(values, rewards, dones)
        expected_advantages = jnp.zeros_like(expected_value_targets)
        chex.assert_trees_all_close(value_targets, expected_value_targets, atol=_TOL)
        chex.assert_trees_all_close(advantages, expected_advantages, atol=_TOL)

    @chex.variants(with_jit=True, without_jit=True)
    def test_advantage_and_value_targets_two_envs(self) -> None:
        # Define two episodes, each with 4 timesteps.
        rewards = jnp.array(
            [
                [0.5, 1.0],
                [0.5, 1.0],
                [0.5, 1.0],
                [0.5, 1.0],
            ]
        )
        dones = jnp.array(
            [
                [False, False],
                [False, False],
                [False, False],
                [True, True],
            ]
        )
        expected_value_targets = jnp.array(
            [
                [1.9702, 3.9404],
                [1.48505, 2.9701],
                [0.995, 1.99],
                [0.5, 1.0],
            ]
        )
        # Set value estimates equal to expected returns so advantages become zero.
        values = expected_value_targets
        advantages, value_targets = self.variant(self.loss.compute_advantages_and_value_targets)(values, rewards, dones)
        expected_advantages = jnp.zeros_like(expected_value_targets)
        chex.assert_trees_all_close(value_targets, expected_value_targets, atol=_TOL)
        chex.assert_trees_all_close(advantages, expected_advantages, atol=_TOL)


if __name__ == "__main__":
    unittest.main()
