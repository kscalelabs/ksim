"""Tests for the PPO loss."""

import chex
import jax.numpy as jnp

from ksim.task.ppo import compute_advantages_and_value_targets, compute_returns, get_deltas

_TOL = 1e-4

_GAMMA = 0.99
_GAE_LAMBDA = 0.95


def test_single_episode_zero_advantages_one_env() -> None:
    rewards = jnp.array([[0.5], [0.5], [0.5], [0.5]])
    dones = jnp.array([[False], [False], [False], [True]])
    expected_value_targets = jnp.array(
        [
            [1.9702],  # 0.5 + 0.99*1.48505 ≈ 1.9702
            [1.48505],  # 0.5 + 0.99*0.995 = 1.48505
            [0.995],  # 0.5 + 0.99*0.5 = 0.995
            [0.5],  # 0.5
        ]
    )
    # Use expected returns as dummy value estimates so that advantages should be zero.
    values = expected_value_targets
    advantages, value_targets = compute_advantages_and_value_targets(values, rewards, dones, _GAMMA, _GAE_LAMBDA)
    expected_advantages = jnp.zeros_like(expected_value_targets)
    chex.assert_trees_all_close(value_targets, expected_value_targets, atol=_TOL)
    chex.assert_trees_all_close(advantages, expected_advantages, atol=_TOL)


def test_two_episodes_zero_advantages_one_env() -> None:
    rewards = jnp.array([[0.5], [0.5], [0.5], [0.5], [0.5], [0.5], [0.5], [0.5]])
    dones = jnp.array([[False], [False], [False], [True], [False], [False], [False], [True]])
    expected_value_targets = jnp.array(
        [
            [1.9702],  # 0.5 + 0.99*1.48505 ≈ 1.9702
            [1.48505],  # 0.5 + 0.99*0.995 = 1.48505
            [0.995],  # 0.5 + 0.99*0.5 = 0.995
            [0.5],  # 0.5
            [1.9702],  # 0.5 + 0.99*1.48505 ≈ 1.9702
            [1.48505],  # 0.5 + 0.99*0.995 = 1.48505
            [0.995],  # 0.5 + 0.99*0.5 = 0.995
            [0.5],  # 0.5
        ]
    )
    values = expected_value_targets  # Setting dummy estimates equal to expected returns.
    advantages, value_targets = compute_advantages_and_value_targets(values, rewards, dones, _GAMMA, _GAE_LAMBDA)
    expected_advantages = jnp.zeros_like(expected_value_targets)
    chex.assert_trees_all_close(value_targets, expected_value_targets, atol=_TOL)
    chex.assert_trees_all_close(advantages, expected_advantages, atol=_TOL)


def test_single_episode_zero_advantages_one_env_no_terminal() -> None:
    rewards = jnp.array([[0.5], [0.5], [0.5], [0.5]])
    dones = jnp.array([[False], [False], [False], [False]])
    # For a non-terminal episode, if we set the value estimates to the fixed point of:
    # v = reward + gamma * v  =>  v = 0.5 + 0.99*v, then v = 50.
    expected_value_targets = jnp.array([[50.0], [50.0], [50.0], [50.0]])
    values = expected_value_targets
    advantages, value_targets = compute_advantages_and_value_targets(values, rewards, dones, _GAMMA, _GAE_LAMBDA)
    expected_advantages = jnp.zeros_like(expected_value_targets)
    chex.assert_trees_all_close(value_targets, expected_value_targets, atol=_TOL)
    chex.assert_trees_all_close(advantages, expected_advantages, atol=_TOL)


def test_advantage_and_value_targets_two_envs() -> None:
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
    advantages, value_targets = compute_advantages_and_value_targets(values, rewards, dones, _GAMMA, _GAE_LAMBDA)
    expected_advantages = jnp.zeros_like(expected_value_targets)
    chex.assert_trees_all_close(value_targets, expected_value_targets, atol=_TOL)
    chex.assert_trees_all_close(advantages, expected_advantages, atol=_TOL)


def test_get_deltas_two_envs() -> None:
    """Test the TD residual calculation function."""
    rewards = jnp.array(
        [
            [1.0, 0.5],
            [0.0, 2.0],
            [3.0, -1.0],
            [0.5, 1.5],
        ]
    )
    values = jnp.array(
        [
            [2.0, 1.0],
            [2.5, 3.0],
            [1.5, 2.0],
            [1.0, 0.5],
        ]
    )

    values_shifted = jnp.array(
        [
            [2.5, 3.0],  # Next values for timestep 0
            [1.5, 2.0],
            [1.0, 0.5],
            [0.0, 0.0],  # Dummy values for last timestep (will be masked)
        ]
    )

    termination_mask = jnp.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],  # Both environments terminate
        ]
    )

    expected_deltas = jnp.array(
        [
            [1.0 + _GAMMA * 2.5 * 1.0 - 2.0, 0.5 + _GAMMA * 3.0 * 1.0 - 1.0],
            [0.0 + _GAMMA * 1.5 * 1.0 - 2.5, 2.0 + _GAMMA * 2.0 * 1.0 - 3.0],
            [3.0 + _GAMMA * 1.0 * 1.0 - 1.5, -1.0 + _GAMMA * 0.5 * 1.0 - 2.0],
            [0.5 + _GAMMA * 0.0 * 0.0 - 1.0, 1.5 + _GAMMA * 0.0 * 0.0 - 0.5],
        ]
    )

    actual_deltas = get_deltas(rewards, values, values_shifted, termination_mask, _GAMMA)
    chex.assert_trees_all_close(actual_deltas, expected_deltas, atol=_TOL)


def test_compute_returns_two_episodes() -> None:
    """Test return calculation for a single episode that terminates."""
    rewards = jnp.array([[0.5], [0.5], [0.5], [0.5], [0.5], [0.5]])
    dones = jnp.array([[False], [False], [False], [True], [False], [True]])

    expected_returns = jnp.array(
        [
            [1.9701995],  # 0.5 + 0.99×1.48505
            [1.48505],  # 0.5 + 0.99×0.995
            [0.995],  # 0.5 + 0.99×0.5
            [0.5],  # 0.5
            [0.995],  # 0.5 + 0.99×0.5
            [0.5],  # 0.5
        ]
    )

    actual_returns = compute_returns(rewards, dones, _GAMMA)
    chex.assert_trees_all_close(actual_returns, expected_returns, atol=_TOL)
