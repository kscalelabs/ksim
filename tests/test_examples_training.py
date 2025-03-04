"""Tests for training examples in the ksim package."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest

from examples.default_humanoid.walking import HumanoidWalkingConfig, HumanoidWalkingTask


@pytest.mark.slow
def test_default_humanoid_training() -> None:
    """Test that training on default_humanoid works for a few iterations."""
    # Configure a minimal training run
    config = HumanoidWalkingConfig(
        num_envs=4,
        num_steps_per_trajectory=10,
        minibatch_size=20,
        num_learning_epochs=1,
        robot_model_name="examples/default_humanoid/",
    )

    # Create the task
    task = HumanoidWalkingTask(config)
    key = jax.random.PRNGKey(0)

    # Get the environment and model
    env = task.get_environment()
    model = task.get_model(key)

    # Initialize model parameters
    key, init_key = jax.random.split(key)
    dummy_state = env.get_dummy_env_state(init_key)
    params = model.init(init_key, dummy_state.obs, dummy_state.command)

    key, rollout_key = jax.random.split(key)
    states, _ = env.unroll_trajectories(
        model=model,
        params=params,
        rng=rollout_key,
        num_steps=config.num_steps_per_trajectory,
        num_envs=config.num_envs,
    )

    # Verify that we got valid outputs
    assert states is not None
    assert states.obs is not None
    assert states.reward is not None
    assert states.done is not None
    assert states.reward.shape == (
        config.num_steps_per_trajectory,
        config.num_envs,
    )  # Shape is (num_steps, num_envs)
    assert states.done.shape == (
        config.num_steps_per_trajectory,
        config.num_envs,
    )  # Shape is (num_steps, num_envs)

    # Test a single model update
    # Get optimizer
    optimizer = task.get_optimizer()
    opt_state = optimizer.init(params)

    # Get a trajectory dataset
    key, rollout_key = jax.random.split(key)
    trajectories_dataset = task.get_trajectory_dataset(model, params, env, rollout_key)

    # Get rollout time loss components
    rollout_time_loss_components = task.get_rollout_time_loss_components(
        model, params, trajectories_dataset
    )

    # Get a minibatch
    minibatch, minibatch_loss_components = task.get_minibatch(
        trajectories_dataset, rollout_time_loss_components, jnp.array(0)
    )

    # Update the model
    new_params, new_opt_state, loss_val, metrics = task.model_update(
        model,
        params,
        optimizer,
        opt_state,
        minibatch,
        minibatch_loss_components,
    )

    # Verify metrics
    assert metrics is not None
    assert "policy_objective" in metrics.keys() or "total_objective" in metrics.keys()
    assert "value_objective" in metrics.keys()
    assert "entropy_objective" in metrics.keys() or "average_advantage_norm" in metrics.keys()


@pytest.mark.slow
def test_default_humanoid_run_method() -> None:
    """Test that the run method of HumanoidWalkingTask works."""
    # Configure a minimal training run
    config = HumanoidWalkingConfig(
        num_envs=2,
        num_steps_per_trajectory=10,
        minibatch_size=20,
        num_learning_epochs=1,
        robot_model_name="examples/default_humanoid/",
        action="train",
    )

    # Create the task
    task = HumanoidWalkingTask(config)

    # Mock the train_iteration method to avoid actual training
    with patch.object(task, "rl_train_loop") as mock_train:
        task.run()
        assert mock_train.called
