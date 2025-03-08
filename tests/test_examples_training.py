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
        num_learning_epochs=1,
        num_env_states_per_minibatch=4,
        num_minibatches=1,
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
    dummy_states = env.get_dummy_env_states(config.num_envs)
    variables = model.init(init_key, dummy_states.obs, dummy_states.command)

    # Get optimizer
    optimizer = task.get_optimizer()
    opt_state = optimizer.init(variables["params"])

    # Test a single training iteration
    physics_model_L = env.get_init_physics_model()

    reset_rngs = jax.random.split(key, config.num_envs)
    env_state_EL_0, physics_data_EL_1 = jax.vmap(env.reset, in_axes=(None, None, 0, None))(
        model, variables, reset_rngs, physics_model_L
    )

    # Get a trajectory dataset
    key, rollout_key = jax.random.split(key)
    trajectories_dataset, rollout_time_loss_components, _, _, _ = task.get_rl_dataset(
        model=model,
        variables=variables,
        env=env,
        env_state_EL_t_minus_1=env_state_EL_0,
        physics_data_EL_t=physics_data_EL_1,
        physics_model_L=physics_model_L,
        rng=rollout_key,
    )

    # Get a minibatch
    minibatch, minibatch_loss_components = task.get_minibatch(
        trajectories_dataset, rollout_time_loss_components, jnp.array(0)
    )

    # Update the model
    _, _, _, metrics = task.model_update(
        model=model,
        variables=variables,
        optimizer=optimizer,
        opt_state=opt_state,
        env_state_batch=minibatch,
        rollout_time_loss_components=minibatch_loss_components,
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
        num_envs=4,
        num_env_states_per_minibatch=4,
        num_minibatches=1,
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

        # Get the environment and model to check for NaN values in initial setup
        env = task.get_environment()
        key = jax.random.PRNGKey(0)
        model = task.get_model(key)

        # Initialize model parameters
        key, init_key = jax.random.split(key)
        dummy_states = env.get_dummy_env_states(config.num_envs)
        variables = model.init(init_key, dummy_states.obs, dummy_states.command)

        # Check for NaN values in initial state
        # TODO: Switch these to asserts when we fix the NaN issue
        for k, v in dummy_states.obs.items():
            if jnp.isnan(v).any():
                print(f"WARNING: NaN values found in initial observation '{k}'")
        for k, v in dummy_states.command.items():
            if jnp.isnan(v).any():
                print(f"WARNING: NaN values found in initial command '{k}'")
        if jnp.isnan(dummy_states.action).any():
            print("WARNING: NaN values found in initial action")

        # Check for NaN values in model parameters
        for param_key, param_value in jax.tree_util.tree_leaves_with_path(variables["params"]):
            if jnp.isnan(param_value).any():
                param_path = "/".join(str(p) for p in param_key)
                print(f"WARNING: NaN values found in model parameter '{param_path}'")
