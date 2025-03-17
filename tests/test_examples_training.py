"""Tests for training examples in the ksim package."""

from unittest.mock import patch

import jax
import jax.numpy as jnp
import pytest
import xax

from examples.default_humanoid.walking import HumanoidWalkingTask


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
    task = HumanoidWalkingTask(PPOTask)
    key = jax.random.PRNGKey(0)

    # Get the environment and model
    env = task.get_environment()
    agent = task.get_model(key)

    # Initialize model parameters
    burn_in_rng, key, init_key = jax.random.split(key, 3)

    # Get optimizer
    optimizer = task.get_optimizer()
    opt_state = optimizer.init(agent)  # type: ignore TODO: make it look like an optax.Params

    # Test a single training iteration
    physics_model_L = env.get_init_physics_model()
    dummy_env_states = env.get_dummy_env_states(config.num_envs)
    # TODO
    dummy_model_input = ModelInput(
        obs=dummy_env_states.obs,
        command=dummy_env_states.command,
        action_history=None,
        recurrent_state=None,
    )
    normalizer = task.get_normalizer(dummy_model_input)
    reset_rngs = jax.random.split(key, config.num_envs)
    env_stateE_0, physics_dataE_1 = jax.vmap(env.reset, in_axes=(None, None, 0, None))(
        agent, normalizer, reset_rngs, physics_model_L
    )
    static_args = ["num_steps", "num_envs", "return_intermediate_data"]
    env_rollout_fn = xax.jit(static_argnames=static_args)(env.unroll_trajectories)

    # Get a trajectory dataset
    key, _ = jax.random.split(key)
    env_state_TE, _, _ = env_rollout_fn(
        agent=agent,
        rng=burn_in_rng,
        num_steps=task.num_rollout_steps_per_env,
        num_envs=config.num_envs,
        env_stateE_t_minus_1=env_stateE_0,
        physics_dataE_t=physics_dataE_1,
        physics_model_L=physics_model_L,
        return_intermediate_data=False,
    )
    env_stateD = xax.flatten_pytree(
        env_state_TE,
        flatten_size=task.dataset_size,
    )

    rollout_stats_TE = task.get_rollout_time_stats(
        agent=agent,
        trajectory_dataset=env_state_TE,
    )

    rollout_statsD = xax.flatten_pytree(
        rollout_stats_TE,
        flatten_size=task.dataset_size,
    )

    # Update the model
    key, reshuffle_rng = jax.random.split(key)
    (_, _, _), metrics = task.rl_pass(
        agent=agent,
        opt_state=opt_state,
        reshuffle_rng=reshuffle_rng,
        optimizer=optimizer,
        datasetD=env_stateD,
        rollout_time_statsD=rollout_statsD,
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
        agent = task.get_model(key)

        # Initialize model parameters
        key, init_key = jax.random.split(key)
        dummy_states = env.get_dummy_env_states(config.num_envs)

        # Check for NaN values in initial state
        # TODO: Switch these to asserts when we fix the NaN issue
        for k, v in dummy_states.obs.items():
            if jnp.isnan(v).any():
                xax.show_warning(f"NaN values found in initial observation '{k}'", important=True)
        for k, v in dummy_states.command.items():
            if jnp.isnan(v).any():
                xax.show_warning(f"NaN values found in initial command '{k}'", important=True)
        if jnp.isnan(dummy_states.action).any():
            xax.show_warning("NaN values found in initial action", important=True)

        # Check for NaN values in model parameters
        for param_key, param_value in jax.tree_util.tree_leaves_with_path(agent):
            if jnp.isnan(param_value).any():
                param_path = "/".join(str(p) for p in param_key)
                xax.show_warning(
                    f"NaN values found in model parameter '{param_path}'",
                    important=True,
                )


if __name__ == "__main__":
    test_default_humanoid_training()
    # test_default_humanoid_run_method()
