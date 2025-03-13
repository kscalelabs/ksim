"""Test script to validate the export function."""

import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

from examples.kbot.standing import NUM_OUTPUTS, KBotCriticModel
from ksim.model.export import export_actor_model
from ksim.model.formulations import ActorCriticAgent, GaussianActionModel
from ksim.model.mlp import MLP


def create_zero_action_model() -> ActorCriticAgent:
    """Create a KBotZeroActions model for testing."""
    mlp = MLP(
        num_hidden_layers=2,
        hidden_features=512,
        out_features=NUM_OUTPUTS,
    )

    # Create a custom KBotZeroActions class that correctly returns zeros with the right shape
    class TestActorModel(GaussianActionModel):
        mlp: MLP
        num_outputs: int
        action_scale: float = 1.0  # Add a scale parameter to test action scaling

        def __call__(self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]) -> Array:
            # Process inputs to get the right shape
            lin_vel_cmd_2 = cmd["linear_velocity_command"]
            batch_size = lin_vel_cmd_2.shape[0]

            # For testing normalization, we'll directly return the normalized observation values
            # This will help us verify that normalization is working correctly
            joint_pos = obs["joint_position_observation"]
            joint_vel = obs["joint_velocity_observation"]

            # Print the first few values of each observation for debugging
            print(f"DEBUG - joint_pos first values: {joint_pos[0, :3]}")
            print(f"DEBUG - joint_vel first values: {joint_vel[0, :3]}")

            # Simply pass through the normalized observation values
            result = jnp.zeros((batch_size, self.num_outputs))
            # Add joint position values to the first half of the output
            result = result.at[:, :7].set(joint_pos[:, :7])
            # Add joint velocity values to the second half of the output
            result = result.at[:, 7:].set(joint_vel[:, :7])

            # Apply action scaling to test if it's preserved during export
            result = result * self.action_scale
            print(
                f"DEBUG - raw action output (before any potential normalization): {result[0, :3]}"
            )

            return result

        def get_raw_and_normalized_actions(
            self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array]
        ) -> tuple[Array, Array]:
            """Return both raw and potentially normalized actions for testing."""
            # Get raw actions (before any potential normalization)
            raw_actions = self(obs, cmd)

            # If there's any action normalization happening in the ActorCriticAgent,
            # it would be applied when calling the actor method
            # So we'll return both the raw actions and the actions from the actor method
            return raw_actions, raw_actions  # If no normalization, these should be identical

        def sample_and_log_prob(
            self, obs: FrozenDict[str, Array], cmd: FrozenDict[str, Array], rng: PRNGKeyArray
        ) -> tuple[Array, Array]:
            mean = self(obs, cmd)
            return mean, mean

    actor = TestActorModel(
        num_outputs=NUM_OUTPUTS,
        mlp=mlp,
        init_log_std=-0.7,
        action_scale=2.5,  # Use a non-trivial scale to test if it's preserved
    )

    critic = KBotCriticModel(
        mlp=MLP(
            num_hidden_layers=4,
            hidden_features=512,
            out_features=1,
        ),
    )

    return ActorCriticAgent(
        actor_module=actor,
        critic_module=critic,
    )


def initialize_model_variables(model: ActorCriticAgent) -> dict:
    """Initialize model variables with dummy data."""
    # Create dummy observations and commands
    dummy_obs = {
        "joint_position_observation_noisy": jnp.zeros((1, NUM_OUTPUTS)),
        "joint_velocity_observation_noisy": jnp.zeros((1, NUM_OUTPUTS)),
        "imu_acc_sensor_observation": jnp.zeros((1, 3)),
        "imu_gyro_sensor_observation": jnp.zeros((1, 3)),
        # Add clean observations too
        "joint_position_observation": jnp.zeros((1, NUM_OUTPUTS)),
        "joint_velocity_observation": jnp.zeros((1, NUM_OUTPUTS)),
        "base_orientation_observation": jnp.zeros((1, 3)),
        "base_linear_velocity_observation": jnp.zeros((1, 3)),
        "base_angular_velocity_observation": jnp.zeros((1, 3)),
    }
    dummy_cmd = {
        "linear_velocity_command": jnp.zeros((1, 2)),
        "angular_velocity_command": jnp.zeros((1, 1)),
    }

    # Initialize the model
    key = jax.random.PRNGKey(0)
    variables = model.init(key, FrozenDict(dummy_obs), FrozenDict(dummy_cmd))

    # Add non-trivial normalization parameters to test normalization
    for obs_name in dummy_obs:
        shape = dummy_obs[obs_name].shape[-1:]
        # Set means to 1.0 and stds to 2.0 for testing
        variables["normalization"][f"obs_mean_{obs_name}"] = jnp.ones(shape)
        variables["normalization"][f"obs_std_{obs_name}"] = jnp.ones(shape) * 2.0

    # Add action normalization parameters if they exist
    # Note: Based on the formulations.py code, there don't appear to be any action normalization parameters
    # But we'll check for them anyway
    if "action_mean" in variables.get("normalization", {}):
        print("Found action normalization parameters!")
        print(f"action_mean: {variables['normalization']['action_mean']}")
        print(f"action_std: {variables['normalization']['action_std']}")

    variables["normalization"]["returns_std"] = jnp.array(1.0)

    return variables


def test_export_and_inference():
    """Test exporting a model and running inference with it, verifying normalization."""
    print("Creating test actor model...")
    model = create_zero_action_model()

    print("Initializing model variables...")
    variables = initialize_model_variables(model)

    # Print the normalization parameters for debugging
    print("\nNormalization parameters:")
    print(
        f"joint_position_observation mean: {variables['normalization']['obs_mean_joint_position_observation']}"
    )
    print(
        f"joint_position_observation std: {variables['normalization']['obs_std_joint_position_observation']}"
    )
    print(
        f"joint_velocity_observation mean: {variables['normalization']['obs_mean_joint_velocity_observation']}"
    )
    print(
        f"joint_velocity_observation std: {variables['normalization']['obs_std_joint_velocity_observation']}"
    )

    # Create test inputs with different values to test normalization
    test_obs = {
        "joint_position_observation_noisy": jnp.ones((1, NUM_OUTPUTS)) * 30.0,
        "joint_velocity_observation_noisy": jnp.ones((1, NUM_OUTPUTS)) * 5.0,
        "imu_acc_sensor_observation": jnp.ones((1, 3)) * 3.0,
        "imu_gyro_sensor_observation": jnp.ones((1, 3)) * 3.0,
        "joint_position_observation": jnp.ones((1, NUM_OUTPUTS))
        * 30.0,  # Should normalize to (30-1)/2 = 14.5
        "joint_velocity_observation": jnp.ones((1, NUM_OUTPUTS))
        * 5.0,  # Should normalize to (5-1)/2 = 2.0
        "base_orientation_observation": jnp.ones((1, 3)) * 3.0,
        "base_linear_velocity_observation": jnp.ones((1, 3)) * 3.0,
        "base_angular_velocity_observation": jnp.ones((1, 3)) * 3.0,
    }

    # Also try a different set of values
    test_obs2 = {
        "joint_position_observation_noisy": jnp.ones((1, NUM_OUTPUTS)) * 11.0,
        "joint_velocity_observation_noisy": jnp.ones((1, NUM_OUTPUTS)) * 9.0,
        "imu_acc_sensor_observation": jnp.ones((1, 3)) * 3.0,
        "imu_gyro_sensor_observation": jnp.ones((1, 3)) * 3.0,
        "joint_position_observation": jnp.ones((1, NUM_OUTPUTS))
        * 11.0,  # Should normalize to (11-1)/2 = 5.0
        "joint_velocity_observation": jnp.ones((1, NUM_OUTPUTS))
        * 9.0,  # Should normalize to (9-1)/2 = 4.0
        "base_orientation_observation": jnp.ones((1, 3)) * 3.0,
        "base_linear_velocity_observation": jnp.ones((1, 3)) * 3.0,
        "base_angular_velocity_observation": jnp.ones((1, 3)) * 3.0,
    }

    test_cmd = {
        "linear_velocity_command": jnp.ones((1, 2)) * 2.0,
        "angular_velocity_command": jnp.ones((1, 1)) * 2.0,
    }

    # Test the JAX model directly with first set of inputs
    print("\nTesting JAX model with first set of inputs (joint_pos=30.0, joint_vel=5.0)...")
    jax_output = model.apply(variables, obs=test_obs, cmd=test_cmd, method="actor")
    print(f"JAX model output shape: {jax_output.shape}")
    print(f"JAX model output: {jax_output}")

    # Test with second set of inputs
    print("\nTesting JAX model with second set of inputs (joint_pos=11.0, joint_vel=9.0)...")
    jax_output2 = model.apply(variables, obs=test_obs2, cmd=test_cmd, method="actor")
    print(f"JAX model output shape: {jax_output2.shape}")
    print(f"JAX model output: {jax_output2}")

    # Create export directory
    export_dir = Path("test_export_zero_actor")
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    print(f"\nExporting model to {export_dir}...")
    export_actor_model(model, variables, str(export_dir))

    print("Loading exported model...")
    loaded_model = tf.saved_model.load(str(export_dir))

    print("\nRunning inference with exported model (first set of inputs)...")
    # Convert inputs to TensorFlow tensors
    tf_obs = {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in test_obs.items()}
    tf_cmd = {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v in test_cmd.items()}

    # Run inference
    result = loaded_model.signatures["serve_actor"](
        # Pass all observation fields as individual arguments
        joint_position_observation_noisy=tf_obs["joint_position_observation_noisy"],
        joint_velocity_observation_noisy=tf_obs["joint_velocity_observation_noisy"],
        imu_acc_sensor_observation=tf_obs["imu_acc_sensor_observation"],
        imu_gyro_sensor_observation=tf_obs["imu_gyro_sensor_observation"],
        joint_position_observation=tf_obs["joint_position_observation"],
        joint_velocity_observation=tf_obs["joint_velocity_observation"],
        base_orientation_observation=tf_obs["base_orientation_observation"],
        base_linear_velocity_observation=tf_obs["base_linear_velocity_observation"],
        base_angular_velocity_observation=tf_obs["base_angular_velocity_observation"],
        # Pass all command fields as individual arguments
        linear_velocity_command=tf_cmd["linear_velocity_command"],
        angular_velocity_command=tf_cmd["angular_velocity_command"],
    )

    # Extract the result
    tf_output = result["output_0"].numpy()
    print(f"TensorFlow model output shape: {tf_output.shape}")
    print(f"TensorFlow model output: {tf_output}")

    # Verify that the JAX and TensorFlow outputs are the same
    is_close = np.allclose(jax_output, tf_output, atol=1e-5)
    print(f"\nJAX and TensorFlow outputs match: {is_close}")

    # Expected values based on our normalization and action scaling:
    # For joint_position_observation: (30-1)/2 * 2.5 = 14.5 * 2.5 = 36.25
    # For joint_velocity_observation: (5-1)/2 * 2.5 = 2.0 * 2.5 = 5.0
    expected_first_half = np.ones((1, 7)) * 14.5 * 2.5  # Normalized joint positions with scaling
    expected_second_half = np.ones((1, 7)) * 2.0 * 2.5  # Normalized joint velocities with scaling
    expected_output = np.concatenate([expected_first_half, expected_second_half], axis=1)

    is_expected = np.allclose(tf_output, expected_output, atol=1e-5)
    print(f"Output matches expected normalized values with action scaling: {is_expected}")

    if is_close and is_expected:
        print(
            "✅ Test passed! The exported model correctly applies normalization and preserves action scaling."
        )
    else:
        print(
            "❌ Test failed! The exported model does not correctly handle normalization or action scaling."
        )
        if not is_close:
            print("JAX and TensorFlow outputs don't match.")
        if not is_expected:
            print("Output doesn't match expected normalized values with action scaling.")
            print(f"Expected: {expected_output}")
            print(f"Actual: {tf_output}")
            print(f"Difference: {np.abs(expected_output - tf_output)}")

    return is_close and is_expected


def test_action_normalization():
    """Test specifically for action normalization."""
    print("\n=== Testing specifically for action normalization ===")

    # Create the model
    print("Creating test actor model...")
    model = create_zero_action_model()

    print("Initializing model variables...")
    variables = initialize_model_variables(model)

    # Create test inputs
    test_obs = {
        "joint_position_observation_noisy": jnp.ones((1, NUM_OUTPUTS)) * 30.0,
        "joint_velocity_observation_noisy": jnp.ones((1, NUM_OUTPUTS)) * 5.0,
        "imu_acc_sensor_observation": jnp.ones((1, 3)) * 3.0,
        "imu_gyro_sensor_observation": jnp.ones((1, 3)) * 3.0,
        "joint_position_observation": jnp.ones((1, NUM_OUTPUTS)) * 30.0,
        "joint_velocity_observation": jnp.ones((1, NUM_OUTPUTS)) * 5.0,
        "base_orientation_observation": jnp.ones((1, 3)) * 3.0,
        "base_linear_velocity_observation": jnp.ones((1, 3)) * 3.0,
        "base_angular_velocity_observation": jnp.ones((1, 3)) * 3.0,
    }

    test_cmd = {
        "linear_velocity_command": jnp.ones((1, 2)) * 2.0,
        "angular_velocity_command": jnp.ones((1, 1)) * 2.0,
    }

    # First, get the normalized observations
    normalized_obs = model.apply(variables, obs=test_obs, method="normalize_obs")
    print(
        f"Normalized joint_position_observation: {normalized_obs['joint_position_observation'][0, :3]}"
    )
    print(
        f"Normalized joint_velocity_observation: {normalized_obs['joint_velocity_observation'][0, :3]}"
    )

    # Get actions from the actor method (which includes observation normalization)
    actor_method_actions = model.apply(variables, obs=test_obs, cmd=test_cmd, method="actor")
    print(f"Actions from actor method: {actor_method_actions[0, :3]}")

    # Get actions from the actor module directly (bypassing the actor method)
    # This requires accessing the actor module and applying it with the normalized observations
    actor_module = model.actor_module
    actor_params = jax.tree_util.tree_map(lambda x: x, variables["params"]["actor_module"])
    direct_actions = actor_module.apply({"params": actor_params}, normalized_obs, test_cmd)
    print(f"Actions directly from actor module with normalized obs: {direct_actions[0, :3]}")

    # Check if the actions match
    actions_match = np.allclose(actor_method_actions, direct_actions, atol=1e-5)
    print(f"Actions match: {actions_match}")

    if not actions_match:
        print("Difference detected! This suggests there might be action normalization happening.")
        print(f"Difference: {np.abs(actor_method_actions - direct_actions).max()}")
    else:
        print("No difference detected. This confirms there is no action normalization happening.")
        print("The difference seen in the main test is solely due to observation normalization.")

    return actions_match


if __name__ == "__main__":
    test_export_and_inference()
    test_action_normalization()
