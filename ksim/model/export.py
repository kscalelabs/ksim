from typing import Any, Callable, Protocol

import jax
import numpy as np
import tensorflow as tf
from jaxtyping import Array, PyTree
from orbax.export import ExportManager, JaxModule, ServingConfig
from orbax.export.constants import ExportModelType

from ksim.model.formulations import ActorCriticAgent


class ActorJaxModule(JaxModule):
    """JaxModule wrapper for ActorCriticAgent's actor component to enable export."""

    def __init__(self, model: ActorCriticAgent, variables: dict[str, Any]) -> None:
        """Initialize the JaxModule wrapper.

        Args:
            model: The ActorCriticAgent model to wrap
            variables: The model variables including normalization parameters
        """

        # Define a function that takes a single PyTree input (packed obs and cmd)
        def actor_fn(params: PyTree, inputs: tuple[dict[str, Array], dict[str, Array]]) -> Any:
            obs, cmd = inputs
            # Apply the actor model and convert result to numpy
            # Note: This uses the actor method which handles normalization internally
            return model.apply(params, obs=obs, cmd=cmd, method="actor")

        # Initialize the JaxModule with our actor function
        super().__init__(params=variables, apply_fn=actor_fn)

        # Store model for reference
        self._model = model
        self._variables = variables

    @property
    def methods(self) -> dict[str, Callable[..., Any]]:
        """Return the methods to be exported.

        Returns:
            Dictionary mapping method names to callables
        """
        # Access the methods property from the parent class
        parent_methods = super().methods
        return parent_methods

    def export_version(self) -> ExportModelType:
        """Return the export version.

        Returns:
            The export version constant
        """
        return ExportModelType.TF_SAVEDMODEL


def create_serving_config(model: ActorCriticAgent, variables: dict[str, Any]) -> ServingConfig:
    """Create a serving config for the model.

    Args:
        model: The ActorCriticAgent model
        variables: The model variables

    Returns:
        A ServingConfig for the model
    """
    # Get a dummy observation and command to determine input shapes
    dummy_obs: dict[str, np.ndarray] = {}
    dummy_cmd: dict[str, np.ndarray] = {}

    # Extract normalization parameters to determine expected input shapes
    for key in variables["normalization"]:
        if key.startswith("obs_mean_"):
            obs_name = key[len("obs_mean_") :]
            shape = variables["normalization"][key].shape
            # Create dummy tensor with appropriate shape
            dummy_obs[obs_name] = np.zeros((1,) + shape)

    # Add dummy commands - this is a simplification, you may need to adjust based on your model
    dummy_cmd["linear_velocity_command"] = np.zeros((1, 2))
    dummy_cmd["angular_velocity_command"] = np.zeros((1, 1))

    # Create TensorSpec for inputs
    obs_signature = {k: tf.TensorSpec(v.shape, tf.float32, name=k) for k, v in dummy_obs.items()}
    cmd_signature = {k: tf.TensorSpec(v.shape, tf.float32, name=k) for k, v in dummy_cmd.items()}

    # Define a preprocessor function to pack obs and cmd into a single input
    def preprocessor(obs: dict[str, tf.Tensor], cmd: dict[str, tf.Tensor]) -> Any:
        # Convert TensorFlow tensors to NumPy arrays
        obs_dict = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in obs.items()}
        cmd_dict = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in cmd.items()}

        # Pack into a tuple for the model function
        return (obs_dict, cmd_dict)

    # Updated parameters based on the correct signature:
    # (self, signature_key: Union[str, Sequence[str]], input_signature: Optional[Sequence[jaxtyping.PyTree]] = None,
    # tf_preprocessor: Optional[Callable[..., Any]] = None, tf_postprocessor: Optional[Callable[..., Any]] = None,
    # extra_trackable_resources: Any = None, method_key: Optional[str] = None)
    return ServingConfig(
        signature_key="serve_actor",
        # The input_signature should be a list of individual arguments, not a list containing a dictionary
        # Each argument should be a separate entry in the input_signature list
        input_signature=[obs_signature, cmd_signature],
        tf_preprocessor=preprocessor,
        # The method_key should match a key in the methods dictionary returned by the JaxModule
        # Based on the error message, the available method key is 'jax_module_default_method'
        method_key="jax_module_default_method",
    )


def export_actor_model(model: ActorCriticAgent, variables: dict[str, Any], export_dir: str) -> None:
    """Export only the actor component of a trained ActorCriticAgent model to SavedModel format.

    This function exports only the actor part of the model, but preserves the normalization
    functionality which is crucial for correct inference.

    Args:
        model: The ActorCriticAgent model to export
        variables: The model variables including normalization parameters
        export_dir: Directory to save the exported model
    """
    jax_module = ActorJaxModule(model, variables)

    # Create serving config
    serving_config = create_serving_config(model, variables)

    # Create export manager
    export_manager = ExportManager(module=jax_module, serving_configs=[serving_config])

    export_manager.save(export_dir)
    print(f"Actor model exported to {export_dir}")


# Define a Protocol for the task interface
class TaskWithModelAndCheckpoint(Protocol):
    def load_checkpoint(self, checkpoint_path: str, part: str) -> dict[str, Any]: ...
    def get_model(self, key: jax.Array) -> ActorCriticAgent: ...


def export_actor_from_task(
    task: TaskWithModelAndCheckpoint, checkpoint_path: str, export_dir: str
) -> None:
    """Export only the actor component from a task's checkpoint.

    This is a helper function to export just the actor model from a task.
    It loads the checkpoint, extracts the model and variables, and exports only the actor.

    Args:
        task: The task instance that contains the model
        checkpoint_path: Path to the checkpoint to load
        export_dir: Directory to save the exported model
    """
    variables = task.load_checkpoint(checkpoint_path, part="model")
    model = task.get_model(jax.random.PRNGKey(0))
    export_actor_model(model, variables, export_dir)

    print(f"Actor model exported from checkpoint {checkpoint_path} to {export_dir}")


# Example usage:
"""
# In your training script, after training is complete:
from ksim.model.export import export_actor_model, export_actor_from_task

# Option 1: Export directly from variables
export_actor_model(model, variables, "exported_models/my_actor_model")

# Option 2: Export from a task with a checkpoint
export_actor_from_task(
    task,
    checkpoint_path="checkpoints/checkpoint_1000",
    export_dir="exported_models/my_actor_model"
)

# Later, to load and use the exported model in TensorFlow:
import tensorflow as tf
import numpy as np

# Load the model
loaded_model = tf.saved_model.load("exported_models/my_actor_model")

# Create input data - must match the expected observation and command structure
obs = {
    "joint_position_observation_noisy": np.zeros((1, 14)),
    "joint_velocity_observation_noisy": np.zeros((1, 14)),
    "imu_acc_sensor_observation": np.zeros((1, 3)),
    "imu_gyro_sensor_observation": np.zeros((1, 3))
}
cmd = {
    "linear_velocity_command": np.zeros((1, 2)),
    "angular_velocity_command": np.zeros((1, 1))
}

# Run inference - this will automatically apply normalization
actions = loaded_model.signatures["serve_actor"](obs=obs, cmd=cmd)
"""
