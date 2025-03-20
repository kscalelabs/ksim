"""Base Types for Environments."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from flax.core import FrozenDict
from jaxtyping import Array, PyTree
from mujoco import mjx

PhysicsData = mjx.Data | mujoco.MjData
PhysicsModel = mjx.Model | mujoco.MjModel


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PhysicsState:
    """Everything you need for the engine to take an action and step physics."""

    most_recent_action: Array
    data: PhysicsData


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Trajectory:
    qpos: Array
    qvel: Array
    obs: FrozenDict[str, Array]
    command: FrozenDict[str, Array]
    action: Array
    done: Array
    timestep: Array
    termination_components: FrozenDict[str, Array]
    aux_outputs: PyTree | None


def chunk_trajectory(trajectory: Trajectory) -> list[Trajectory]:
    """Chunks a non-implicit trajectory PyTree using `done`.

    After collecting a trajectory of trajectories, we end up with trajectories
    whose `done` has shape `(num_batches, num_steps)`. This function splits
    these trajectories into a list of trajectories where each trajectory is
    terminated by either a `done` or the end of the trajectory.

    Args:
        trajectories: A non-implicit trajectories PyTree.

    Returns:
        A list of trajectories, where each trajectory represents a complete
        episode or a segment ending with the trajectory's end. Note that the
        batch dimension is preserved in each trajectory.
    """
    # Get the done flags from trajectories
    done_flags = trajectory.done
    num_batches, num_steps = done_flags.shape

    # Initialize list to store chunked trajectories
    chunked_trajectories: list[Trajectory] = []

    # Process each batch
    for batch_idx in range(num_batches):
        # Find indices where episodes end (done=True)
        episode_ends = jnp.where(done_flags[batch_idx])[0]

        # Add the last step index if it's not already included
        if len(episode_ends) == 0 or episode_ends[-1] != num_steps - 1:
            episode_ends = jnp.append(episode_ends, num_steps - 1)

        # Initialize the start index for slicing
        start_idx = 0

        # Split the trajectories based on episode ends
        for end_idx in episode_ends:
            # Create a slice for this episode
            batch_slice = slice(batch_idx, batch_idx + 1)
            step_slice = slice(start_idx, end_idx + 1)

            # Extract the chunk using tree_map to handle the PyTree structure
            chunk = jax.tree_map(lambda x: x[batch_slice, step_slice] if x is not None else None, trajectory)

            chunked_trajectories.append(chunk)

            # Update start index for next chunk
            start_idx = end_idx + 1

            # Break if we've reached the end of the trajectory
            if end_idx == num_steps - 1:
                break

    return chunked_trajectories


def concatenate_trajectories(trajectories: list[Trajectory]) -> Trajectory:
    length = max(trajectory.done.shape[1] for trajectory in trajectories)

    def pad_array(x: Any, target_length: int) -> Any:  # noqa: ANN401
        if x is None:
            return None
        if not isinstance(x, jnp.ndarray):
            return x
        ndim = x.ndim
        pad_width = [(0, 0)] * ndim
        if ndim > 1:
            pad_width[1] = (0, target_length - x.shape[1])
        return jnp.pad(x, pad_width, mode="constant")

    def update_done_flag(trajectory: Trajectory) -> Trajectory:
        new_done = jnp.cumsum(trajectory.done, axis=-1) > 0

        # Need to do it this way because we trajectories are frozen.
        kwargs = trajectory.__dict__
        kwargs["done"] = new_done
        return Trajectory(**kwargs)

    padded_trajectories: list[Trajectory] = jax.tree_map(lambda x: pad_array(x, length), trajectories)
    padded_trajectories = [update_done_flag(trajectory) for trajectory in padded_trajectories]
    return jax.tree_map(lambda *x: jnp.concatenate(x, axis=0), *padded_trajectories)


def generate_trajectory_batches(
    trajectories: Trajectory,
    batch_size: int,
    min_batch_size: int = 2,
    min_trajectory_length: int = 3,
    group_by_length: bool = False,
    include_last_batch: bool = True,
) -> list[Trajectory]:
    """Generates batches of trajectories.

    Args:
        trajectories: The collected trajectories to batch.
        batch_size: The size of the batches to generate.
        min_batch_size: The minimum number of trajectories to include in a batch.
        min_trajectory_length: The minimum number of trajectories in a
            trajectory for it to be included in the batch.
        group_by_length: Whether to group trajectories by length, otherwise,
            trajectories are grouped randomly.
        include_last_batch: Whether to include the last batch if it's not full.

    Returns:
        An iterator over the batches of trajectories.
    """
    trajectory_list = chunk_trajectory(trajectories)

    # Remove trajectories that are shorter than `min_trajectory_length`.
    trajectory_list = [t for t in trajectory_list if t.done.shape[-1] >= min_trajectory_length]

    if group_by_length:
        # Sort trajectories so that adjacent trajectories have similar lengths.
        trajectory_list.sort(key=lambda x: x.done.shape[-1])

    # Group trajectories by batch size
    batches: list[list[Trajectory]] = [
        trajectory_list[i : i + batch_size] for i in range(0, len(trajectory_list), batch_size)
    ]

    # Remove batches that are smaller than `min_batch_size`.
    batches = [batch for batch in batches if len(batch) >= min_batch_size]

    # Remove the last batch if it's not full and `include_last_batch` is False.
    if not include_last_batch and len(batches[-1]) < batch_size:
        batches = batches[:-1]

    return [concatenate_trajectories(batch) for batch in batches]
