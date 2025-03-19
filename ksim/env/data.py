"""Base Types for Environments."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax.core import FrozenDict
from jaxtyping import Array
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
class Transition:
    qpos: Array
    qvel: Array
    obs: FrozenDict[str, Array]
    command: FrozenDict[str, Array]
    action: Array
    reward: Array
    done: Array
    timestep: Array

    termination_components: FrozenDict[str, Array]
    reward_components: FrozenDict[str, Array]


def chunk_transitions(transitions: Transition) -> list[Transition]:
    """Chunks a non-implicit transitions PyTree using `done`.

    After collecting a trajectory of transitions, we end up with transitions
    whose `done` has shape `(num_batches, num_steps)`. This function splits
    these transitions into a list of transitions where each transition is
    terminated by either a `done` or the end of the trajectory.

    Args:
        transitions: A non-implicit transitions PyTree.

    Returns:
        A list of transitions, where each transition represents a complete episode
        or a segment ending with the trajectory's end.
    """
    # Get the done flags from transitions
    done_flags = transitions.done
    num_batches, num_steps = done_flags.shape

    # Initialize list to store chunked transitions
    chunked_transitions: list[Transition] = []

    # Process each batch
    for batch_idx in range(num_batches):
        # Find indices where episodes end (done=True)
        episode_ends = np.where(done_flags[batch_idx])[0]

        # Add the last step index if it's not already included
        if len(episode_ends) == 0 or episode_ends[-1] != num_steps - 1:
            episode_ends = np.append(episode_ends, num_steps - 1)

        # Initialize the start index for slicing
        start_idx = 0

        # Split the transitions based on episode ends
        for end_idx in episode_ends:
            # Create a slice for this episode
            batch_slice = slice(batch_idx, batch_idx + 1)
            step_slice = slice(start_idx, end_idx + 1)

            # Extract the chunk using tree_map to handle the PyTree structure
            chunk = jax.tree_map(lambda x: x[batch_slice, step_slice] if x is not None else None, transitions)

            chunked_transitions.append(chunk)

            # Update start index for next chunk
            start_idx = end_idx + 1

            # Break if we've reached the end of the trajectory
            if end_idx == num_steps - 1:
                break

    return chunked_transitions


def concatenate_transitions(transitions: list[Transition]) -> Transition:
    length = max(transition.done.shape[1] for transition in transitions)
    padded_transitions = [
        jax.tree_map(lambda x: jnp.pad(x, ((0, 0), (0, length - x.shape[1])), mode="constant"), transitions)
    ]
    return jax.tree_map(lambda *x: jnp.concatenate(x, axis=0), *padded_transitions)


def generate_transition_batches(
    transitions: Transition,
    batch_size: int,
    group_by_length: bool = False,
    include_last_batch: bool = True,
) -> list[Transition]:
    """Generates batches of transitions.

    Args:
        transitions: The collected trajectories to batch.
        batch_size: The size of the batches to generate.
        group_by_length: Whether to group transitions by length, otherwise,
            transitions are grouped randomly.
        include_last_batch: Whether to include the last batch if it's not full.

    Returns:
        An iterator over the batches of transitions.
    """
    transitions = chunk_transitions(transitions)

    if group_by_length:
        # Sort transitions by length
        transitions.sort(key=lambda x: x.done.shape[1])

    # Group transitions by batch size
    batches: list[list[Transition]] = [transitions[i : i + batch_size] for i in range(0, len(transitions), batch_size)]

    # Remove the last batch if it's not full and `include_last_batch` is False.
    if not include_last_batch and len(batches[-1]) < batch_size:
        batches = batches[:-1]

    return [concatenate_transitions(batch) for batch in batches]
