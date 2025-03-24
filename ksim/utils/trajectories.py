"""Utilities for handling trajectory data."""

__all__ = [
    "split_and_pad_trajectories",
    "unpad_trajectories",
]

import jax
import jax.numpy as jnp


def split_and_pad_trajectories(
    tensor_tne: jnp.ndarray,
    dones_tn: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Split trajectories at done flags and pad to equal length.

    Args:
        tensor_tne: Tensor of shape [time, num_envs, ...] containing trajectory data
        dones_tn: Binary tensor of shape [time, num_envs] containing done flags

    Returns:
        padded_trajectories: Padded trajectories with done flags used as splits
        trajectory_masks: Binary masks indicating valid (unpadded) timesteps
    """
    # Ensure dones end each trajectory
    dones = dones_tn.copy()
    dones = dones.at[-1].set(True)

    # Get indices where trajectories end
    flat_dones = jnp.reshape(jnp.transpose(dones, (1, 0)), (-1,))
    done_indices = jnp.concatenate([jnp.array([-1]), jnp.where(flat_dones)[0]])

    # Calculate trajectory lengths
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    max_trajectory_length = jnp.max(trajectory_lengths)

    # Flatten the time and batch dimensions
    flat_tensor = jnp.reshape(
        jnp.transpose(tensor_tne, (1, 0, *range(2, tensor_tne.ndim))), (-1, *tensor_tne.shape[2:])
    )

    # Initialize padded trajectories and masks
    num_trajectories = len(trajectory_lengths)
    padded_shape = (max_trajectory_length, num_trajectories, *tensor_tne.shape[2:])
    padded_trajectories = jnp.zeros(padded_shape, dtype=tensor_tne.dtype)
    trajectory_masks = jnp.zeros((max_trajectory_length, num_trajectories), dtype=jnp.bool_)

    def scan_fn(
        carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        x: jnp.ndarray,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], None]:
        start_idx, traj_idx, padded_trajectories, trajectory_masks = carry
        length = x

        # Extract and pad trajectory
        trajectory = flat_tensor[start_idx : start_idx + length]
        padded = jnp.pad(trajectory, [(0, max_trajectory_length - length)] + [(0, 0)] * (tensor_tne.ndim - 2))
        mask = jnp.arange(max_trajectory_length) < length

        # Update arrays
        padded_trajectories = padded_trajectories.at[:, traj_idx].set(padded)
        trajectory_masks = trajectory_masks.at[:, traj_idx].set(mask)

        return (start_idx + length, traj_idx + 1, padded_trajectories, trajectory_masks), None

    init_carry = (
        jnp.zeros((), dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
        padded_trajectories,
        trajectory_masks,
    )

    (_, _, padded_trajectories, trajectory_masks), _ = jax.lax.scan(scan_fn, init_carry, trajectory_lengths)

    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories_tne: jnp.ndarray, masks_tn: jnp.ndarray) -> jnp.ndarray:
    """Removes padding from trajectories using the masks.

    Args:
        trajectories_tne: Padded trajectories tensor [time, num_envs, feature_dim]
        masks_tn: Boolean mask [time, num_envs]

    Returns:
        trajectories_tne: Unpadded trajectories
    """
    return (
        trajectories_tne.transpose(1, 0)[masks_tn.transpose(1, 0)]
        .reshape(-1, trajectories_tne.shape[0], trajectories_tne.shape[-1])
        .transpose(1, 0)
    )
