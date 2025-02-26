"""MJX Utils."""

from dataclasses import dataclass

import jax
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx

from ksim.env.types import EnvState


@jax.tree_util.register_dataclass
@dataclass
class MjxEnvState(EnvState):
    """The state of the environment.

    Attributes (inheriteds):
        model: Handles physics and model definition (latter shouldn't be touched).
        data: Includes current state of the robot.
        obs: The post-processed observations of the environment.
        reward: The reward of the environment.
        done: Whether the episode is done.
        info: Additional information about the environment.
    """

    # MJX attributes
    mjx_model: mjx.Model
    mjx_data: mjx.Data

    # Classic state attributes
    obs: dict[str, Array]
    reward: Array
    done: Array
    commands: dict[str, Array]  # added for MJX envs... decoupled commands from observations

    # Auxiliary attributes
    time: Array
    rng: PRNGKeyArray
    command_at_prev_step: Array
    action_at_prev_step: Array
    action_log_prob_at_prev_step: Array
