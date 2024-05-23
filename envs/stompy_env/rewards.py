from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jp
from brax import base
from brax.mjx.base import State as mjxState

DEFAULT_REWARD_PARAMS = {
    "rew_forward": {"weight": 1.25},
    "rew_healthy": {"weight": 5.0, "healthy_z_lower": 1.0, "healthy_z_upper": 2.0},
    "rew_ctrl_cost": {"weight": 0.1},
}

def get_reward_fn(
    reward_params: Dict[str, Dict[str, float]], dt, include_reward_breakdown
) -> Callable[[mjxState, jp.ndarray, mjxState], Tuple[jp.ndarray, jp.ndarray, Dict[str, jp.ndarray]]]:
    """Get a combined reward function.

    Args:
        reward_params: Dictionary of reward parameters.
        dt: Time step.
    Returns:
        A reward function that takes in a state, action, and next state and returns a float wrapped in a jp.ndarray.
    """

    def reward_fn(
        state: mjxState, action: jp.ndarray, next_state: mjxState
    ) -> Tuple[jp.ndarray, jp.ndarray, Dict[str, jp.ndarray]]:
        reward, is_healthy = jp.array(0.0), jp.array(1.0)
        rewards = {}
        for key, params in reward_params.items():
            r, h = reward_functions[key](state, action, next_state, dt, params)
            is_healthy *= h
            reward += r
            if include_reward_breakdown:  # For more detailed logging, can be disabled for performance
                rewards[key] = r
        return reward, is_healthy, rewards

    return reward_fn


def forward_reward_fn(
    state: mjxState, action: jp.ndarray, next_state: mjxState, dt: jax.Array, params: Dict[str, float]
) -> Tuple[jp.ndarray, jp.ndarray]:
    """Reward function for moving forward.

    Args:
        state: Current state.
        action: Action taken.
        next_state: Next state.
        dt: Time step.
        params: Reward parameters.
    Returns:
        A float wrapped in a jax array.
    """
    xpos = state.subtree_com[1][0]  # TODO: include stricter typing than mjxState to avoid this type error
    next_xpos = next_state.subtree_com[1][0]
    velocity = (next_xpos - xpos) / dt
    forward_reward = params["weight"] * velocity

    return forward_reward, jp.array(1.0)  # TODO: ensure everything is initialized in a size 2 array instead...


def healthy_reward_fn(
    state: mjxState, action: jp.ndarray, next_state: mjxState, dt: jax.Array, params: Dict[str, float]
) -> Tuple[jp.ndarray, jp.ndarray]:
    """Reward function for staying healthy.

    Args:
        state: Current state.
        action: Action taken.
        next_state: Next state.
        dt: Time step.
        params: Reward parameters.
    Returns:
        A float wrapped in a jax array.
    """
    min_z = params["healthy_z_lower"]
    max_z = params["healthy_z_upper"]
    is_healthy = jp.where(state.q[2] < min_z, 0.0, 1.0)
    is_healthy = jp.where(state.q[2] > max_z, 0.0, is_healthy)
    healthy_reward = jp.array(params["weight"]) * is_healthy

    return healthy_reward, is_healthy


def ctrl_cost_reward_fn(
    state: mjxState, action: jp.ndarray, next_state: mjxState, dt: jax.Array, params: Dict[str, float]
) -> Tuple[jp.ndarray, jp.ndarray]:
    """Reward function for control cost.

    Args:
        state: Current state.
        action: Action taken.
        next_state: Next state.
        dt: Time step.
        params: Reward parameters.
    Returns:
        A float wrapped in a jax array.
    """
    ctrl_cost = -params["weight"] * jp.sum(jp.square(action))

    return ctrl_cost, jp.array(1.0)


reward_functions = {
    "rew_forward": forward_reward_fn,
    "rew_healthy": healthy_reward_fn,
    "rew_ctrl_cost": ctrl_cost_reward_fn,
}
