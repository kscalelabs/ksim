"""Tests for the MjxEngine class."""

import jax
import jax.numpy as jnp
import mujoco
import pytest
from jax import random
from mujoco import mjx

from ksim.actuators import TorqueActuators
from ksim.env.data import PhysicsState
from ksim.env.mjx_engine import MjxEngine
from ksim.resets import RandomizeJointPositions, RandomizeJointVelocities


@pytest.fixture
def engine() -> MjxEngine:
    mj_model = mujoco.MjModel.from_xml_path("tests/fixed_assets/default_humanoid_test.mjcf")
    mjx_model = mjx.put_model(mj_model)
    engine = MjxEngine(
        default_physics_model=mjx_model,
        resetters=[
            RandomizeJointPositions(scale=0.01),
            RandomizeJointVelocities(scale=0.01),
        ],
        actuators=TorqueActuators(),
        dt=0.005,
        ctrl_dt=0.02,
        min_action_latency_step=0,
        max_action_latency_step=0,
    )
    return engine


def test_engine_jittable(engine: MjxEngine) -> None:
    """Test that engine can be jitted."""
    action = jnp.zeros(21)
    rng = random.PRNGKey(0)
    state = jax.jit(engine.reset)(rng)
    jitted_step = jax.jit(engine.step)
    next_state = jitted_step(action, state, rng)

    assert isinstance(next_state, PhysicsState)


def test_engine_vmappable(engine: MjxEngine) -> None:
    """Test that engine reset can be vmapped across multiple environments."""
    num_envs = 3
    rng = random.PRNGKey(0)
    keys = random.split(rng, num_envs)

    vmapped_reset = jax.vmap(jax.jit(engine.reset), in_axes=(0,))
    states = vmapped_reset(keys)

    assert len(jax.tree_util.tree_leaves(states)[0]) == num_envs

    action = jnp.zeros((num_envs, 21))
    vmapped_step = jax.vmap(jax.jit(engine.step), in_axes=(0, 0, 0))
    next_states = vmapped_step(action, states, keys)

    assert isinstance(next_states, PhysicsState)
