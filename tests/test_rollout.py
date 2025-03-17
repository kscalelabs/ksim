"""Tests for rollouts functionality."""

from typing import Collection

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import pytest
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray
from mujoco import mjx
from xax.nn.distributions import TanhGaussianDistribution

from ksim.actuators import TorqueActuators
from ksim.commands import Command
from ksim.env.data import PhysicsData, PhysicsState
from ksim.env.mjx_engine import MjxEngine
from ksim.env.unroll import unroll_trajectory
from ksim.model.base import ActorCriticAgent, KSimModule
from ksim.model.types import ModelCarry
from ksim.normalization import Normalizer, PassThrough
from ksim.observation import Observation
from ksim.resets import RandomizeJointPositions, RandomizeJointVelocities
from ksim.rewards import Reward
from ksim.terminations import Termination

_ACTION_DIM = 21
_NUM_STEPS = 4


class DummyTermination(Termination):
    def __call__(self, physics_state: PhysicsState) -> Array:
        return jnp.array(False)


class DummyReward(Reward):
    def __call__(
        self,
        prev_action: Array | None,
        physics_state: PhysicsData,
        command: FrozenDict[str, Array],
        action: Array,
        next_physics_state: PhysicsData,
        next_state_terminates: Array,
    ) -> Array:
        return jnp.array(0.0)


class DummyObservation(Observation):
    def observe(self, physics_state: PhysicsState, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(1)


class DummyCommand(Command):
    def initial_command(self, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(1)

    def __call__(self, prev_command: Array | None, time: Array, rng: PRNGKeyArray) -> Array:
        return jnp.zeros(1)


def get_observation(
    physics_state: PhysicsState, rng: PRNGKeyArray, *, obs_generators: Collection[Observation]
) -> FrozenDict[str, Array]:
    return FrozenDict({"some_observation": jnp.zeros(1)})


def get_initial_commands(rng: PRNGKeyArray, *, command_generators: Collection[Command]) -> FrozenDict[str, Array]:
    return FrozenDict({"some_command": jnp.zeros(1)})


def get_obs_normalizer(dummy_obs: FrozenDict[str, Array]) -> Normalizer:
    return PassThrough()


def get_cmd_normalizer(dummy_cmd: FrozenDict[str, Array]) -> Normalizer:
    return PassThrough()


class DummyActor(eqx.Module, KSimModule):
    def __init__(self, key: PRNGKeyArray) -> None:
        pass

    def forward(
        self, obs: FrozenDict[str, Array], command: FrozenDict[str, Array], carry: ModelCarry | None
    ) -> tuple[Array, ModelCarry]:
        return jnp.zeros(2 * _ACTION_DIM), None


class DummyCritic(eqx.Module, KSimModule):
    def __init__(self, key: PRNGKeyArray) -> None:
        pass

    def forward(
        self, obs: FrozenDict[str, Array], command: FrozenDict[str, Array], carry: ModelCarry | None
    ) -> tuple[Array, ModelCarry]:
        return jnp.zeros(1), None


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


@pytest.fixture
def agent() -> ActorCriticAgent:
    rng = jax.random.PRNGKey(0)
    return ActorCriticAgent(
        actor_model=DummyActor(rng),
        critic_model=DummyCritic(rng),
        action_distribution=TanhGaussianDistribution(action_dim=_ACTION_DIM),
    )


dummy_obs = DummyObservation()
dummy_cmd = DummyCommand()
dummy_term = DummyTermination()
dummy_rew = DummyReward(scale=1.0)

obs_normalizer = PassThrough()
cmd_normalizer = PassThrough()


@pytest.mark.slow
def test_unroll_shapes(engine: MjxEngine, agent: ActorCriticAgent):
    """Test that unroll_trajectory returns the correct shapes."""
    rng = jax.random.PRNGKey(0)
    initial_physics_state = engine.reset(rng)

    transitions, final_state, unroll_nan_detector, intermediate_data = unroll_trajectory(
        physics_state=initial_physics_state,
        rng=rng,
        agent=agent,
        obs_normalizer=obs_normalizer,
        cmd_normalizer=cmd_normalizer,
        engine=engine,
        obs_generators=[dummy_obs],
        command_generators=[dummy_cmd],
        reward_generators=[dummy_rew],
        termination_generators=[dummy_term],
        num_steps=_NUM_STEPS,
        return_intermediate_physics_data=False,
    )

    assert transitions.obs["dummy_observation_proprio_gaussian"].shape == (4, 1)
    assert transitions.command["dummy_command_vector"].shape == (4, 1)
    assert final_state.data.qpos.shape == (28,)
    for field, value in unroll_nan_detector._asdict().items():
        assert not value, f"{field} contains NaNs"
    assert intermediate_data is None


@pytest.mark.slow
def test_unroll_jittable(engine: MjxEngine, agent: ActorCriticAgent):
    """Test that engine can be jitted."""
    rng = jax.random.PRNGKey(0)
    initial_physics_state = engine.reset(rng)

    jitted_unroll = eqx.filter_jit(unroll_trajectory)
    _, _, unroll_nan_detector, _ = jitted_unroll(
        physics_state=initial_physics_state,
        rng=rng,
        agent=agent,
        obs_normalizer=obs_normalizer,
        cmd_normalizer=cmd_normalizer,
        engine=engine,
        obs_generators=[dummy_obs],
        command_generators=[dummy_cmd],
        reward_generators=[dummy_rew],
        termination_generators=[dummy_term],
        num_steps=_NUM_STEPS,
        return_intermediate_physics_data=False,
    )
    for field, value in unroll_nan_detector._asdict().items():
        assert not value, f"{field} contains NaNs"
