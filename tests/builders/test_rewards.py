"""Tests for reward builders in the ksim package."""

from dataclasses import dataclass, field

import attrs
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jaxtyping import Array
from mujoco import mjx

from ksim.builders.rewards import (
    ActionSmoothnessPenalty,
    AngularVelocityXYPenalty,
    FootContactPenalty,
    LinearVelocityZPenalty,
    Reward,
    TrackAngularVelocityZReward,
    TrackLinearVelocityXYReward,
)


@attrs.define(frozen=True, kw_only=True)
class DummyReward(Reward):
    """Dummy reward for testing."""

    def __call__(
        self,
        action_t_minus_1: Array | None,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> Array:
        return jnp.zeros((1,), dtype=jnp.float32)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DummyContact:
    """Mock mjx.Contact for testing."""

    geom1: Array = field(default_factory=lambda: jnp.array([0, 1, 2]))
    geom2: Array = field(default_factory=lambda: jnp.array([3, 4, 5]))
    dist: Array = field(default_factory=lambda: jnp.array([0.01, 0.02, 0.03]))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DummyMjxData:
    """Mock mjx.Data for testing."""

    time: float = 0.0
    qpos: Array = field(default_factory=lambda: jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    qvel: Array = field(default_factory=lambda: jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    ctrl: Array = field(default_factory=lambda: jnp.array([0.5, -0.5, 0.25]))
    body_xpos: Array = field(default_factory=lambda: jnp.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))
    body_xmat: Array = field(
        default_factory=lambda: jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    contact: DummyContact = field(default_factory=DummyContact)


def test_reward_name() -> None:
    """Test that reward names are correctly generated."""
    reward = DummyReward(scale=1.0)
    assert reward.get_name() == "dummy_reward"
    assert reward.reward_name == "dummy_reward"


class TestLinearVelocityZPenalty:
    """Tests for the LinearVelocityZPenalty class."""

    def test_linear_velocity_z_penalty(self) -> None:
        """Test that the LinearVelocityZPenalty returns the correct penalty."""
        scale = 0.1
        reward = LinearVelocityZPenalty(scale=scale)

        data_t = DummyMjxData()
        data_t_plus_1 = DummyMjxData(qvel=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

        command: FrozenDict[str, Array] = FrozenDict({})
        action_t_minus_1 = jnp.array([0.0, 0.0, 0.0])
        action_t = jnp.array([0.1, 0.1, 0.1])

        result = reward(action_t_minus_1, data_t, command, action_t, data_t_plus_1) * scale

        # Expected penalty is scale * z_velocity^2 for L2 norm
        expected = scale * (data_t_plus_1.qvel[2] ** 2)
        assert jnp.allclose(result, expected)


class TestAngularVelocityXYPenalty:
    """Tests for the AngularVelocityXYPenalty class."""

    def test_angular_velocity_xy_penalty(self) -> None:
        """Test that the AngularVelocityXYPenalty returns the correct penalty."""
        scale = 0.2
        reward = AngularVelocityXYPenalty(scale=scale)

        data_t = DummyMjxData()
        data_t_plus_1 = DummyMjxData(qvel=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

        command: FrozenDict[str, Array] = FrozenDict({})
        action_t_minus_1 = jnp.array([0.0, 0.0, 0.0])
        action_t = jnp.array([0.1, 0.1, 0.1])

        result = reward(action_t_minus_1, data_t, command, action_t, data_t_plus_1) * scale

        # Expected penalty is scale * sum(xy_angular_velocity^2) for L2 norm
        expected = scale * ((data_t_plus_1.qvel[3] ** 2) + (data_t_plus_1.qvel[4] ** 2))
        assert jnp.allclose(result, expected)


class TestTrackAngularVelocityZReward:
    """Tests for the TrackAngularVelocityZReward class."""

    def test_track_angular_velocity_z_reward(self) -> None:
        """Test that the TrackAngularVelocityZReward returns the correct reward."""
        scale = 0.3
        reward = TrackAngularVelocityZReward(scale=scale)

        data_t = DummyMjxData()
        data_t_plus_1 = DummyMjxData(qvel=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

        command: FrozenDict[str, Array] = FrozenDict(
            {"angular_velocity_command_vector": jnp.array([0.5])}
        )
        action_t_minus_1 = jnp.array([0.0, 0.0, 0.0])
        action_t = jnp.array([0.1, 0.1, 0.1])

        result = reward(action_t_minus_1, data_t, command, action_t, data_t_plus_1) * scale

        # Expected reward is scale * (z_angular_velocity * command)^2 for L2 norm
        expected = (
            scale * (data_t_plus_1.qvel[5] * command["angular_velocity_command_vector"][0]) ** 2
        )
        assert jnp.allclose(result, expected)


class TestTrackLinearVelocityXYReward:
    """Tests for the TrackLinearVelocityXYReward class."""

    def test_track_linear_velocity_xy_reward(self) -> None:
        """Test that the TrackLinearVelocityXYReward returns the correct reward."""
        scale = 0.4
        sensitivity = 1.0
        reward = TrackLinearVelocityXYReward(scale=scale, sensitivity=sensitivity)

        data_t = DummyMjxData()
        data_t_plus_1 = DummyMjxData(qvel=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

        command: FrozenDict[str, Array] = FrozenDict(
            {"linear_velocity_command_vector": jnp.array([0.5, 0.5])}
        )
        action_t_minus_1 = jnp.array([0.0, 0.0, 0.0])
        action_t = jnp.array([0.1, 0.1, 0.1])

        result = reward(action_t_minus_1, data_t, command, action_t, data_t_plus_1) * scale

        # Expected reward uses exponential decay based on L2 tracking error
        tracking_error = jnp.linalg.norm(
            command["linear_velocity_command_vector"] - data_t_plus_1.qvel[:2]
        )
        expected = scale * jnp.exp(-sensitivity * tracking_error)

        assert jnp.allclose(result, expected)


class TestActionSmoothnessPenalty:
    """Tests for the ActionSmoothnessPenalty class."""

    def test_action_smoothness_penalty(self) -> None:
        """Test that the ActionSmoothnessPenalty returns the correct penalty."""
        scale = 0.5
        reward = ActionSmoothnessPenalty(scale=scale)

        data_t = DummyMjxData()
        data_t_plus_1 = DummyMjxData(qvel=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

        command: FrozenDict[str, Array] = FrozenDict({})
        action_t_minus_1 = jnp.array([0.0, 0.0, 0.0])
        action_t = jnp.array([0.1, 0.1, 0.1])

        result = reward(action_t_minus_1, data_t, command, action_t, data_t_plus_1) * scale

        # Expected penalty is scale * sum((action_t - action_t_minus_1)^2) for L2 norm
        expected = scale * jnp.sum((action_t - action_t_minus_1) ** 2)
        assert jnp.allclose(result, expected)

        # Test with None for action_t_minus_1
        result = reward(None, data_t, command, action_t, data_t_plus_1)
        assert jnp.allclose(result, 0.0)


class TestFootContactPenalty:
    """Tests for the FootContactPenalty class."""

    def test_foot_contact_penalty(self) -> None:
        """Test that the FootContactPenalty returns the correct penalty."""
        scale = 0.6
        illegal_geom_idxs = jnp.array([0, 1])
        allowed_contact_prct = 0.2
        reward = FootContactPenalty(
            scale=scale,
            illegal_geom_idxs=illegal_geom_idxs,
            allowed_contact_prct=allowed_contact_prct,
        )

        data_t = DummyMjxData()
        data_t_plus_1 = DummyMjxData(contact=DummyContact(geom1=jnp.array([0, 3, 5])))

        command: FrozenDict[str, Array] = FrozenDict({})
        action_t_minus_1 = jnp.array([0.0, 0.0, 0.0])
        action_t = jnp.array([0.1, 0.1, 0.1])

        result = reward(action_t_minus_1, data_t, command, action_t, data_t_plus_1) * scale

        # Expected penalty is scale * 1.0 (since there is contact with an illegal geom)
        expected = scale * 1.0
        assert jnp.allclose(result, expected)

        # Test with no illegal contacts
        data_t_plus_1 = DummyMjxData(contact=DummyContact(geom1=jnp.array([3, 4, 5])))

        result = reward(action_t_minus_1, data_t, command, action_t, data_t_plus_1) * scale

        # Expected penalty is scale * 0.0 (since there is no contact with an illegal geom)
        expected = scale * 0.0
        assert jnp.allclose(result, expected)
