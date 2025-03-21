"""Tests for termination builders in the ksim package."""

from dataclasses import dataclass

import attrs
import jax
import jax.numpy as jnp
import mujoco
from jaxtyping import Array
from mujoco import mjx

from ksim.terminations import (
    IllegalContactTermination,
    MinimumHeightTermination,
    PitchTooGreatTermination,
    RollTooGreatTermination,
    Termination,
)


@attrs.define(frozen=True)
class DummyTermination(Termination):
    """Dummy termination for testing."""

    def __call__(self, state: mjx.Data) -> Array:
        return jnp.zeros((1,), dtype=jnp.bool_)


class DummyContact:
    """Mock contact data for testing."""

    geom1: Array = jnp.array([0, 2])
    geom2: Array = jnp.array([1, 3])
    dist: Array = jnp.array([-0.01, -0.02])


@jax.tree_util.register_dataclass
@dataclass
class DummyMjxData:
    """Mock mjx.Data for testing."""

    def __init__(self, has_contact: bool = True) -> None:
        self._has_contact = has_contact
        self._qpos = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    @property
    def qpos(self) -> Array:
        return self._qpos

    @qpos.setter
    def qpos(self, value: Array) -> None:
        self._qpos = value

    @property
    def ncon(self) -> int:
        return 2 if self._has_contact else 0

    @property
    def contact(self) -> DummyContact:
        return DummyContact()


def test_termination_name() -> None:
    """Test that termination names are correctly generated."""
    term = DummyTermination()
    assert term.get_name() == "dummy_termination"
    assert term.termination_name == "dummy_termination"


class TestPitchTooGreatTermination:
    """Tests for the PitchTooGreatTermination class."""

    def test_pitch_too_great_termination(self) -> None:
        """Test that the PitchTooGreatTermination terminates when pitch exceeds the max."""
        data = DummyMjxData()

        # With a small max_pitch, should terminate
        term = PitchTooGreatTermination(max_pitch=0.1)
        result = term(data)
        assert result.item()

        # With a large max_pitch, should not terminate
        term = PitchTooGreatTermination(max_pitch=10.0)
        result = term(data)
        assert not result.item()


class TestRollTooGreatTermination:
    """Tests for the RollTooGreatTermination class."""

    def test_roll_too_great_termination(self) -> None:
        """Test that the RollTooGreatTermination terminates when roll exceeds the max."""
        data = DummyMjxData()

        # With a small max_roll, should terminate
        term = RollTooGreatTermination(max_roll=0.1)
        result = term(data)
        assert result.item()

        # With a large max_roll, should not terminate
        term = RollTooGreatTermination(max_roll=10.0)
        result = term(data)
        assert not result.item()


class TestMinimumHeightTermination:
    """Tests for the MinimumHeightTermination class."""

    def test_minimum_height_termination(self) -> None:
        """Test that the MinimumHeightTermination terminates when height is below the minimum."""
        # With min_height above the current height, should terminate
        # The default qpos[2] in DummyMjxData is 3.0
        data = DummyMjxData()
        term = MinimumHeightTermination(min_height=4.0)
        result = term(data)
        assert result.item()

        # With min_height below the current height, should not terminate
        term = MinimumHeightTermination(min_height=2.0)
        result = term(data)
        assert not result.item()


class TestIllegalContactTermination:
    """Tests for the IllegalContactTermination class."""

    def test_illegal_contact_termination(self) -> None:
        """Test that the IllegalContactTermination terminates when there's an illegal contact."""
        # Test with illegal contact present (original DummyMjxData has geom1=[0,2] and geom2=[1,3])
        data = DummyMjxData()
        term = IllegalContactTermination(illegal_geom_idxs=jnp.array([0, 3]))
        result = term(data)
        assert result.item()

        # Test with no illegal contacts
        term = IllegalContactTermination(illegal_geom_idxs=jnp.array([4, 5]))
        result = term(data)
        assert not result.item()

        # Test with no contacts at all
        data_no_contact = DummyMjxData(has_contact=False)
        term = IllegalContactTermination(illegal_geom_idxs=jnp.array([0, 1]))
        result = term(data_no_contact)
        assert not result.item()

    def test_illegal_contact_termination_builder(self, humanoid_model: mujoco.MjModel) -> None:
        """Test that the IllegalContactTerminationBuilder creates a termination function."""
        geom_names = ["hand_left", "hand_right"]
        term = IllegalContactTermination.create(humanoid_model, geom_names=geom_names)
        assert term.termination_name == "illegal_contact_termination"
        assert term.illegal_geom_idxs.shape == (2,)
