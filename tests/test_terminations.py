"""Tests for termination builders in the ksim package."""

from dataclasses import dataclass

import attrs
import jax
import jax.numpy as jnp
import mujoco
from jaxtyping import Array

import ksim


@attrs.define(frozen=True)
class DummyTermination(ksim.Termination):
    """Dummy termination for testing."""

    def __call__(self, state: ksim.PhysicsData, curriculum_level: Array) -> Array:
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


class TestNotUprightTermination:
    """Tests for the NotUprightTermination class."""

    def test_not_upright_termination(self) -> None:
        """Test that the NotUprightTermination terminates when pitch or roll exceeds the max."""
        data = DummyMjxData()
        curriculum_level = jnp.array(0.0)

        # With a small max_radians, should terminate
        term = ksim.NotUprightTermination(max_radians=0.1)
        result = term(data, curriculum_level)
        assert result.item()

        # With a large max_radians, should not terminate
        term = ksim.NotUprightTermination(max_radians=10.0)
        result = term(data, curriculum_level)
        assert not result.item()


class TestMinimumHeightTermination:
    """Tests for the MinimumHeightTermination class."""

    def test_minimum_height_termination(self) -> None:
        """Test that the MinimumHeightTermination terminates when height is below the minimum."""
        # With min_height above the current height, should terminate
        # The default qpos[2] in DummyMjxData is 3.0
        data = DummyMjxData()
        curriculum_level = jnp.array(0.0)

        term = ksim.MinimumHeightTermination(min_height=4.0)
        result = term(data, curriculum_level)
        assert result.item()

        # With min_height below the current height, should not terminate
        term = ksim.MinimumHeightTermination(min_height=2.0)
        result = term(data, curriculum_level)
        assert not result.item()


class TestIllegalContactTermination:
    """Tests for the IllegalContactTermination class."""

    def test_illegal_contact_termination(self) -> None:
        """Test that the IllegalContactTermination terminates when there's an illegal contact."""
        # Test with illegal contact present (original DummyMjxData has geom1=[0,2] and geom2=[1,3])
        data = DummyMjxData()
        curriculum_level = jnp.array(0.0)

        term = ksim.IllegalContactTermination(illegal_geom_idxs=jnp.array([0, 3]))
        result = term(data, curriculum_level)
        assert result.item()

        # Test with no illegal contacts
        term = ksim.IllegalContactTermination(illegal_geom_idxs=jnp.array([4, 5]))
        result = term(data, curriculum_level)
        assert not result.item()

        # Test with no contacts at all
        data_no_contact = DummyMjxData(has_contact=False)
        term = ksim.IllegalContactTermination(illegal_geom_idxs=jnp.array([0, 1]))
        result = term(data_no_contact, curriculum_level)
        assert not result.item()

    def test_illegal_contact_termination_builder(self, humanoid_model: mujoco.MjModel) -> None:
        """Test that the IllegalContactTerminationBuilder creates a termination function."""
        geom_names = ["hand_left", "hand_right"]
        term = ksim.IllegalContactTermination.create(humanoid_model, geom_names=geom_names)
        assert term.termination_name == "illegal_contact_termination"
        assert term.illegal_geom_idxs.shape == (2,)
