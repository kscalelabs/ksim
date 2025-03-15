"""Tests for randomization classes in the ksim package."""

import jax
import jax.numpy as jnp
import pytest

from ksim.env.types import PhysicsModel
from ksim.randomization import (
    ArmatureRandomization,
    FloorFrictionRandomization,
    JointDampingRandomization,
    LinkMassRandomization,
    StaticFrictionRandomization,
    TorsoMassRandomization,
)


class DummyPhysicsModel:
    """Mock mjx.Model for testing."""

    def __init__(self) -> None:
        self.dof_frictionloss = jnp.ones((18,))
        self.dof_armature = jnp.ones((12,))
        self.body_mass = jnp.ones((10,))
        self.geom_friction = jnp.ones((10, 1))
        self.dof_damping = jnp.ones((18,))
        self.nq = 12
        self.nbody = 10


def test_randomization_name() -> None:
    """Test that the randomization name is correctly generated."""
    randomizer = StaticFrictionRandomization()
    assert randomizer.randomization_name == "dof_frictionloss"


class TestRandomizers:
    """Tests for the randomization classes."""

    @pytest.fixture
    def model(self) -> PhysicsModel:
        """Return a dummy PhysicsModel."""
        return DummyPhysicsModel()

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(44)

    def test_static_friction_randomization(self, model: PhysicsModel, rng: jax.Array) -> None:
        randomizer = StaticFrictionRandomization()
        result = randomizer(model, rng)
        assert result.shape == model.dof_frictionloss.shape
        assert not jnp.allclose(result, model.dof_frictionloss)

    def test_armature_randomization(self, model: PhysicsModel, rng: jax.Array) -> None:
        randomizer = ArmatureRandomization()
        result = randomizer(model, rng)
        assert result.shape == model.dof_armature.shape
        assert not jnp.allclose(result, model.dof_armature)

    def test_link_mass_randomization(self, model: PhysicsModel, rng: jax.Array) -> None:
        randomizer = LinkMassRandomization()
        result = randomizer(model, rng)
        assert result.shape == model.body_mass.shape
        assert not jnp.allclose(result, model.body_mass)

    def test_torso_mass_randomization(self, model: PhysicsModel, rng: jax.Array) -> None:
        randomizer = TorsoMassRandomization(torso_body_id=0)
        result = randomizer(model, rng)
        assert result.shape == model.body_mass.shape
        assert result[0] != model.body_mass[0]
        assert jnp.allclose(result[1:], model.body_mass[1:])

    def test_floor_friction_randomization(self, model: PhysicsModel, rng: jax.Array) -> None:
        randomizer = FloorFrictionRandomization(floor_body_id=0)
        result = randomizer(model, rng)
        assert result.shape == model.geom_friction.shape
        assert not jnp.allclose(result, model.geom_friction)

    def test_joint_damping_randomization(self, model: PhysicsModel, rng: jax.Array) -> None:
        randomizer = JointDampingRandomization()
        result = randomizer(model, rng)
        assert result.shape == model.dof_damping.shape
        assert not jnp.allclose(result, model.dof_damping)
