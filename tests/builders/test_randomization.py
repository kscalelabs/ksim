"""Tests for randomization classes."""

import jax
import jax.numpy as jnp
import mujoco
import pytest
from mujoco import mjx

from ksim.env.data import PhysicsData, PhysicsModel
from ksim.randomization import (
    ArmatureRandomizer,
    FloorFrictionRandomizer,
    JointDampingRandomizer,
    LinkMassRandomizer,
    StaticFrictionRandomizer,
    TorsoMassRandomizer,
)


def get_dummy_model() -> mjx.Model:
    """Get a dummy mjx.Model for testing."""
    mj_model = mujoco.MjModel.from_xml_path("tests/fixed_assets/default_humanoid_test.mjcf")
    mjx_model = mjx.put_model(mj_model)
    return mjx_model


class DummyPhysicsData:
    """Mock mjx.Data for testing."""

    def __init__(self) -> None:
        self.qpos = jnp.ones((12,))
        self.qvel = jnp.ones((12,))


def test_randomization_name() -> None:
    """Test that the randomization name is correctly generated."""
    randomizer = StaticFrictionRandomizer()
    assert randomizer.name == "dof_frictionloss"


class TestRandomizers:
    """Tests for the randomization classes."""

    @pytest.fixture
    def model(self) -> mjx.Model:
        """Return a dummy mjx.Model."""
        return get_dummy_model()

    @pytest.fixture
    def data(self) -> mjx.Data:
        """Return a dummy mjx.Data."""
        return DummyPhysicsData()

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(44)

    def test_static_friction_randomization(self, model: PhysicsModel, data: PhysicsData, rng: jax.Array) -> None:
        randomizer = StaticFrictionRandomizer()
        new_model, data = randomizer(model, data, rng)
        assert new_model.dof_frictionloss.shape == model.dof_frictionloss.shape
        assert not jnp.allclose(new_model.dof_frictionloss, model.dof_frictionloss)

    def test_armature_randomization(self, model: PhysicsModel, data: PhysicsData, rng: jax.Array) -> None:
        randomizer = ArmatureRandomizer()
        new_model, data = randomizer(model, data, rng)
        assert new_model.dof_armature.shape == model.dof_armature.shape
        assert not jnp.allclose(new_model.dof_armature, model.dof_armature)

    def test_link_mass_randomization(self, model: PhysicsModel, data: PhysicsData, rng: jax.Array) -> None:
        randomizer = LinkMassRandomizer()
        new_model, data = randomizer(model, data, rng)
        assert new_model.body_mass.shape == model.body_mass.shape
        assert not jnp.allclose(new_model.body_mass, model.body_mass)

    def test_torso_mass_randomization(self, model: PhysicsModel, data: PhysicsData, rng: jax.Array) -> None:
        randomizer = TorsoMassRandomizer(torso_body_id=0)
        new_model, data = randomizer(model, data, rng)
        assert new_model.body_mass.shape == model.body_mass.shape
        assert new_model.body_mass[0] != model.body_mass[0]
        assert jnp.allclose(new_model.body_mass[1:], model.body_mass[1:])

    def test_floor_friction_randomization(self, model: PhysicsModel, data: PhysicsData, rng: jax.Array) -> None:
        randomizer = FloorFrictionRandomizer(floor_body_id=0)
        new_model, data = randomizer(model, data, rng)
        assert new_model.geom_friction.shape == model.geom_friction.shape
        assert not jnp.allclose(new_model.geom_friction, model.geom_friction)

    def test_joint_damping_randomization(self, model: PhysicsModel, data: PhysicsData, rng: jax.Array) -> None:
        randomizer = JointDampingRandomizer()
        new_model, data = randomizer(model, data, rng)
        assert new_model.dof_damping.shape == model.dof_damping.shape
        assert not jnp.allclose(new_model.dof_damping, model.dof_damping)
