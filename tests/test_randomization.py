"""Tests for randomization classes."""

import jax
import jax.numpy as jnp
import pytest
from mujoco import mjx

from ksim.env.data import PhysicsModel
from ksim.randomization import (
    ArmatureRandomization,
    FloorFrictionRandomization,
    JointDampingRandomization,
    StaticFrictionRandomization,
    TorsoMassRandomization,
    WeightRandomization,
)


class TestWeightRandomization:
    """Tests for the WeightRandomization class."""

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(0)

    def test_weight_randomization(self, humanoid_model: PhysicsModel, rng: jax.Array) -> None:
        """Test that the weight randomization works."""
        randomizer = WeightRandomization(scale=0.1)
        new_model = randomizer(humanoid_model, rng)
        assert new_model.body_mass.shape == humanoid_model.body_mass.shape
        assert not jnp.allclose(new_model.body_mass, humanoid_model.body_mass)


class TestStaticFrictionRandomization:
    """Tests for the StaticFrictionRandomization class."""

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(0)

    def test_static_friction_randomization(self, humanoid_model: PhysicsModel, rng: jax.Array) -> None:
        randomizer = StaticFrictionRandomization()
        new_model = randomizer(humanoid_model, rng)
        assert new_model.dof_frictionloss.shape == humanoid_model.dof_frictionloss.shape
        assert not jnp.allclose(new_model.dof_frictionloss, humanoid_model.dof_frictionloss)

    def test_randomization_name(self) -> None:
        """Test that the randomization name is correctly generated."""
        randomizer = StaticFrictionRandomization()
        assert randomizer.name == "dof_frictionloss"


class TestArmatureRandomization:
    """Tests for the ArmatureRandomization class."""

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(0)

    def test_armature_randomization(self, humanoid_model: PhysicsModel, rng: jax.Array) -> None:
        randomizer = ArmatureRandomization()
        new_model = randomizer(humanoid_model, rng)
        assert new_model.dof_armature.shape == humanoid_model.dof_armature.shape
        assert not jnp.allclose(new_model.dof_armature, humanoid_model.dof_armature)


class TestTorsoMassRandomization:
    """Tests for the TorsoMassRandomization class."""

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(0)

    def test_torso_mass_randomization(self, humanoid_model: PhysicsModel, rng: jax.Array) -> None:
        randomizer = TorsoMassRandomization(torso_body_id=0)
        new_model = randomizer(humanoid_model, rng)
        assert new_model.body_mass.shape == humanoid_model.body_mass.shape
        assert new_model.body_mass[0] != humanoid_model.body_mass[0]
        assert jnp.allclose(new_model.body_mass[1:], humanoid_model.body_mass[1:])


class TestFloorFrictionRandomization:
    """Tests for the FloorFrictionRandomization class."""

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(0)

    def test_floor_friction_randomization(self, humanoid_model: PhysicsModel, rng: jax.Array) -> None:
        randomizer = FloorFrictionRandomization(floor_body_id=0)
        new_model = randomizer(humanoid_model, rng)
        assert new_model.geom_friction.shape == humanoid_model.geom_friction.shape
        assert not jnp.allclose(new_model.geom_friction, humanoid_model.geom_friction)


class TestJointDampingRandomization:
    """Tests for the JointDampingRandomization class."""

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(0)

    def test_joint_damping_randomization(self, humanoid_model: mjx.Model, rng: jax.Array) -> None:
        randomizer = JointDampingRandomization()
        new_model = randomizer(humanoid_model, rng)
        assert new_model.dof_damping.shape == humanoid_model.dof_damping.shape
        assert not jnp.allclose(new_model.dof_damping, humanoid_model.dof_damping)
