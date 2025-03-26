"""Tests for randomization functions in the ksim package."""

import jax
import mujoco
import numpy as np
import pytest
from mujoco import mjx

import ksim


@pytest.fixture
def rng() -> jax.Array:
    """Return a random number generator key."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def simple_model() -> mujoco.MjModel:
    """Create a simple model with 3 bodies and 3 joints for testing."""
    xml = """
    <mujoco>
        <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0" />
            <body name="torso" pos="0 0 0.5">
                <joint name="free_joint" type="free"/>
                <geom name="torso_geom" type="sphere" size="0.1" />
                <body name="body1" pos="0.2 0 0">
                    <joint name="joint1" type="hinge" axis="0 0 1" damping="0.1" armature="0.1" frictionloss="0.1" />
                    <geom name="body1_geom" type="capsule" size="0.05" fromto="0 0 0 0.2 0 0" />
                    <body name="body2" pos="0.2 0 0">
                        <joint name="joint2" type="hinge" axis="0 1 0"
                        damping="0.1" armature="0.1" frictionloss="0.1" />
                        <geom name="body2_geom" type="capsule" size="0.05" fromto="0 0 0 0.2 0 0" />
                    </body>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """
    mj_model = mujoco.MjModel.from_xml_string(xml)

    return mj_model


@pytest.fixture
def simple_mjx_model(simple_model: mujoco.MjModel) -> mjx.Model:
    """Create an MJX version of the simple model."""
    return mjx.put_model(simple_model)


def test_weight_randomization(simple_mjx_model: mjx.Model, rng: jax.Array) -> None:
    """Test the weight randomization."""
    scale = 0.1
    num_iterations = 10

    original_body_mass = simple_mjx_model.body_mass

    randomization = ksim.WeightRandomization(scale=scale)

    # Run multiple randomizations with different RNG keys
    for i in range(num_iterations):
        # Split the RNG key for this iteration
        rng, subkey = jax.random.split(rng)

        randomized_model = randomization(simple_mjx_model, subkey)

        assert randomized_model is not simple_mjx_model

        rand_mass = np.array(randomized_model.body_mass)
        orig_mass = np.array(original_body_mass)

        assert np.all(rand_mass >= orig_mass * (1 - scale))
        assert np.all(rand_mass <= orig_mass * (1 + scale))

        for attr_name in ["dof_damping", "dof_frictionloss", "dof_armature", "qpos0"]:
            assert np.array_equal(
                np.array(getattr(simple_mjx_model, attr_name)), np.array(getattr(randomized_model, attr_name))
            )


def test_floor_friction_randomization(simple_mjx_model: mjx.Model, rng: jax.Array) -> None:
    """Test the floor friction randomization."""
    # Test parameters
    floor_body_id = 0
    scale_lower = 0.4
    scale_upper = 1.0
    num_iterations = 10

    original_geom_friction = simple_mjx_model.geom_friction

    randomization = ksim.FloorFrictionRandomization(
        floor_body_id=floor_body_id, scale_lower=scale_lower, scale_upper=scale_upper
    )

    # Run multiple randomizations with different RNG keys
    for i in range(num_iterations):
        # Split the RNG key for this iteration
        rng, subkey = jax.random.split(rng)

        randomized_model = randomization(simple_mjx_model, subkey)

        assert randomized_model is not simple_mjx_model

        rand_friction = np.array(randomized_model.geom_friction)
        orig_friction = np.array(original_geom_friction)

        assert scale_lower <= rand_friction[floor_body_id, 0] <= scale_upper

        modified_friction = orig_friction.copy()
        modified_friction[floor_body_id, 0] = rand_friction[floor_body_id, 0]
        assert np.array_equal(rand_friction, modified_friction)

        for attr_name in ["body_mass", "dof_damping", "dof_armature", "qpos0"]:
            assert np.array_equal(
                np.array(getattr(simple_mjx_model, attr_name)), np.array(getattr(randomized_model, attr_name))
            )


def test_armature_randomization(simple_mjx_model: mjx.Model, rng: jax.Array) -> None:
    """Test the armature randomization."""
    # Test parameters
    scale_lower = 0.9
    scale_upper = 1.1
    num_iterations = 10

    original_dof_armature = simple_mjx_model.dof_armature

    randomization = ksim.ArmatureRandomization(scale_lower=scale_lower, scale_upper=scale_upper)

    # Run multiple randomizations with different RNG keys
    for i in range(num_iterations):
        # Split the RNG key for this iteration
        rng, subkey = jax.random.split(rng)

        randomized_model = randomization(simple_mjx_model, subkey)

        assert randomized_model is not simple_mjx_model

        rand_armature = np.array(randomized_model.dof_armature)
        orig_armature = np.array(original_dof_armature)

        assert np.array_equal(rand_armature[:6], orig_armature[:6])

        assert np.all(rand_armature[6:] >= orig_armature[6:] * scale_lower)
        assert np.all(rand_armature[6:] <= orig_armature[6:] * scale_upper)

        for attr_name in ["body_mass", "dof_damping", "dof_frictionloss", "qpos0"]:
            assert np.array_equal(
                np.array(getattr(simple_mjx_model, attr_name)), np.array(getattr(randomized_model, attr_name))
            )


def test_torso_mass_randomization(simple_mjx_model: mjx.Model, rng: jax.Array) -> None:
    """Test the torso mass randomization."""
    torso_body_id = 0
    scale_lower = -1.0
    scale_upper = 1.0
    num_iterations = 10

    original_body_mass = simple_mjx_model.body_mass

    randomization = ksim.TorsoMassRandomization(
        torso_body_id=torso_body_id, scale_lower=scale_lower, scale_upper=scale_upper
    )

    # Run multiple randomizations with different RNG keys
    for i in range(num_iterations):
        # Split the RNG key for this iteration
        rng, subkey = jax.random.split(rng)

        randomized_model = randomization(simple_mjx_model, subkey)

        assert randomized_model is not simple_mjx_model

        rand_mass = np.array(randomized_model.body_mass)

        assert scale_lower <= rand_mass[torso_body_id] <= scale_upper

        for i in range(len(original_body_mass)):
            if i != torso_body_id:
                assert np.array_equal(rand_mass[i], original_body_mass[i])

        for attr_name in ["dof_damping", "dof_frictionloss", "dof_armature", "qpos0"]:
            assert np.array_equal(
                np.array(getattr(simple_mjx_model, attr_name)), np.array(getattr(randomized_model, attr_name))
            )


def test_joint_damping_randomization(simple_mjx_model: mjx.Model, rng: jax.Array) -> None:
    """Test the joint damping randomization."""
    # Test parameters
    scale_lower = 0.9
    scale_upper = 1.1
    num_iterations = 10

    original_dof_damping = simple_mjx_model.dof_damping

    randomization = ksim.JointDampingRandomization(scale_lower=scale_lower, scale_upper=scale_upper)

    # Run multiple randomizations with different RNG keys
    for i in range(num_iterations):
        # Split the RNG key for this iteration
        rng, subkey = jax.random.split(rng)

        randomized_model = randomization(simple_mjx_model, subkey)

        assert randomized_model is not simple_mjx_model

        rand_damping = np.array(randomized_model.dof_damping)
        orig_damping = np.array(original_dof_damping)

        assert np.array_equal(rand_damping[:6], orig_damping[:6])

        assert np.all(rand_damping[6:] >= orig_damping[6:] * scale_lower)
        assert np.all(rand_damping[6:] <= orig_damping[6:] * scale_upper)

        for attr_name in ["body_mass", "dof_frictionloss", "dof_armature", "qpos0"]:
            assert np.array_equal(
                np.array(getattr(simple_mjx_model, attr_name)), np.array(getattr(randomized_model, attr_name))
            )


def test_joint_zero_position_randomization(simple_mjx_model: mjx.Model, rng: jax.Array) -> None:
    """Test the joint zero position randomization."""
    # Test parameters
    scale_lower = -0.1
    scale_upper = 0.1
    num_iterations = 10

    original_qpos0 = simple_mjx_model.qpos0

    randomization = ksim.JointZeroPositionRandomization(scale_lower=scale_lower, scale_upper=scale_upper)

    # Run multiple randomizations with different RNG keys
    for i in range(num_iterations):
        # Split the RNG key for this iteration
        rng, subkey = jax.random.split(rng)

        randomized_model = randomization(simple_mjx_model, subkey)

        assert randomized_model is not simple_mjx_model

        rand_qpos0 = np.array(randomized_model.qpos0)
        orig_qpos0 = np.array(original_qpos0)

        assert np.array_equal(rand_qpos0[:6], orig_qpos0[:6])

        assert np.all(rand_qpos0[6:] >= orig_qpos0[6:] + scale_lower)
        assert np.all(rand_qpos0[6:] <= orig_qpos0[6:] + scale_upper)

        for attr_name in ["body_mass", "dof_damping", "dof_frictionloss", "dof_armature"]:
            assert np.array_equal(
                np.array(getattr(simple_mjx_model, attr_name)), np.array(getattr(randomized_model, attr_name))
            )


def test_static_friction_randomization(simple_mjx_model: mjx.Model, rng: jax.Array) -> None:
    """Test the static friction randomization."""
    # Test parameters
    scale_lower = 0.5
    scale_upper = 2.0
    num_iterations = 10

    original_frictionloss = simple_mjx_model.dof_frictionloss

    randomization = ksim.StaticFrictionRandomization(scale_lower=scale_lower, scale_upper=scale_upper)

    # Run multiple randomizations with different RNG keys
    for i in range(num_iterations):
        # Split the RNG key for this iteration
        rng, subkey = jax.random.split(rng)

        # Apply randomization
        randomized_model = randomization(simple_mjx_model, subkey)

        assert randomized_model is not simple_mjx_model

        rand_frictionloss = np.array(randomized_model.dof_frictionloss)
        orig_frictionloss = np.array(original_frictionloss)

        assert np.array_equal(rand_frictionloss[:6], orig_frictionloss[:6])
        assert np.all(rand_frictionloss[6:] >= orig_frictionloss[6:] * scale_lower)
        assert np.all(rand_frictionloss[6:] <= orig_frictionloss[6:] * scale_upper)

        for attr_name in ["body_mass", "dof_damping", "dof_armature", "qpos0"]:
            assert np.array_equal(
                np.array(getattr(simple_mjx_model, attr_name)), np.array(getattr(randomized_model, attr_name))
            )
