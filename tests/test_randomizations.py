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


def test_floor_friction_randomization(
    simple_model: mujoco.MjModel,
    simple_mjx_model: mjx.Model,
    rng: jax.Array,
) -> None:
    randomizer = ksim.FloorFrictionRandomizer.from_geom_name(simple_model, "floor", scale_lower=0.4, scale_upper=1.0)
    mj_dict = randomizer(simple_model, rng)
    mjx_dict = randomizer(simple_mjx_model, rng)
    assert np.allclose(mj_dict["geom_friction"], mjx_dict["geom_friction"])


def test_armature_randomization(
    simple_model: mujoco.MjModel,
    simple_mjx_model: mjx.Model,
    rng: jax.Array,
) -> None:
    randomizer = ksim.ArmatureRandomizer(scale_lower=0.9, scale_upper=1.1)
    mj_dict = randomizer(simple_model, rng)
    mjx_dict = randomizer(simple_mjx_model, rng)
    assert np.allclose(mj_dict["dof_armature"], mjx_dict["dof_armature"])


def test_torso_mass_randomization(
    simple_model: mujoco.MjModel,
    simple_mjx_model: mjx.Model,
    rng: jax.Array,
) -> None:
    randomizer = ksim.MassAdditionRandomizer(body_id=0, scale_lower=-1.0, scale_upper=1.0)
    mj_dict = randomizer(simple_model, rng)
    mjx_dict = randomizer(simple_mjx_model, rng)
    assert np.allclose(mj_dict["body_mass"], mjx_dict["body_mass"])


def test_joint_damping_randomization(
    simple_model: mujoco.MjModel,
    simple_mjx_model: mjx.Model,
    rng: jax.Array,
) -> None:
    randomizer = ksim.JointDampingRandomizer(scale_lower=0.9, scale_upper=1.1)
    mj_dict = randomizer(simple_model, rng)
    mjx_dict = randomizer(simple_mjx_model, rng)
    assert np.allclose(mj_dict["dof_damping"], mjx_dict["dof_damping"])


def test_joint_zero_position_randomization(
    simple_model: mujoco.MjModel,
    simple_mjx_model: mjx.Model,
    rng: jax.Array,
) -> None:
    randomizer = ksim.JointZeroPositionRandomizer(scale_lower=-0.1, scale_upper=0.1)
    mj_dict = randomizer(simple_model, rng)
    mjx_dict = randomizer(simple_mjx_model, rng)
    assert np.allclose(mj_dict["qpos0"], mjx_dict["qpos0"])


def test_static_friction_randomization(
    simple_model: mujoco.MjModel,
    simple_mjx_model: mjx.Model,
    rng: jax.Array,
) -> None:
    randomizer = ksim.StaticFrictionRandomizer(scale_lower=0.5, scale_upper=2.0)
    mj_dict = randomizer(simple_model, rng)
    mjx_dict = randomizer(simple_mjx_model, rng)
    assert np.allclose(mj_dict["dof_frictionloss"], mjx_dict["dof_frictionloss"])
