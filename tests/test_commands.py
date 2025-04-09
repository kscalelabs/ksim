"""Tests for command builders in the ksim package."""

import attrs
import chex
import jax
import jax.numpy as jnp
import mujoco
import pytest
from jaxtyping import Array

import ksim

_TOL = 1e-4


@attrs.define(frozen=True)
class DummyCommand(ksim.Command):
    def initial_command(self, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: jax.Array) -> jnp.ndarray:
        return jnp.zeros((1,))

    def __call__(
        self, prev_command: Array, physics_data: ksim.PhysicsData, curriculum_level: Array, rng: jax.Array
    ) -> jnp.ndarray:
        return jnp.zeros((1,))


def get_simple_model() -> mujoco.MjModel:
    """Get a simple model for testing."""
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
    return mujoco.MjData(mj_model)


class TestVectorCommand:
    """Tests for the VectorCommand class."""

    @pytest.fixture
    def rng(self) -> jax.Array:
        """Return a random number generator key."""
        return jax.random.PRNGKey(0)

    @pytest.fixture
    def physics_data(self) -> ksim.PhysicsData:
        """Get a simple model for testing."""
        return get_simple_model()

    def test_command_shape(self, rng: jax.Array, physics_data: ksim.PhysicsData) -> None:
        """Test that the command returns the correct shape."""
        cmd = ksim.FloatVectorCommand(ranges=((0.0, 1.0), (0.0, 1.0)))
        curriculum_level = jnp.array(0.0)
        initial_command = cmd.initial_command(physics_data, curriculum_level, rng)
        result = cmd(initial_command, physics_data, curriculum_level, rng)
        chex.assert_shape(result, (2,))

    def test_command_bounds(self, rng: jax.Array, physics_data: ksim.PhysicsData) -> None:
        """Test that the command values are within the expected bounds."""
        scale = 2.0
        cmd = ksim.FloatVectorCommand(ranges=((0.0, scale), (0.0, scale)))
        curriculum_level = jnp.array(0.0)

        # Run multiple times to test bounds probabilistically
        for i in range(100):
            key = jax.random.fold_in(rng, i)
            command = cmd.initial_command(physics_data, curriculum_level, key)
            result = cmd(command, physics_data, curriculum_level, key)

            assert result[0] >= -scale
            assert result[0] <= scale

    def test_update_mechanism(self, rng: jax.Array, physics_data: ksim.PhysicsData) -> None:
        """Test that the command update mechanism works correctly."""
        cmd = ksim.FloatVectorCommand(ranges=((0.0, 1.0), (0.0, 1.0)))
        curriculum_level = jnp.array(0.0)
        command = cmd.initial_command(physics_data, curriculum_level, rng)

        next_command = cmd(command, physics_data, curriculum_level, rng)
        assert jnp.array_equal(next_command, command)
