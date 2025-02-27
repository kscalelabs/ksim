import unittest
from typing import Any

import chex
import jax.numpy as jnp
from jaxtyping import Array

from ksim.builders.terminations import (
    IllegalContactTermination,
    IllegalContactTerminationBuilder,
    MinimumHeightTermination,
    PitchTooGreatTermination,
    RollTooGreatTermination,
    Termination,
)
from ksim.utils.data import BuilderData, MujocoMappings


class DummyTermination(Termination):
    """Dummy termination class for testing base functionality."""

    def __call__(self, state: Any) -> Array:
        return jnp.zeros((), dtype=bool)


class DummyMjxData:
    """Mock mjx.Data for testing."""

    def __init__(self, has_contact=True):
        self._has_contact = has_contact

    @property
    def qpos(self) -> Array:
        return jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

    @property
    def ncon(self) -> int:
        return 2 if self._has_contact else 0

    @property
    def contact(self):
        class Contact:
            geom1 = jnp.array([0, 2])
            geom2 = jnp.array([1, 3])
            dist = jnp.array([-0.01, -0.02])

        return Contact()


class BaseTerminationTest(chex.TestCase):
    def test_termination_name(self):
        term = DummyTermination()
        self.assertEqual(term.get_name(), "dummy_termination")
        self.assertEqual(term.termination_name, "dummy_termination")


class TerminationTest(chex.TestCase):
    def setUp(self):
        self.data = DummyMjxData()

    def test_pitch_too_great_termination(self):
        # With a small max_pitch, should terminate
        term = PitchTooGreatTermination(max_pitch=0.1)
        result = term(self.data)  # type: ignore
        self.assertTrue(jnp.all(result))

        # With a large max_pitch, should not terminate
        term = PitchTooGreatTermination(max_pitch=10.0)
        result = term(self.data)  # type: ignore
        self.assertFalse(jnp.all(result))

    def test_roll_too_great_termination(self):
        # With a small max_roll, should terminate
        term = RollTooGreatTermination(max_roll=0.1)
        result = term(self.data)  # type: ignore
        self.assertTrue(jnp.all(result))

        # With a large max_roll, should not terminate
        term = RollTooGreatTermination(max_roll=10.0)
        result = term(self.data)  # type: ignore
        self.assertFalse(jnp.all(result))

    def test_minimum_height_termination(self):
        # With min_height above the current height, should terminate
        term = MinimumHeightTermination(min_height=4.0)
        result = term(self.data)  # type: ignore
        self.assertTrue(jnp.all(result))

        # With min_height below the current height, should not terminate
        term = MinimumHeightTermination(min_height=2.0)
        result = term(self.data)  # type: ignore
        self.assertFalse(jnp.all(result))

    def test_illegal_contact_termination(self):
        # Test with illegal contact present
        term = IllegalContactTermination(illegal_geom_idxs=jnp.array([0, 3]))
        result = term(self.data)  # type: ignore
        self.assertTrue(jnp.all(result))

        # Test with no illegal contacts
        term = IllegalContactTermination(illegal_geom_idxs=jnp.array([4, 5]))
        result = term(self.data)  # type: ignore
        self.assertFalse(jnp.all(result))

        # Test with no contacts at all
        data_no_contact = DummyMjxData(has_contact=False)
        term = IllegalContactTermination(illegal_geom_idxs=jnp.array([0, 1]))
        result = term(data_no_contact)  # type: ignore
        self.assertFalse(jnp.all(result))


class IllegalContactTerminationBuilderTest(chex.TestCase):
    def setUp(self):
        self.mappings = MujocoMappings(
            geom_idx_to_body_name={
                0: "torso",
                1: "left_leg",
                2: "right_leg",
                3: "head",
            },
            sensor_name_to_idx_range={},
            qpos_name_to_idx_range={},
            qvelacc_name_to_idx_range={},
            ctrl_name_to_idx={},
        )
        self.builder_data = BuilderData(
            model=None,
            dt=0.004,
            ctrl_dt=0.02,
            mujoco_mappings=self.mappings,
        )

    def test_builder_creates_correct_termination(self):
        builder = IllegalContactTerminationBuilder(
            body_names=["torso", "head"],
            contact_eps=-0.005,
        )
        term = builder(self.builder_data)

        self.assertIsInstance(term, IllegalContactTermination)
        self.assertEqual(term.contact_eps, -0.005)
        self.assertTrue(jnp.array_equal(term.illegal_geom_idxs, jnp.array([0, 3])))


class TerminationShapeTest(chex.TestCase):
    """Test that all terminations return boolean arrays of the correct shape."""

    def setUp(self):
        self.mjx_data = DummyMjxData()
        self.mjx_data_no_contact = DummyMjxData(has_contact=False)

    def test_all_terminations_are_boolean_scalars(self):
        # Test all termination classes
        terminations = [
            DummyTermination(),
            PitchTooGreatTermination(max_pitch=0.5),
            RollTooGreatTermination(max_roll=0.5),
            MinimumHeightTermination(min_height=1.0),
            IllegalContactTermination(illegal_geom_idxs=jnp.array([0, 3])),
        ]

        for termination in terminations:
            term_result = termination(self.mjx_data)
            term_name = termination.get_name()
            chex.assert_shape(
                term_result,
                (),
                custom_message=f"Termination {term_name} must return shape ()",
            )
            self.assertEqual(
                term_result.dtype,
                jnp.bool_,
                f"Termination {term_name} must return boolean dtype",
            )

    def test_illegal_contact_termination_shape_with_no_contacts(self):
        """Test that IllegalContactTermination returns correct shape when ncon=0."""
        termination = IllegalContactTermination(illegal_geom_idxs=jnp.array([0, 3]))
        term_result = termination(self.mjx_data_no_contact)
        chex.assert_shape(
            term_result,
            (),
            custom_message="IllegalContactTermination must return shape () when ncon=0",
        )
        self.assertEqual(
            term_result.dtype,
            jnp.bool_,
            "IllegalContactTermination must return boolean dtype when ncon=0",
        )

    def test_termination_shape_with_batched_state(self):
        """Test terminations with mock batched input states."""

        class BatchedMjxData:
            def __init__(self):
                self.qpos = jnp.ones((2, 9))  # Batched qpos
                self.ncon = jnp.array([2, 0])  # Different ncon per batch

                class BatchedContact:
                    geom1 = jnp.array([[0, 2], [0, 0]])
                    geom2 = jnp.array([[1, 3], [0, 0]])
                    dist = jnp.array([[-0.01, -0.02], [0.0, 0.0]])

                self.contact = BatchedContact()

        batched_data = BatchedMjxData()

        # A termination that correctly handles batched input should still return scalar
        termination = IllegalContactTermination(illegal_geom_idxs=jnp.array([0, 3]))
        term_result = termination(self.mjx_data)  # Non-batched input

        chex.assert_shape(term_result, ())
        self.assertEqual(term_result.dtype, jnp.bool_)

        # For completeness, test other terminations as well with non-batched input
        min_height_term = MinimumHeightTermination(min_height=1.0)
        term_result = min_height_term(self.mjx_data)
        chex.assert_shape(term_result, ())


if __name__ == "__main__":
    unittest.main()
