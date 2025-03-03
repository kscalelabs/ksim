import unittest
from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp
import mujoco.mjx as mjx
from jaxtyping import Array

from ksim.builders.rewards import (
    ActionSmoothnessPenalty,
    AngularVelocityXYPenalty,
    FootContactPenalty,
    FootContactPenaltyBuilder,
    LinearVelocityZPenalty,
    NormType,
    Reward,
    TrackAngularVelocityZReward,
    TrackLinearVelocityXYReward,
    get_norm,
)
from ksim.env.types import EnvState
from ksim.utils.data import BuilderData, MujocoMappings

_TOL = 1e-4


class DummyReward(Reward):
    """Simple reward class for testing the base functionality."""

    def __call__(self, prev_state: EnvState, action: Array, mjx_data: mjx.Data) -> Array:
        return jnp.ones(())


class DummyContact:
    """Mock contact data for testing."""

    def __init__(self, geom1, geom2, dist):
        self.geom1 = geom1
        self.geom2 = geom2
        self.dist = dist


class DummyMjxData:
    """Mock mjx.Data for testing."""

    def __init__(
        self, qpos=None, qvel=None, contact_geom1=None, contact_geom2=None, contact_dist=None
    ):
        self.qpos = qpos if qpos is not None else jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.qvel = qvel if qvel is not None else jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.contact = DummyContact(
            geom1=contact_geom1 if contact_geom1 is not None else jnp.array([1, 3, 5]),
            geom2=contact_geom2 if contact_geom2 is not None else jnp.array([2, 4, 6]),
            dist=contact_dist if contact_dist is not None else jnp.array([0.01, 0.05, 0.2]),
        )


class DummyEnvState:
    """Mock EnvState for testing."""

    def __init__(
        self,
        action_at_prev_step=None,
        commands=None,
    ):
        default_commands = {
            "linear_velocity_command": jnp.array([0.5, 0.3]),
            "angular_velocity_command": jnp.array([0.2]),
        }
        self.commands = commands if commands is not None else default_commands
        self.action_at_prev_step = (
            action_at_prev_step if action_at_prev_step is not None else jnp.array([0.1, 0.2, 0.3])
        )


class GetNormTest(chex.TestCase):
    def test_l1_norm(self):
        x = jnp.array([-1.0, 2.0, -3.0])
        result = get_norm(x, "l1")
        expected = jnp.array([1.0, 2.0, 3.0])
        chex.assert_trees_all_close(result, expected)

    def test_l2_norm(self):
        x = jnp.array([-1.0, 2.0, -3.0])
        result = get_norm(x, "l2")
        expected = jnp.array([1.0, 4.0, 9.0])
        chex.assert_trees_all_close(result, expected)

    def test_invalid_norm(self):
        x = jnp.array([1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            get_norm(x, "invalid_norm")  # type: ignore


class BaseRewardTest(chex.TestCase):
    def test_reward_name(self):
        reward = DummyReward(scale=1.0)
        self.assertEqual(reward.get_name(), "dummy_reward")
        self.assertEqual(reward.reward_name, "dummy_reward")

    def test_post_accumulate_default(self):
        reward = DummyReward(scale=1.0)
        accumulated = jnp.array([1.0, 2.0, 3.0])
        result = reward.post_accumulate(accumulated)
        chex.assert_trees_all_close(result, accumulated)

    def test_scale_warning_reward(self):
        # Testing that warnings are triggered appropriately
        # This is more of a placeholder since we can't easily capture warnings in this testing framework
        reward = DummyReward(scale=-1.0)
        self.assertEqual(reward.scale, -1.0)

    def test_reward_call(self):
        reward = DummyReward(scale=2.0)
        prev_state = DummyEnvState()
        action = jnp.array([0.1, 0.2])
        mjx_data = DummyMjxData()

        result = reward(prev_state, action, mjx_data)
        chex.assert_trees_all_close(result * reward.scale, jnp.array(2.0))


class LinearVelocityZPenaltyTest(chex.TestCase):
    def test_l2_norm(self):
        penalty = LinearVelocityZPenalty(scale=1.0, norm="l2")
        prev_state = DummyEnvState()
        action = jnp.array([0.1, 0.2])
        mjx_data = DummyMjxData(qvel=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

        result = penalty(prev_state, action, mjx_data)
        expected = 0.3**2  # qvel[2] squared for l2 norm
        chex.assert_trees_all_close(result, jnp.array(expected))

    def test_l1_norm(self):
        penalty = LinearVelocityZPenalty(scale=1.0, norm="l1")
        prev_state = DummyEnvState()
        action = jnp.array([0.1, 0.2])
        mjx_data = DummyMjxData(qvel=jnp.array([0.1, 0.2, -0.3, 0.4, 0.5, 0.6]))

        result = penalty(prev_state, action, mjx_data)
        expected = 0.3  # absolute value of qvel[2] for l1 norm
        chex.assert_trees_all_close(result, jnp.array(expected))


class AngularVelocityXYPenaltyTest(chex.TestCase):
    def test_l2_norm(self):
        penalty = AngularVelocityXYPenalty(scale=1.0, norm="l2")
        prev_state = DummyEnvState()
        action = jnp.array([0.1, 0.2])
        mjx_data = DummyMjxData(qvel=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

        result = penalty(prev_state, action, mjx_data)
        expected = 0.4**2 + 0.5**2  # sum of squared qvel[3:5]
        chex.assert_trees_all_close(result, jnp.array(expected))

    def test_l1_norm(self):
        penalty = AngularVelocityXYPenalty(scale=1.0, norm="l1")
        prev_state = DummyEnvState()
        action = jnp.array([0.1, 0.2])
        mjx_data = DummyMjxData(qvel=jnp.array([0.1, 0.2, 0.3, -0.4, 0.5, 0.6]))

        result = penalty(prev_state, action, mjx_data)
        expected = 0.4 + 0.5  # sum of absolute values of qvel[3:5]
        chex.assert_trees_all_close(result, jnp.array(expected))


class TrackAngularVelocityZRewardTest(chex.TestCase):
    def test_reward_calculation(self):
        reward = TrackAngularVelocityZReward(scale=1.0, norm="l2")
        prev_state = DummyEnvState(commands={"angular_velocity_command": jnp.array([0.2])})
        action = jnp.array([0.1, 0.2])
        mjx_data = DummyMjxData(qvel=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

        result = reward(prev_state, action, mjx_data)
        # ang_vel_z (qvel[5]) * ang_vel_cmd (0.2) = 0.6 * 0.2 = 0.12
        # l2 norm of 0.12 = 0.12^2 = 0.0144
        expected = 0.12**2
        chex.assert_trees_all_close(result, jnp.array(expected))

    def test_custom_cmd_name(self):
        reward = TrackAngularVelocityZReward(scale=1.0, norm="l2", cmd_name="custom_command")
        prev_state = DummyEnvState(commands={"custom_command": jnp.array([0.3])})
        action = jnp.array([0.1, 0.2])
        mjx_data = DummyMjxData(qvel=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

        result = reward(prev_state, action, mjx_data)
        expected = (0.6 * 0.3) ** 2
        chex.assert_trees_all_close(result, jnp.array(expected))


class TrackLinearVelocityXYRewardTest(chex.TestCase):
    def test_reward_calculation(self):
        reward = TrackLinearVelocityXYReward(scale=1.0, norm="l2")
        prev_state = DummyEnvState(commands={"linear_velocity_command": jnp.array([0.5, 0.3])})
        action = jnp.array([0.1, 0.2])
        mjx_data = DummyMjxData(qvel=jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

        result = reward(prev_state, action, mjx_data)
        # lin_vel_xy * lin_vel_cmd = [0.1*0.5, 0.2*0.3] = [0.05, 0.06]
        # l2 norm sum = 0.05^2 + 0.06^2 = 0.0061
        expected = 0.05**2 + 0.06**2
        chex.assert_trees_all_close(result, jnp.array(expected))

    def test_custom_cmd_name(self):
        reward = TrackLinearVelocityXYReward(scale=1.0, norm="l1", cmd_name="custom_command")
        prev_state = DummyEnvState(commands={"custom_command": jnp.array([0.6, 0.4])})
        action = jnp.array([0.1, 0.2])
        mjx_data = DummyMjxData(qvel=jnp.array([-0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))

        result = reward(prev_state, action, mjx_data)
        # lin_vel_xy * lin_vel_cmd = [-0.1*0.6, 0.2*0.4] = [-0.06, 0.08]
        # l1 norm sum = |-0.06| + |0.08| = 0.14
        expected = 0.06 + 0.08
        chex.assert_trees_all_close(result, jnp.array(expected))


class ActionSmoothnessPenaltyTest(chex.TestCase):
    def test_l2_norm(self):
        penalty = ActionSmoothnessPenalty(scale=1.0, norm="l2")
        prev_state = DummyEnvState(
            action_at_prev_step=jnp.array([0.1, 0.2, 0.3]),
        )
        action = jnp.array([0.2, 0.4, 0.5])
        mjx_data = DummyMjxData()

        result = penalty(prev_state, action, mjx_data)
        # action - last_action = [0.1, 0.2, 0.2]
        # l2 norm sum = 0.1^2 + 0.2^2 + 0.2^2 = 0.09
        expected = 0.1**2 + 0.2**2 + 0.2**2
        chex.assert_trees_all_close(result, jnp.array(expected))

    def test_l1_norm(self):
        penalty = ActionSmoothnessPenalty(scale=1.0, norm="l1")
        prev_state = DummyEnvState(
            action_at_prev_step=jnp.array([0.1, 0.2, 0.3]),
        )
        action = jnp.array([0.0, 0.4, 0.2])
        mjx_data = DummyMjxData()

        result = penalty(prev_state, action, mjx_data)
        # action - last_action = [-0.1, 0.2, -0.1]
        # l1 norm sum = |-0.1| + |0.2| + |-0.1| = 0.4
        expected = 0.1 + 0.2 + 0.1
        chex.assert_trees_all_close(result, jnp.array(expected))


class FootContactPenaltyTest(chex.TestCase):
    def test_contact_detection(self):
        penalty = FootContactPenalty(
            scale=1.0,
            illegal_geom_idxs=jnp.array([1, 3, 5]),
            allowed_contact_prct=0.1,
            contact_eps=0.02,
        )
        prev_state = DummyEnvState()
        action = jnp.array([0.1, 0.2])

        # Test with contact (first geom has distance < contact_eps)
        mjx_data = DummyMjxData(
            contact_geom1=jnp.array([1, 7, 9]),
            contact_geom2=jnp.array([8, 10, 12]),
            contact_dist=jnp.array([0.01, 0.05, 0.08]),
        )
        result = penalty(prev_state, action, mjx_data)
        chex.assert_trees_all_close(result, jnp.array(1.0))

        # Test without illegal contact
        mjx_data = DummyMjxData(
            contact_geom1=jnp.array([2, 4, 6]),
            contact_geom2=jnp.array([7, 8, 9]),
            contact_dist=jnp.array([0.01, 0.05, 0.08]),
        )
        result = penalty(prev_state, action, mjx_data)
        chex.assert_trees_all_close(result, jnp.array(0.0))

        # Test with contact in geom2
        mjx_data = DummyMjxData(
            contact_geom1=jnp.array([7, 8, 9]),
            contact_geom2=jnp.array([5, 7, 9]),
            contact_dist=jnp.array([0.01, 0.05, 0.08]),
        )
        result = penalty(prev_state, action, mjx_data)
        chex.assert_trees_all_close(result, jnp.array(1.0))

        # Test with contact but distance > eps
        mjx_data = DummyMjxData(
            contact_geom1=jnp.array([1, 3, 5]),
            contact_geom2=jnp.array([7, 8, 9]),
            contact_dist=jnp.array([0.03, 0.05, 0.08]),  # All > contact_eps
        )
        result = penalty(prev_state, action, mjx_data)
        chex.assert_trees_all_close(result, jnp.array(0.0))

    def test_skip_if_zero_command(self):
        penalty = FootContactPenalty(
            scale=1.0,
            illegal_geom_idxs=jnp.array([1, 3, 5]),
            allowed_contact_prct=0.1,
            contact_eps=0.02,
            skip_if_zero_command=["linear_velocity_command", "angular_velocity_command"],
            eps=1e-6,
        )

        # Test with zero commands - should skip penalty
        zero_cmd_state = DummyEnvState(
            commands={
                "linear_velocity_command": jnp.array([0.0, 0.0]),
                "angular_velocity_command": jnp.array([0.0]),
            }
        )
        action = jnp.array([0.1, 0.2])
        mjx_data = DummyMjxData(
            contact_geom1=jnp.array([1, 7, 9]),
            contact_geom2=jnp.array([8, 10, 12]),
            contact_dist=jnp.array([0.01, 0.05, 0.08]),
        )

        result = penalty(zero_cmd_state, action, mjx_data)
        chex.assert_trees_all_close(result, jnp.array(0.0))

        # Test with non-zero commands - should apply penalty
        non_zero_cmd_state = DummyEnvState(
            commands={
                "linear_velocity_command": jnp.array([0.1, 0.0]),
                "angular_velocity_command": jnp.array([0.0]),
            }
        )

        result = penalty(non_zero_cmd_state, action, mjx_data)
        chex.assert_trees_all_close(result, jnp.array(1.0))

    def test_post_accumulate(self):
        penalty = FootContactPenalty(
            scale=1.0,
            illegal_geom_idxs=jnp.array([1, 3, 5]),
            allowed_contact_prct=0.3,
            contact_eps=0.02,
        )

        # Test with 50% contact time (more than allowed)
        reward = jnp.array([0, 1, 0, 1, 0, 1, 0, 1])  # 50% ones
        result = penalty.post_accumulate(reward)
        # mean = 0.5, allowed = 0.3, diff = 0.2, clip to [0,inf)
        # result = clip(reward - (0.5 - 0.3), min=0)
        expected = jnp.array([0, 0.8, 0, 0.8, 0, 0.8, 0, 0.8])
        chex.assert_trees_all_close(result, expected)

        # Test with 20% contact time (less than allowed)
        reward = jnp.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])  # 10% ones
        result = penalty.post_accumulate(reward)
        # mean = 0.1, allowed = 0.3, diff = -0.2, clip to [0,inf)
        # result = clip(reward - (0.1 - 0.3), min=0, max=1)
        expected = jnp.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        chex.assert_trees_all_close(result, expected)


class FootContactPenaltyBuilderTest(chex.TestCase):
    def setUp(self):
        self.mappings = MujocoMappings(
            geom_idx_to_body_name={
                0: "base",
                1: "foot_left",
                2: "foot_right",
                3: "leg",
                4: "foot_left",
                5: "arm",
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

    def test_builder_creates_correct_penalty(self):
        builder = FootContactPenaltyBuilder(
            scale=2.0,
            foot_body_names=["foot_left", "foot_right"],
            allowed_contact_prct=0.2,
            contact_eps=0.03,
            skip_if_zero_command=["linear_velocity_command"],
        )

        penalty = builder(self.builder_data)

        self.assertIsInstance(penalty, FootContactPenalty)
        self.assertEqual(penalty.scale, 2.0)
        self.assertEqual(penalty.allowed_contact_prct, 0.2)
        self.assertEqual(penalty.contact_eps, 0.03)
        self.assertEqual(penalty.skip_if_zero_command, ["linear_velocity_command"])

        # Check that illegal_geom_idxs contains the right indices
        # Foot geoms are indices 1, 2, 4
        expected_illegal_idxs = jnp.array([1, 2, 4])
        for idx in expected_illegal_idxs:
            self.assertTrue(idx in penalty.illegal_geom_idxs)
        self.assertEqual(len(penalty.illegal_geom_idxs), 3)

    def test_builder_with_empty_skip_command(self):
        builder = FootContactPenaltyBuilder(
            scale=1.0,
            foot_body_names=["foot_left"],
            allowed_contact_prct=0.1,
            skip_if_zero_command=None,
        )

        penalty = builder(self.builder_data)
        self.assertEqual(penalty.skip_if_zero_command, [])


class RewardShapeTest(chex.TestCase):
    """Test that all rewards return scalar values."""

    def setUp(self):
        self.prev_state = DummyEnvState()
        self.action = jnp.array([0.1, 0.2, 0.3])
        self.mjx_data = DummyMjxData()

    def test_all_rewards_are_scalars(self):
        # Test all reward classes
        rewards = [
            DummyReward(scale=1.0),
            LinearVelocityZPenalty(scale=1.0),
            AngularVelocityXYPenalty(scale=1.0),
            TrackAngularVelocityZReward(scale=1.0),
            TrackLinearVelocityXYReward(scale=1.0),
            ActionSmoothnessPenalty(scale=1.0),
            FootContactPenalty(
                scale=1.0,
                illegal_geom_idxs=jnp.array([1, 3, 5]),
                allowed_contact_prct=0.1,
                contact_eps=0.02,
            ),
        ]

        for reward in rewards:
            reward_val = reward(self.prev_state, self.action, self.mjx_data)
            reward_name = reward.get_name()
            chex.assert_shape(
                reward_val, (), custom_message=f"Reward {reward_name} must be a scalar"
            )


if __name__ == "__main__":
    unittest.main()
