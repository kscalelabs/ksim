"""Tests for rewards in the ksim package."""

import jax.numpy as jnp
import xax

import ksim


def create_test_trajectory(
    actions: jnp.ndarray,
    done: jnp.ndarray | None = None,
    timesteps: int | None = None,
    action_dims: int = 3,
) -> ksim.Trajectory:
    """Create a test trajectory with given actions.

    Args:
        actions: Action array of shape (timesteps, action_dims)
        done: Optional done mask of shape (timesteps,). If None, all False.
        timesteps: Number of timesteps. If None, inferred from actions.
        action_dims: Number of action dimensions.

    Returns:
        A ksim.Trajectory object for testing.
    """
    if timesteps is None:
        timesteps = actions.shape[0]

    if done is None:
        done = jnp.zeros(timesteps, dtype=bool)

    dummy_qpos = jnp.zeros((timesteps, 10))
    dummy_qvel = jnp.zeros((timesteps, 9))
    dummy_xpos = jnp.zeros((timesteps, 5, 3))
    dummy_xquat = jnp.zeros((timesteps, 5, 4))
    dummy_ctrl = jnp.zeros((timesteps, action_dims))
    dummy_timestep = jnp.arange(timesteps, dtype=jnp.float32)
    dummy_success = jnp.zeros(timesteps, dtype=bool)

    return ksim.Trajectory(
        qpos=dummy_qpos,
        qvel=dummy_qvel,
        xpos=dummy_xpos,
        xquat=dummy_xquat,
        ctrl=dummy_ctrl,
        obs=xax.FrozenDict({}),
        command=xax.FrozenDict({}),
        event_state=xax.FrozenDict({}),
        action=actions,
        done=done,
        success=dummy_success,
        timestep=dummy_timestep,
        termination_components=xax.FrozenDict({}),
        aux_outputs=xax.FrozenDict({}),
    )


class TestActionVelocityPenalty:
    """Test suite for ActionVelocityPenalty."""

    def test_constant_action_zero_penalty(self) -> None:
        """Test that constant actions result in zero velocity penalty."""
        penalty = ksim.ActionVelocityPenalty(scale=-1.0)

        constant_actions = jnp.ones((5, 3)) * 2.0
        trajectory = create_test_trajectory(constant_actions)

        reward_per_timestep = penalty.get_reward(trajectory)

        expected_per_timestep = jnp.zeros(5)
        assert jnp.allclose(reward_per_timestep, expected_per_timestep, atol=1e-6)

    def test_linear_action_change_constant_velocity(self) -> None:
        """Test linear action changes result in constant velocity penalty."""
        penalty = ksim.ActionVelocityPenalty(scale=-1.0, norm="l2")

        # Linear increase: actions go from [0,0,0] to [4,4,4] over 5 steps
        actions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ]
        )
        trajectory = create_test_trajectory(actions)

        reward_per_timestep = penalty.get_reward(trajectory)

        # With masking, first timestep is 0, rest have constant velocity penalty
        # Velocities from timestep 1 onwards: [1,1,1], [1,1,1], [1,1,1], [1,1,1]
        # L2 element-wise squares: [1,1,1], [1,1,1], [1,1,1], [1,1,1]
        # Mean over dimensions: 1, 1, 1, 1
        expected_per_timestep = jnp.array([0.0, 1.0, 1.0, 1.0, 1.0])
        assert jnp.allclose(reward_per_timestep, expected_per_timestep, atol=1e-6)

    def test_step_action_change(self) -> None:
        """Test step change in actions."""
        penalty = ksim.ActionVelocityPenalty(scale=-1.0, norm="l1")

        # Step change: constant then sudden jump
        actions = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 2.0, -1.0],
                [3.0, 2.0, -1.0],
            ]
        )
        trajectory = create_test_trajectory(actions)

        reward_per_timestep = penalty.get_reward(trajectory)

        # With masking, first timestep is 0
        # Velocity changes from timestep 1: [0,0,0], [0,0,0], [2,2,-1], [0,0,0]
        # L1 element-wise absolute values: [0,0,0], [0,0,0], [2,2,1], [0,0,0]
        # Mean over dimensions: 0, 0, 5/3, 0
        expected_per_timestep = jnp.array([0.0, 0.0, 0.0, 5.0 / 3.0, 0.0])
        assert jnp.allclose(reward_per_timestep, expected_per_timestep, atol=1e-6)

    def test_done_mask_resets_velocity(self) -> None:
        """Test that done episodes reset action velocity calculations."""
        penalty = ksim.ActionVelocityPenalty(scale=-1.0, norm="l2")

        actions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ]
        )
        done = jnp.array([False, True, False, False])
        trajectory = create_test_trajectory(actions, done=done)

        reward_per_timestep = penalty.get_reward(trajectory)

        # With masking, first timestep is 0
        # Velocities from timestep 1: [2,0,0], [0,0,0] (reset), [1,0,0]
        # L2 element-wise squares: [4,0,0], [0,0,0], [1,0,0]
        # Mean over dimensions: 4/3, 0, 1/3
        expected_per_timestep = jnp.array([0.0, 4.0 / 3.0, 0.0, 1.0 / 3.0])
        assert jnp.allclose(reward_per_timestep, expected_per_timestep, atol=1e-6)


class TestActionAccelerationPenalty:
    """Test suite for ActionAccelerationPenalty."""

    def test_constant_velocity_zero_acceleration(self) -> None:
        """Test that constant velocity (linear actions) gives zero acceleration penalty."""
        penalty = ksim.ActionAccelerationPenalty(scale=-1.0)

        # Linear actions: constant velocity
        actions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0],
            ]
        )
        trajectory = create_test_trajectory(actions)

        reward_per_timestep = penalty.get_reward(trajectory)

        # With masking, first two timesteps are 0
        # True constant velocity should give zero acceleration
        expected_per_timestep = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        assert jnp.allclose(reward_per_timestep, expected_per_timestep, atol=1e-6)

    def test_quadratic_action_constant_acceleration(self) -> None:
        """Test quadratic action progression gives constant acceleration."""
        penalty = ksim.ActionAccelerationPenalty(scale=-1.0, norm="l2")

        # Quadratic progression: 0, 1, 4, 9, 16 (squared values)
        actions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [4.0, 4.0, 4.0],
                [9.0, 9.0, 9.0],
                [16.0, 16.0, 16.0],
            ]
        )
        trajectory = create_test_trajectory(actions)

        reward_per_timestep = penalty.get_reward(trajectory)

        # With masking, first two timesteps are 0
        # True quadratic progression: velocities [1,3,5,7], accelerations [2,2,2]
        # L2 element-wise squares: [4,4,4] for timesteps 2,3,4
        # Mean over dimensions: 4, 4, 4
        expected_per_timestep = jnp.array([0.0, 0.0, 4.0, 4.0, 4.0])
        assert jnp.allclose(reward_per_timestep, expected_per_timestep, atol=1e-6)

    def test_acceleration_step_change(self) -> None:
        """Test step change in acceleration."""
        penalty = ksim.ActionAccelerationPenalty(scale=-1.0, norm="l1")

        # Velocities: constant, then linear increase
        actions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # vel = [1,0,0]
                [2.0, 0.0, 0.0],  # vel = [1,0,0]
                [2.0, 0.0, 0.0],  # vel = [0,0,0]
                [1.0, 0.0, 0.0],  # vel = [-1,0,0]
            ]
        )
        trajectory = create_test_trajectory(actions)

        reward_per_timestep = penalty.get_reward(trajectory)

        # With masking, first two timesteps are 0
        # True velocities: [1,1,0,-1], accelerations: [0,-1,-1]
        # L1 element-wise absolute values: [0,1,1] for timesteps 2,3,4
        # Mean over dimensions: 0, 1/3, 1/3
        expected_per_timestep = jnp.array([0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0])
        assert jnp.allclose(reward_per_timestep, expected_per_timestep, atol=1e-6)


class TestActionJerkPenalty:
    """Test suite for ActionJerkPenalty."""

    def test_constant_acceleration_zero_jerk(self) -> None:
        """Test that constant acceleration gives zero jerk penalty."""
        penalty = ksim.ActionJerkPenalty(scale=-1.0)

        # Quadratic actions: constant acceleration
        actions = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [4.0, 4.0, 4.0],
                [9.0, 9.0, 9.0],
                [16.0, 16.0, 16.0],
                [25.0, 25.0, 25.0],
            ]
        )
        trajectory = create_test_trajectory(actions)

        reward_per_timestep = penalty.get_reward(trajectory)

        # With masking, first three timesteps are 0
        # True quadratic progression should give zero jerk
        expected_per_timestep = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        assert jnp.allclose(reward_per_timestep, expected_per_timestep, atol=1e-6)

    def test_cubic_action_constant_jerk(self) -> None:
        """Test cubic action progression gives constant jerk."""
        penalty = ksim.ActionJerkPenalty(scale=-1.0, norm="l2")

        # Cubic progression: t^3 for t = 0,1,2,3,4,5
        t = jnp.arange(6)
        cubic_values = t**3
        actions = jnp.stack([cubic_values, cubic_values, cubic_values], axis=1)
        trajectory = create_test_trajectory(actions)

        reward_per_timestep = penalty.get_reward(trajectory)

        # With masking, first three timesteps are 0
        # True cubic progression: velocities [1,7,19,37,61], accelerations [6,12,18,24], jerks [6,6,6]
        # L2 element-wise squares: [36,36,36] for timesteps 3,4,5
        # Mean over dimensions: 36, 36, 36
        expected_per_timestep = jnp.array([0.0, 0.0, 0.0, 36.0, 36.0, 36.0])
        assert jnp.allclose(reward_per_timestep, expected_per_timestep, atol=1e-5)
