"""Reward visualizer for Mujoco environments."""

import logging
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import matplotlib

from ksim.env.mjx.mjx_env import MjxEnv
from ksim.task.rl import RLTask
from ksim.utils.reward_visualization.base import RewardVisualizer, RewardVisualizerConfig

matplotlib.use("Agg")  # Use non-interactive backend

import matplotlib.lines
import matplotlib.pyplot as plt
import mujoco

jax.config.update("jax_disable_jit", True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MujocoRewardVisualizerConfig(RewardVisualizerConfig):
    suspended_pos: jnp.ndarray = field(
        # x_pos, y_pos, z_pos, w_quat, x_quat, y_quat, z_quat
        default_factory=lambda: jnp.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    )
    suspended_vel: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    )


class MujocoRewardVisualizer(RewardVisualizer):
    def __init__(self, task: RLTask, config: MujocoRewardVisualizerConfig | None = None) -> None:
        """Initialize the Mujoco reward visualizer.

        Args:
            task: The RL task to visualize
            config: Configuration for the visualizer
        """
        if config is None:
            config = MujocoRewardVisualizerConfig()

        super().__init__(task, config)

        self.env = self.setup_environment()

        # Initialize Mujoco-specific variables
        self.model = self.env.default_mj_model
        self.data = mujoco.MjData(self.model)  # Make separate data from env
        self.command = self.env.get_initial_commands(jax.random.PRNGKey(0), None)
        self.dummy_action = jnp.zeros(self.env.action_size)

    def setup_environment(self) -> MjxEnv:
        """Set up the MjxEnv environment for visualization.

        Returns:
            The MjxEnv environment instance
        """
        env = self.task.get_environment()
        assert isinstance(env, MjxEnv), "MujocoRewardVisualizer only works with MjxEnv"
        return env

    def run(self) -> None:
        """Run the Mujoco visualization loop."""
        # For typing, strictly assert that all keycodes are set
        msg = "All keycodes must be set"
        assert hasattr(self, "key_up"), msg
        assert hasattr(self, "key_down"), msg
        assert hasattr(self, "key_right"), msg
        assert hasattr(self, "key_left"), msg
        assert hasattr(self, "key_p"), msg
        assert hasattr(self, "key_l"), msg
        assert hasattr(self, "key_n"), msg
        assert hasattr(self, "key_r"), msg

        assert isinstance(self.viz_config, MujocoRewardVisualizerConfig)

        # Create a viewer
        with mujoco.viewer.launch_passive(
            model=self.model, data=self.data, key_callback=self.key_callback
        ) as viewer:
            data_last = self.data

            try:
                while viewer.is_running():
                    # Process key inputs
                    if self.data_modifying_keycode is not None:
                        with viewer.lock():
                            match self.data_modifying_keycode:
                                case self.key_up:
                                    self.data.qpos[0] += self.viz_config.pos_step_size
                                    logger.info(
                                        "Moving x position by %.3f", self.viz_config.pos_step_size
                                    )
                                case self.key_down:
                                    self.data.qpos[0] -= self.viz_config.pos_step_size
                                    logger.info(
                                        "Moving x position by -%.3f", self.viz_config.pos_step_size
                                    )
                                case self.key_right:
                                    self.data.qpos[1] += self.viz_config.pos_step_size
                                    logger.info(
                                        "Moving y position by %.3f", self.viz_config.pos_step_size
                                    )
                                case self.key_left:
                                    self.data.qpos[1] -= self.viz_config.pos_step_size
                                    logger.info(
                                        "Moving y position by -%.3f", self.viz_config.pos_step_size
                                    )
                                case self.key_p:
                                    self.data.qpos[2] += self.viz_config.pos_step_size
                                    logger.info(
                                        "Moving z position by %.3f", self.viz_config.pos_step_size
                                    )
                                case self.key_l:
                                    self.data.qpos[2] -= self.viz_config.pos_step_size
                                    logger.info(
                                        "Moving z position by -%.3f", self.viz_config.pos_step_size
                                    )
                                case self.key_n:
                                    logger.info("Stepping forward by one frame")
                                case self.key_r:
                                    logger.info("Resetting joints and orientation to initial state")
                                    self.data.qpos[3:] = self.env.default_mj_data.qpos[3:]
                                    self.data.qvel[6:] = self.env.default_mj_data.qvel[6:]
                                    self.data.qacc[6:] = self.env.default_mj_data.qacc[6:]
                                    self.data.ctrl[:] = self.env.default_mj_data.ctrl
                                case _:
                                    logger.warning(
                                        "Unknown keycode: %d", self.data_modifying_keycode
                                    )
                            mujoco.mj_step(self.model, self.data)

                        self.data_modifying_keycode = None

                    if self.paused and self.viz_config.reward_when_paused:
                        with viewer.lock():
                            rewards = self.env.get_rewards(
                                self.dummy_action,
                                data_last,
                                self.command,
                                self.dummy_action,
                                self.data,
                            )

                        self.timestamps.append(time.time() - self.start_time)

                        for reward_name, value in rewards.items():
                            self.reward_history[reward_name].append(float(value))

                        data_last = self.data

                    if not self.paused:
                        mujoco.mj_step(self.model, self.data)

                        if self.suspended:
                            self.data.qpos[:7] = self.viz_config.suspended_pos
                            self.data.qvel[:6] = self.viz_config.suspended_vel

                        with viewer.lock():
                            rewards = self.env.get_rewards(
                                self.dummy_action,
                                data_last,
                                self.command,
                                self.dummy_action,
                                self.data,
                            )

                        # Store the current time
                        current_time = time.time() - self.start_time
                        self.timestamps.append(current_time)

                        # Store each reward component
                        for reward_name, value in rewards.items():
                            self.reward_history[reward_name].append(float(value))

                        data_last = self.data

                    # Update the viewer
                    viewer.sync()
                    time.sleep(0.0001)
            finally:
                self.cleanup()

    def cleanup(self) -> None:
        # Clean up
        self.is_running = False
        self.plot_thread.join(timeout=1.0)
        plt.close(self.fig)
        logger.info("Visualization terminated")
