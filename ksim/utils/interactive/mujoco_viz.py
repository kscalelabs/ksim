"""Interactive visualizer for Mujoco environments.

This module provides interactive visualization capabilities for Mujoco-based environments,
supporting both regular Mujoco and MJX physics backends. It allows interactive
visualization of rewards and states during environment execution.
"""

import logging
import time

import attrs
import jax
import jax.numpy as jnp
import matplotlib
from jaxtyping import Array, PyTree

from ksim.env.mjx.mjx_env import MjxEnv
from ksim.task.rl import RLTask
from ksim.utils.interactive.base import InteractiveVisualizer, InteractiveVisualizerConfig

matplotlib.use("Agg")  # Use non-interactive backend

import matplotlib.lines
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx, viewer as mujoco_viewer
from mujoco.viewer import Handle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@attrs.define(frozen=True, kw_only=True)
class MujocoInteractiveVisualizerConfig(InteractiveVisualizerConfig):
    """Configuration for the Mujoco interactive visualizer."""

    suspended_pos: Array = attrs.field(
        factory=lambda: jnp.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    )
    suspended_vel: Array = attrs.field(factory=lambda: jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    physics_backend: str = attrs.field(default="mjx")


class MujocoInteractiveVisualizer(InteractiveVisualizer):
    """Visualizer for Mujoco-based environments supporting both Mujoco and MJX backends."""

    viz_config: MujocoInteractiveVisualizerConfig

    def __init__(
        self, task: RLTask, config: MujocoInteractiveVisualizerConfig | None = None
    ) -> None:
        """Initialize the Mujoco interactive visualizer.

        Args:
            task: The RL task to visualize
            config: Configuration for the visualizer
        """
        if config is None:
            config = MujocoInteractiveVisualizerConfig()

        super().__init__(task, config)
        if config.physics_backend != "mjx":
            # Disable JIT for regular MuJoCo stepping
            jax.config.update("jax_disable_jit", True)
        logger.info("Using physics backend: %s", config.physics_backend)

        self.env = self.setup_environment()

        # Initialize Mujoco-specific variables
        self.model = self.env.default_mj_model
        self.data = mujoco.MjData(self.model)  # Separate data from env
        self.command = self.env.get_initial_commands(jax.random.PRNGKey(0), None)
        self.dummy_action = jnp.zeros(self.env.action_size)

    def setup_environment(self) -> MjxEnv:
        """Set up the MjxEnv environment for visualization."""
        env = self.task.get_environment()
        assert isinstance(env, MjxEnv), "MujocoRewardVisualizer only works with MjxEnv"
        return env

    def _step(
        self,
        mx: mjx.Model | None,
        state: mjx.Data | mujoco.MjData,
        step_fn: PyTree | None,
        viewer: Handle,
    ) -> mjx.Data | mujoco.MjData:
        """Unified physics step function that returns a state object.

        This function uses either Mujoco or MJX physics step depending on the config.

        Args:
            mx: The MJX model (or None for mujoco backend)
            state: The current state (MJX data object for mjx, self.data for mujoco)
            step_fn: The compiled physics step function (for mjx; None for mujoco)
            viewer: The viewer instance

        Returns:
            The updated state (MJX data for mjx, or self.data for mujoco)
        """
        if self.viz_config.physics_backend == "mjx":
            # For MJX backend, step_fn must be callable
            assert step_fn is not None, "step_fn cannot be None when using mjx backend"

            # Update state from self.data and step using MJX.
            state = state.replace(
                qpos=jnp.array(self.data.qpos),
                qvel=jnp.array(self.data.qvel),
                time=jnp.array(self.data.time),
                ctrl=jnp.array(self.data.ctrl),
                act=jnp.array(self.data.act),
                xfrc_applied=jnp.array(self.data.xfrc_applied),
            )
            state = step_fn(mx, state)
            mjx.get_data_into(self.data, self.model, state)
            viewer.sync()
            return state
        else:
            # For regular MuJoCo stepping, simply call mj_step and return self.data.
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
            return self.data

    def run(self) -> None:
        """Run the Mujoco visualization loop.

        This method starts the visualization loop that:
        1. Handles keyboard input for controlling visualization
        2. Steps the physics simulation
        3. Updates rewards and plots in real-time
        4. Manages suspended state and paused state

        The loop continues until the viewer is closed or an error occurs.
        """
        # For static typing, we need to assert that keycodes are set.
        msg = "All keycodes must be set"
        assert hasattr(self, "key_up"), msg
        assert hasattr(self, "key_down"), msg
        assert hasattr(self, "key_right"), msg
        assert hasattr(self, "key_left"), msg
        assert hasattr(self, "key_p"), msg
        assert hasattr(self, "key_l"), msg
        assert hasattr(self, "key_n"), msg
        assert hasattr(self, "key_r"), msg

        assert isinstance(self.viz_config, MujocoInteractiveVisualizerConfig)

        # Set up MJX objects only if using the MJX backend.
        if self.viz_config.physics_backend == "mjx":
            mx = mjx.put_model(self.model)
            state = mjx.put_data(self.model, self.data)
            # Compile the physics step function with JIT.
            step_fn = mjx.step
            logger.info("JIT-compiling the physics step function...")
            start_compile = time.time()
            step_fn = jax.jit(step_fn).lower(mx, state).compile()
            compile_time = time.time() - start_compile
            logger.info("Compilation took %.4fs", compile_time)
        else:
            mx = None
            # Use self.data directly for the mujoco backend.
            state = self.data
            step_fn = None

        # Create a viewer.
        with mujoco_viewer.launch_passive(
            model=self.model, data=self.data, key_callback=self.key_callback
        ) as viewer:
            # Initialize the "previous state" appropriately.
            state_t_minus_1 = state
            target_time = time.time()
            try:
                while viewer.is_running():
                    # Process key inputs.
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

                            # Call the unified physics step function.
                            state = self._step(mx, state, step_fn, viewer)

                        self.data_modifying_keycode = None

                    # When paused and if rewards are updated during pause.
                    if self.paused and self.viz_config.reward_when_paused:
                        with viewer.lock():
                            rewards = self.env.get_rewards(
                                self.dummy_action,
                                state_t_minus_1,
                                self.command,
                                self.dummy_action,
                                state,
                            )

                        self.timestamps.append(time.time() - self.start_time)
                        for reward_name, value in rewards.items():
                            self.reward_history[reward_name].append(float(value))
                        state_t_minus_1 = state

                    if not self.paused:
                        state = self._step(mx, state, step_fn, viewer)
                        if self.suspended:
                            self.data.qpos[:7] = self.viz_config.suspended_pos
                            self.data.qvel[:6] = self.viz_config.suspended_vel

                        with viewer.lock():
                            rewards = self.env.get_rewards(
                                self.dummy_action,
                                state_t_minus_1,
                                self.command,
                                self.dummy_action,
                                state,
                            )

                        current_time = time.time() - self.start_time
                        self.timestamps.append(current_time)
                        for reward_name, value in rewards.items():
                            self.reward_history[reward_name].append(float(value))

                        state_t_minus_1 = state

                    viewer.sync()
                    target_time += self.model.opt.timestep
                    current_time = time.time()
                    if target_time > current_time:
                        time.sleep(target_time - current_time)

            finally:
                self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources used by the visualizer."""
        self.is_running = False
        self.plot_thread.join(timeout=1.0)
        plt.close(self.fig)
        logger.info("Visualization terminated")
