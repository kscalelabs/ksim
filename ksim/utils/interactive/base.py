"""Base class for interactive visualizers."""

import abc
import collections
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

from ksim.env.base_env import BaseEnv
from ksim.task.rl import RLTask

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InteractiveVisualizerConfig:
    window_size: int = 5000
    fig_save_dir: Path = Path("/tmp/plots/")
    update_when_paused: bool = True
    pos_step_size: float = 0.01  # Size of step when manually modifying position (meters)
    vel_step_size: float = 0.1  # Size of step when manually modifying velocity (meters/second)
    angle_step_size: float = 0.01  # Size of step when manually modifying angles (radians)


class InteractiveVisualizer(abc.ABC):
    """Base class for visualizing rewards in RL environments."""

    # Mapping of key names to keycodes for commands that modify the environment state
    ENV_STATE_KEYCODES = {
        "up": 265,
        "down": 264,
        "right": 262,
        "left": 263,
        "p": ord("P"),
        "l": ord("L"),
        "n": ord("N"),
        "r": ord("R"),
        "q": ord("Q"),
        "e": ord("E"),
        "semicolon": ord(";"),
        "apostrophe": ord("'"),
        "period": ord("."),
        "slash": ord("/"),
    }

    def __init__(self, task: RLTask, config: InteractiveVisualizerConfig | None = None) -> None:
        """Initialize the reward and termination visualizer.

        Args:
            task: The RL task to visualize
            config: Configuration for the visualizer
        """
        if config is None:
            config = InteractiveVisualizerConfig()

        for key, value in self.ENV_STATE_KEYCODES.items():
            # Register key bindings for environment state control
            setattr(self, f"key_{key}", value)

        self.viz_config = config
        self.pos_step_size = self.viz_config.pos_step_size
        self.vel_step_size = self.viz_config.vel_step_size

        self.task = task

        # Add pause state variables
        self.paused = False
        self.suspended = False
        self.alternate_controls = False
        self.data_modifying_keycode: int | None = None

        # Set up reward history
        self.reward_history: dict[str, collections.deque[float]] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.viz_config.window_size)
        )

        # Set up termination history
        self.termination_history: dict[str, collections.deque[float]] = collections.defaultdict(
            lambda: collections.deque(maxlen=self.viz_config.window_size)
        )
        self.timestamps: collections.deque[float] = collections.deque(
            maxlen=self.viz_config.window_size
        )
        self.start_time = time.time()

        # Flag to control thread execution
        self.is_running = True

        self.viz_config.fig_save_dir.mkdir(parents=True, exist_ok=True)
        self.fig_save_path = self.viz_config.fig_save_dir / "plot.png"

        # Setup figure but don't display it yet
        self.fig, (self.ax_reward, self.ax_term) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.fig.savefig(self.fig_save_path)  # Save an initial empty plots

        # Start a thread to periodically save plot images
        self.plot_thread = threading.Thread(target=self.plot_loop)
        self.plot_thread.daemon = True
        self.plot_thread.start()

    @abc.abstractmethod
    def setup_environment(self) -> BaseEnv:
        """Set up the environment for visualization.

        Returns:
            The environment instance
        """
        pass

    def key_callback(self, keycode: int) -> None:
        """Handle key press events.

        Args:
            keycode: The keycode of the pressed key
        """
        logger.info("Key pressed: %s", chr(keycode))
        logger.info("Keycode: %d", keycode)

        if chr(keycode) == " ":  # Space key
            self.paused = not self.paused
            status = "PAUSED" if self.paused else "RUNNING"
            logger.info("Simulation %s", status)

        if chr(keycode) == "S":
            self.suspended = not self.suspended
            status = "SUSPENDED" if self.suspended else "RUNNING"
            logger.info("Simulation %s", status)

        # Shift key
        if keycode == 340:
            self.alternate_controls = not self.alternate_controls
            status = "ALTERNATE CONTROLS" if self.alternate_controls else "NORMAL CONTROLS"
            logger.info("Controls %s", status)

        if keycode in self.ENV_STATE_KEYCODES.values():
            self.data_modifying_keycode = keycode

    @abc.abstractmethod
    def run(self) -> None:
        """Run the visualization loop."""
        pass

    def update_plot(self) -> None:
        """Update plot and save to disk."""
        # Clear the axis
        self.ax_reward.clear()
        self.ax_term.clear()

        # Plot both reward and termination components
        for data_type, history in [
            ("reward", self.reward_history),
            ("term", self.termination_history),
        ]:
            for name, values in history.items():
                if values:  # Only plot if we have data
                    values_list = list(values)
                    timestamps_list = list(self.timestamps)
                    # Make sure we have matching lengths
                    min_len = min(len(values_list), len(timestamps_list))
                    if min_len > 0:
                        if data_type == "reward":
                            self.ax_reward.plot(
                                timestamps_list[-min_len:], values_list[-min_len:], label=name
                            )
                        else:
                            self.ax_term.plot(
                                timestamps_list[-min_len:], values_list[-min_len:], label=name
                            )

        # Set labels and legends
        self.ax_reward.set_ylabel("Reward")
        self.ax_reward.set_title("Rewards Visualization")
        self.ax_reward.legend(loc="upper left")

        self.ax_term.set_xlabel("Time (s)")
        self.ax_term.set_ylabel("Termination")
        self.ax_term.set_title("Termination Signals")
        self.ax_term.legend(loc="upper left")

        if self.timestamps:
            self.ax_reward.set_xlim(self.timestamps[0], self.timestamps[-1])
            self.ax_term.set_xlim(self.timestamps[0], self.timestamps[-1])

        # Add some spacing between subplots
        self.fig.tight_layout()

        # Save to disk
        self.fig.savefig(self.fig_save_path)

        # Log update message
        logger.debug("-" * 80)
        logger.debug("Plot updated with data from %d reward components", len(self.reward_history))
        for name, values in self.reward_history.items():
            if values:
                logger.debug("  %s: latest value = %.6f", name, values[-1])

        logger.debug("-" * 80)
        logger.debug(
            "Plot updated with data from %d termination components", len(self.termination_history)
        )
        for name, values in self.termination_history.items():
            if values:
                logger.debug("  %s: latest value = %.6f", name, values[-1])

        logger.debug("-" * 80)

    def plot_loop(self) -> None:
        """Background thread to periodically update the plot."""
        while self.is_running:
            if self.timestamps:  # Only update if we have data
                self.update_plot()
            time.sleep(0.05)  # Update at 20Hz

    def cleanup(self) -> None:
        """Clean up resources before shutting down."""
        self.is_running = False
        self.plot_thread.join(timeout=1.0)
        plt.close(self.fig)
        logger.info("Visualization terminated")
