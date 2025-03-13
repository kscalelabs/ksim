"""Static definitions for KBot experiments."""

from dataclasses import dataclass

import xax

from ksim.env.mjx.mjx_env import MjxEnvConfig
from ksim.task.ppo import PPOConfig


@dataclass
class KBotConfig(PPOConfig, MjxEnvConfig):
    """Combining configs for the KBot experiments and fixing params."""

    robot_model_name: str = xax.field(value="examples/kbot/")
