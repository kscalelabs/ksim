"""Defines simple task for training a walking policy for K-Bot."""

import argparse
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import optax
import xax

from ksim.action.mjcf import MjcfAction
from ksim.env.kscale import KScaleEnvironment, KScaleEnvironmentConfig
from ksim.resets.mjcf import RandomYawReset, XYPositionReset
from ksim.state.mjcf import MjcfState
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.terminations.mjcf import (
    EpisodeLengthTermination,
    PitchTooGreatTermination,
    RollToGreatTermination,
)


class Model(eqx.Module):
    layers: list

    def __init__(self, rng_key: jnp.ndarray) -> None:
        super().__init__()

        # Split the PRNG key into four keys for the four layers.
        key1, key2, key3, key4 = jax.random.split(rng_key, 4)

        self.layers = [
            eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(1728, 512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, 10, key=key4),
            jax.nn.log_softmax,
        ]

    def forward(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.forward)(x)


@dataclass
class KBotWalkingConfig(PPOConfig, KScaleEnvironmentConfig):
    # Robot model name to use.
    model_name: str = xax.field(value="kbot-v1")
    kp: float = xax.field(value=100.0)
    kd: float = xax.field(value=10.0)

    # ML model parameters.
    actor_hidden_dims: list[int] = xax.field(value=[512, 256, 128])
    critic_hidden_dims: list[int] = xax.field(value=[512, 256, 128])
    init_noise_std: float = xax.field(value=1.0)

    # Environment configuration options.
    base_init_pos: tuple[float, float, float] = xax.field(value=(0.0, 0.0, 0.5))

    # Termination conditions.
    max_episode_length: float = xax.field(value=20.0)
    max_pitch: float = xax.field(value=0.1)
    max_roll: float = xax.field(value=0.1)


class KBotEnvironment(KScaleEnvironment[KBotWalkingConfig, MjcfState, MjcfAction]):
    def configure_mj_model(self, model: mujoco.MjModel, joint_names: list[str]) -> mujoco.MjModel:
        # Modify PD gains.
        model.dof_damping[:] = self.config.kp
        model.actuator_gainprm[:, 0] = self.config.kd
        model.actuator_biasprm[:, 0] = -self.config.kp

        # Increase offscreen framebuffer size to render at higher resolution.
        model.vis.global_.offwidth = 3840
        model.vis.global_.offheight = 2160

        return model

    def check_termination(self, state: MjcfState) -> jnp.ndarray:
        raise NotImplementedError


class KBotWalkingTask(PPOTask[KBotWalkingConfig]):
    def __init__(self, config: KBotWalkingConfig) -> None:
        super().__init__(config)

    def get_environment(self) -> KScaleEnvironment:
        return KScaleEnvironment(
            self.config,
            terminations=[
                EpisodeLengthTermination(max_episode_length_seconds=self.config.max_episode_length, dt=self.config.dt),
                PitchTooGreatTermination(max_pitch=self.config.max_pitch),
                RollToGreatTermination(max_roll=self.config.max_roll),
            ],
            resets=[
                XYPositionReset(x_range=(-0.5, 0.5), y_range=(-0.5, 0.5)),
                RandomYawReset(),
            ],
        )

    def get_model(self) -> Model:
        return Model(self.prng_key())

    def get_optimizer(self) -> optax.GradientTransformation:
        return optax.adam(1e-3)

    def get_output(self, model: Model, batch: jnp.ndarray) -> jnp.ndarray:
        return model(batch)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["env", "train", "test"])
    parser.add_argument("--num-steps", type=int, default=1000)
    args, rest = parser.parse_known_args()

    config = KBotWalkingConfig(
        # Training parameters.
        num_environments=1,
        batch_size=32,
        # Learning rate.
        learning_rate=1e-3,
    )

    match args.action:
        case "env":
            config.show_viewer = True
            KBotWalkingTask.run_environment(config, *rest, num_steps=args.num_steps, use_cli=False)

        case "train":
            KBotWalkingTask.launch(config, *rest, use_cli=False)

        case "test":
            raise NotImplementedError("Test mode not implemented.")

        case _:
            raise ValueError(f"Invalid action: {args.action}")


if __name__ == "__main__":
    # python -m examples.kbot.walking train
    main()
    main()
