"""Defines simple task for training a walking policy for K-Bot."""

import argparse
from dataclasses import dataclass

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import xax
from jaxtyping import PRNGKeyArray

from ksim.env.brax import KScaleEnv, KScaleEnvConfig
from ksim.observation.mjcf import (
    BaseAngularVelocityObservation,
    BaseLinearVelocityObservation,
    BaseOrientationObservation,
    BasePositionObservation,
    JointPositionObservation,
    JointVelocityObservation,
)
from ksim.resets.mjcf import XYPositionReset
from ksim.rewards.mjcf import LinearVelocityZPenalty
from ksim.task.ppo import PPOConfig, PPOTask
from ksim.terminations.mjcf import PitchTooGreatTermination, RollTooGreatTermination


class Model(eqx.Module):
    input_layer: eqx.nn.Linear
    rnn: eqx.nn.GRUCell
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        num_hidden: int,
        *,
        key: PRNGKeyArray,
    ) -> None:
        super().__init__()

        # Split the PRNG key into four keys for the four layers.
        key1, key2, key3 = jax.random.split(key, 3)

        self.input_layer = eqx.nn.Linear(
            in_features=num_inputs,
            out_features=num_hidden,
            use_bias=True,
            key=key1,
        )

        self.rnn = eqx.nn.GRUCell(
            input_size=num_hidden,
            hidden_size=num_hidden,
            use_bias=True,
            key=key2,
        )

        self.output_layer = eqx.nn.Linear(
            in_features=num_hidden,
            out_features=num_outputs,
            use_bias=True,
            key=key3,
        )

    def forward(self, x_tn: jnp.ndarray) -> jnp.ndarray:
        x_tn = self.input_layer(x_tn)
        scan_fn = lambda state, x: (self.rnn(x, state), None)
        init_state = jnp.zeros(self.rnn.hidden_size)
        final_state, _ = jax.lax.scan(scan_fn, init_state, x_tn)
        return self.output_layer(final_state)

    def __call__(self, x_tn: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self.forward)(x_tn)


@dataclass
class KBotWalkingConfig(PPOConfig, KScaleEnvConfig):
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


class KBotWalkingTask(PPOTask[KBotWalkingConfig]):
    def __init__(self, config: KBotWalkingConfig) -> None:
        super().__init__(config)

    def get_environment(self) -> KScaleEnv:
        return KScaleEnv(
            self.config,
            terminations=[
                PitchTooGreatTermination(max_pitch=self.config.max_pitch),
                RollTooGreatTermination(max_roll=self.config.max_roll),
            ],
            resets=[
                XYPositionReset(x_range=(-0.5, 0.5), y_range=(-0.5, 0.5)),
            ],
            rewards=[
                LinearVelocityZPenalty(scale=-1.0),
            ],
            observations=[
                BasePositionObservation(),
                BaseOrientationObservation(),
                BaseLinearVelocityObservation(),
                BaseAngularVelocityObservation(),
                JointPositionObservation(),
                JointVelocityObservation(),
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
        batch_size=32,
        # Learning rate.
        learning_rate=1e-3,
    )

    match args.action:
        case "env":
            config.show_viewer = True
            KBotWalkingTask.run_environment(
                config,
                *rest,
                num_steps=args.num_steps,
                render_path="kbot-walking.mp4",
                use_cli=False,
            )

        case "train":
            KBotWalkingTask.launch(config, *rest, use_cli=False)

        case "test":
            raise NotImplementedError("Test mode not implemented.")

        case _:
            raise ValueError(f"Invalid action: {args.action}")


if __name__ == "__main__":
    # python -m examples.kbot.walking train
    main()
