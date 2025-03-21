"""Randomize each environment when gathering trajectories."""

import functools
from abc import ABC, abstractmethod

import attrs
import jax
import jax.numpy as jnp
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

from ksim.env.data import PhysicsData, PhysicsModel
from ksim.utils.mujoco import update_data_field


@attrs.define(frozen=True, kw_only=True)
class Randomization(ABC):
    """Randomize the joint positions of the robot."""

    @abstractmethod
    def initial_randomization(self, rng: PRNGKeyArray) -> tuple[PhysicsModel, PhysicsData]:
        """Randomize the model for a single environment."""

    @abstractmethod
    def __call__(self, model: PhysicsModel, data: PhysicsData, rng: PRNGKeyArray) -> tuple[PhysicsModel, PhysicsData]:
        """Randomize the model for a single environment."""

    def get_name(self) -> str:
        """Get the name of the command."""
        return xax.camelcase_to_snakecase(self.__class__.__name__)

    @functools.cached_property
    def randomization_name(self) -> str:
        return self.get_name()


@attrs.define(frozen=True, kw_only=True)
class ForceRandomization(Randomization):
    """Randomize the force of the robot."""

    push_magnitude_range: tuple[float, float] = attrs.field()
    push_interval_range: tuple[float, float] = attrs.field()
    trajectory_length: float = attrs.field()

    def initial_randomization(self, rng: PRNGKeyArray) -> tuple[PhysicsModel, PhysicsData]:
        """Randomize the force of the robot."""
        rng, push1_rng, push2_rng = jax.random.split(rng, 3)
        push_interval = jax.random.uniform(
            push2_rng,
            minval=self.push_interval_range[0],
            maxval=self.push_interval_range[1],
        )
        push_interval_steps = jnp.round(push_interval / self.dt).astype(jnp.int32)
        push_step = jax.random.randint(
            push1_rng, shape=(), minval=0, maxval=jnp.maximum(1, self.trajectory_length - push_interval)
        )
        return jnp.array([push_interval_steps, push_step])

    def __call__(
        self, randomization_state: FrozenDict[str, Array], model: PhysicsModel, data: PhysicsData, rng: PRNGKeyArray
    ) -> tuple[FrozenDict[str, Array], PhysicsModel, PhysicsData]:
        """Push the model for a single environment."""
        rng, push1_rng, push2_rng = jax.random.split(rng, 3)
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jnp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self.push_magnitude_range[0],
            maxval=self.push_magnitude_range[1],
        )
        push = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)])
        push *= (
            jnp.mod(
                randomization_state[self.randomization_name][0] + 1, randomization_state[self.randomization_name][0]
            )
            == 0
        )
        randomization_state = randomization_state.copy(add_or_replace={self.randomization_name: jnp.array([1, 0])})

        push = push * push_magnitude + data.qvel[:2]
        new_qvel = jnp.concatenate([push, data.qvel[2:]])
        return randomization_state, model, update_data_field(data, "qvel", new_qvel)
