"""Minimal API that interfaces with env to unroll trajectories."""

from abc import ABC, abstractmethod
from typing import Collection, NamedTuple

import chex
import jax
import jax.numpy as jnp
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

from ksim.commands import Command
from ksim.env.base_engine import BaseEngine
from ksim.env.data import PhysicsData, PhysicsState, Transition
from ksim.model.base import Agent
from ksim.model.types import ModelInput, ModelRecurrence
from ksim.normalization import Normalizer
from ksim.observation import Observation
from ksim.rewards import Reward
from ksim.terminations import Termination


class UnrollNaNDetector(NamedTuple):
    obs_has_nans: Array
    command_has_nans: Array
    action_has_nans: Array
    next_physics_data_has_nans: Array
    reset_physics_data_has_nans: Array
    termination_has_nans: Array
    reward_has_nans: Array


UnrollCarry = tuple[FrozenDict[str, Array], ModelRecurrence, PhysicsState]
UnrollYs = tuple[Transition, UnrollNaNDetector, PhysicsData | None]
# personal preference... don't like tuple-mania


def unroll_trajectory(
    agent: Agent,
    physics_state: PhysicsState,
    obs_normalizer: Normalizer,
    cmd_normalizer: Normalizer,
    rng: PRNGKeyArray,
    *,  # everything below this must be static at runtime (helpful for jit)
    engine: BaseEngine,
    obs_generators: Collection[Observation],
    command_generators: Collection[Command],
    reward_generators: Collection[Reward],
    termination_generators: Collection[Termination],
    num_steps: int,
    return_intermediate_physics_data: bool = False,
) -> tuple[Transition, PhysicsState, UnrollNaNDetector, PhysicsData | None]:
    """Returns (stacked) transitions and final physics state.

    Key insight: unrolling really doesn't need the previous Transition since we
    can safely reset command, and since prev action is in PhysicsState.
    """
    initial_recurrence = agent.actor_model.initial_recurrence()
    initial_command = get_initial_commands(rng, command_generators=command_generators)

    def step_auto_reset_fn(
        carry: UnrollCarry,
        rng: PRNGKeyArray,
    ) -> tuple[UnrollCarry, UnrollYs]:
        """Gets obs, gets action, steps, gets reward, gets done, repeat."""
        obs_rng, cmd_rng, act_rng, reset_rng = jax.random.split(rng, 4)
        prev_command, recurrence, physics_state = carry
        prev_action = physics_state.most_recent_action
        obs = get_observation(physics_state, obs_rng, obs_generators=obs_generators)
        command = get_commands(prev_command, physics_state, cmd_rng, command_generators=command_generators)

        model_input = ModelInput(obs_normalizer(obs), cmd_normalizer(command), None, None)
        prediction, next_recurrence = agent.actor_model.forward(model_input, recurrence)
        action = agent.action_distribution.sample(prediction, act_rng)
        next_physics_state = engine.step(action, physics_state)

        termination_components = get_terminations(next_physics_state, termination_generators=termination_generators)
        action_causes_termination = jax.tree_util.tree_reduce(jnp.logical_or, termination_components.values())
        reward_components = get_rewards(
            prev_action=prev_action,
            physics_state=physics_state,
            command=command,
            action=action,
            next_physics_state=next_physics_state,
            next_state_terminates=action_causes_termination,
            reward_generators=reward_generators,
        )
        rewards = jax.tree_util.tree_reduce(jnp.add, reward_components.values())

        # resetting if nans or termination, resetting everything...
        next_data_has_nans = xax.pytree_has_nans(next_physics_state.data)
        do_reset = jnp.logical_or(action_causes_termination, next_data_has_nans)
        reset_state = engine.reset(reset_rng)
        next_physics_state = xax.update_pytree(do_reset, next_physics_state, reset_state)
        next_recurrence = xax.update_pytree(do_reset, next_recurrence, initial_recurrence)
        command = xax.update_pytree(do_reset, command, initial_command)

        transition = Transition(
            obs=obs,
            command=command,
            action=action,
            reward=rewards,
            done=do_reset,
            timestep=physics_state.data.time,
            termination_components=termination_components,
            reward_components=reward_components,
        )

        # there are a lot of places nans can occur... unlikely to occur outside
        # next_physics_state, but better to check all of them until they're gone
        nan_mask = UnrollNaNDetector(
            obs_has_nans=xax.pytree_has_nans(obs),
            command_has_nans=xax.pytree_has_nans(command),
            action_has_nans=xax.pytree_has_nans(action),
            next_physics_data_has_nans=next_data_has_nans,
            reset_physics_data_has_nans=xax.pytree_has_nans(reset_state.data),
            termination_has_nans=xax.pytree_has_nans(termination_components),
            reward_has_nans=xax.pytree_has_nans(rewards),
        )

        # if is fine since condition will be static at runtime
        if return_intermediate_physics_data:
            return (command, next_recurrence, next_physics_state), (transition, nan_mask, next_physics_state)
        else:
            return (command, next_recurrence, next_physics_state), (transition, nan_mask, None)

    (_, _, final_physics_state), (transition, nan_mask, intermediate_physics_data) = jax.lax.scan(
        step_auto_reset_fn,
        (initial_command, initial_recurrence, physics_state),
        jax.random.split(rng, num_steps),
    )

    return transition, final_physics_state, nan_mask, intermediate_physics_data


def get_observation(
    physics_state: PhysicsState,
    rng: PRNGKeyArray,
    *,
    obs_generators: Collection[Observation],
) -> FrozenDict[str, Array]:
    """Get the observation from the physics state."""
    observations = {}
    for observation in obs_generators:
        rng, obs_rng = jax.random.split(rng)
        observation_value = observation(physics_state.data, obs_rng)
        observations[observation.observation_name] = observation_value
    return FrozenDict(observations)


def get_rewards(
    prev_action: Array,
    physics_state: PhysicsState,
    command: FrozenDict[str, Array],
    action: Array,
    next_physics_state: PhysicsState,
    next_state_terminates: Array,
    *,
    reward_generators: Collection[Reward],
) -> FrozenDict[str, Array]:
    """Get the rewards from the physics state."""
    rewards = {}
    for reward_generator in reward_generators:
        reward_val = (
            reward_generator(
                prev_action=prev_action,
                physics_state=physics_state,
                command=command,
                action=action,
                next_physics_state=next_physics_state,
                next_state_terminates=next_state_terminates,
            )
            * reward_generator.scale
        )
        name = reward_generator.reward_name
        chex.assert_shape(
            reward_val,
            (),
            custom_message=f"Reward {name} must be a scalar",
        )
        rewards[name] = reward_val
    return FrozenDict(rewards)


def get_terminations(
    physics_state: PhysicsState,
    *,
    termination_generators: Collection[Termination],
) -> FrozenDict[str, Array]:
    """Get the terminations from the physics state."""
    terminations = {}
    for termination in termination_generators:
        termination_val = termination(physics_state.data)
        name = termination.termination_name
        terminations[name] = termination_val
    return FrozenDict(terminations)


def get_commands(
    prev_commands: FrozenDict[str, Array],
    physics_state: PhysicsState,
    rng: PRNGKeyArray,
    *,
    command_generators: Collection[Command],
) -> FrozenDict[str, Array]:
    """Get the commands from the physics state."""
    commands = {}
    for command_generator in command_generators:
        command_name = command_generator.command_name
        prev_command = prev_commands[command_name]
        assert isinstance(prev_command, Array)
        command_val = command_generator(prev_command, physics_state.data.time, rng)
        commands[command_name] = command_val
    return FrozenDict(commands)


def get_initial_commands(
    rng: PRNGKeyArray,
    *,
    command_generators: Collection[Command],
) -> FrozenDict[str, Array]:
    """Get the initial commands from the physics state."""
    commands = {}
    for command_generator in command_generators:
        command_name = command_generator.command_name
        command_val = command_generator.initial_command(rng)
        commands[command_name] = command_val
    return FrozenDict(commands)
