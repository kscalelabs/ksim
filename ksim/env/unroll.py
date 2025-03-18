"""Minimal API that interfaces with env to unroll trajectories."""

from dataclasses import replace
from typing import Collection, NamedTuple

import chex
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

from ksim.commands import Command
from ksim.env.base_engine import PhysicsEngine
from ksim.env.data import PhysicsData, PhysicsState, Transition
from ksim.model import Agent, ModelCarry
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


UnrollCarry = tuple[FrozenDict[str, Array], ModelCarry, PhysicsState]
UnrollYs = tuple[Transition, UnrollNaNDetector, PhysicsData | None]


def unroll_trajectory(
    physics_state: PhysicsState,
    rng: PRNGKeyArray,
    agent: Agent,
    engine: PhysicsEngine,
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
    initial_carry = agent.actor_model.initial_carry()
    initial_command = get_initial_commands(rng, command_generators=command_generators)

    def state_transition(
        carry: UnrollCarry,
        rng: PRNGKeyArray,
    ) -> tuple[UnrollCarry, UnrollYs]:
        """Constructs transition, resets if needed."""
        obs_rng, cmd_rng, act_rng, reset_rng, physics_rng = jax.random.split(rng, 5)
        prev_command, carry, physics_state = carry
        prev_action = physics_state.most_recent_action
        obs = get_observation(physics_state, obs_rng, obs_generators=obs_generators)
        command = get_commands(prev_command, physics_state, cmd_rng, command_generators=command_generators)

        # we still return unnormalized obs and command to calculate normalization statistics
        prediction, next_carry = agent.actor_model.forward(obs, command, carry)
        action = agent.action_distribution.sample(prediction, act_rng)
        next_physics_state = engine.step(action, physics_state, physics_rng)

        termination_components = get_terminations(next_physics_state, termination_generators=termination_generators)
        action_causes_termination = jax.tree_util.tree_reduce(jnp.logical_or, list(termination_components.values()))
        reward_components = get_rewards(
            prev_action=prev_action,
            physics_state=physics_state,
            command=command,
            action=action,
            next_physics_state=next_physics_state,
            next_state_terminates=action_causes_termination,
            reward_generators=reward_generators,
        )
        rewards = jax.tree_util.tree_reduce(jnp.add, reward_components)

        # resetting if nans or termination, resetting everything...
        next_data_has_nans = xax.pytree_has_nans(next_physics_state.data)
        do_reset = jnp.logical_or(action_causes_termination, next_data_has_nans)
        reset_state = engine.reset(reset_rng)
        next_physics_state = xax.update_pytree(do_reset, reset_state, next_physics_state)
        next_carry = xax.update_pytree(do_reset, initial_carry, next_carry)
        command = xax.update_pytree(do_reset, initial_command, command)

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
            return (command, next_carry, next_physics_state), (
                transition,
                nan_mask,
                next_physics_state,
            )
        else:
            return (command, next_carry, next_physics_state), (transition, nan_mask, None)

    (_, _, final_physics_state), (transition, nan_mask, intermediate_physics_data) = jax.lax.scan(
        state_transition,
        (initial_command, initial_carry, physics_state),
        jax.random.split(rng, num_steps),
    )

    # Compute post-accumulated rewards (the only thing that makes sense to accumulate).
    post_accumulated_reward_components = post_accumulate_rewards(
        transition.reward_components,
        transition.done,
        reward_generators=reward_generators,
    )
    post_accumulated_rewards = jax.tree_util.tree_reduce(jnp.add, list(post_accumulated_reward_components.values()))

    transition = replace(
        transition,
        reward_components=post_accumulated_reward_components,
        reward=post_accumulated_rewards,
    )
    has_nans_any = jax.tree_util.tree_map(jnp.any, nan_mask)

    return transition, final_physics_state, has_nans_any, intermediate_physics_data


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
    next_physics_state: PhysicsState,  # TODO - rewards only process data
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
                physics_state=physics_state.data,
                command=command,
                action=action,
                next_physics_state=next_physics_state.data,
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


def post_accumulate_rewards(
    reward_components: FrozenDict[str, Array],
    done: Array,
    *,
    reward_generators: Collection[Reward],
) -> FrozenDict[str, Array]:
    """Post-accumulate rewards."""
    post_accumulated_reward_components = dict(reward_components)
    for reward_generator in reward_generators:
        original_reward = reward_components[reward_generator.reward_name]
        assert isinstance(original_reward, Array)
        reward_val = reward_generator.post_accumulate(original_reward, done)
        post_accumulated_reward_components[reward_generator.reward_name] = reward_val

    return FrozenDict(post_accumulated_reward_components)


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


def render_data_to_frames(
    data: PhysicsData,
    default_mj_model: mujoco.MjModel,
    camera: int | str | mujoco.MjvCamera = -1,
    height: int = 240,
    width: int = 320,
) -> list[np.ndarray]:
    """Render the data to a sequence of Numpy arrays."""
    for leaf in jax.tree.leaves(data):
        if isinstance(leaf, Array):
            num_steps = leaf.shape[0]
            break
    else:
        raise ValueError("No array found in data")

    mjx_data_list = [jax.tree.map(lambda x: x[i], data) for i in range(num_steps)]
    scene_option = mujoco.MjvOption()

    renderer = mujoco.Renderer(default_mj_model, height=height, width=width)
    frames = []
    for mjx_data in mjx_data_list:
        renderer.update_scene(mjx_data, camera=camera, scene_option=scene_option)
        frames.append(renderer.render())

    return frames
