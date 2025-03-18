"""Minimal API that interfaces with env to unroll trajectories."""

from dataclasses import replace
from typing import Collection, NamedTuple

import chex
import jax
import jax.numpy as jnp
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

from ksim.commands import Command
from ksim.env.base_engine import PhysicsEngine
from ksim.env.data import PhysicsData, PhysicsState, Transition
from ksim.model.base import Agent
from ksim.model.types import ModelCarry
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


UnrollCarry = tuple[FrozenDict[str, Array], ModelCarry, PhysicsState]
UnrollYs = tuple[Transition, UnrollNaNDetector, PhysicsData | None]
# personal preference... don't like tuple-mania


def unroll_trajectory(
    physics_state: PhysicsState,
    rng: PRNGKeyArray,
    agent: Agent,
    obs_normalizer: Normalizer,
    cmd_normalizer: Normalizer,
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
    jax.debug.print("[300] unroll_trajectory: starting unroll")
    try:
        initial_carry = agent.actor_model.initial_carry()
        initial_command = get_initial_commands(rng, command_generators=command_generators)
        jax.debug.print("[301] unroll_trajectory: initial setup done")

        def state_transition(
            carry: UnrollCarry,
            rng: PRNGKeyArray,
        ) -> tuple[UnrollCarry, UnrollYs]:
            """Constructs transition, resets if needed."""
            jax.debug.print("[310.1] state_transition: ENTRY POINT")
            
            # Debug the RNG first
            jax.debug.print("[310.2] state_transition: rng shape={shape}, dtype={dtype}", 
                        shape=rng.shape if hasattr(rng, 'shape') else None, 
                        dtype=rng.dtype if hasattr(rng, 'dtype') else None)
            
            try:
                jax.debug.print("[310.3] state_transition: about to split rng")
                obs_rng, cmd_rng, act_rng, reset_rng, physics_rng = jax.random.split(rng, 5)
                jax.debug.print("[310.4] state_transition: rng split successful")
            except Exception as e:
                jax.debug.print("[ERR310] state_transition: Error splitting rng: {error}", error=str(e))
                raise
            
            try:
                jax.debug.print("[310.5] state_transition: unpacking carry")
                prev_command, carry, physics_state = carry
                jax.debug.print("[310.6] state_transition: carry unpacked successfully")
            except Exception as e:
                jax.debug.print("[ERR311] state_transition: Error unpacking carry: {error}", error=str(e))
                raise
            
            try:
                jax.debug.print("[310.7] state_transition: getting prev_action from physics_state")
                prev_action = physics_state.most_recent_action
                jax.debug.print("[310.8] state_transition: prev_action obtained, shape={shape}", 
                            shape=prev_action.shape if hasattr(prev_action, 'shape') else None)
            except Exception as e:
                jax.debug.print("[ERR312] state_transition: Error getting prev_action: {error}", error=str(e))
                raise
            
            # Debug what's in physics_state
            jax.debug.print("[310.9] state_transition: physics_state data shape: {shape}", 
                         shape=jax.tree_map(lambda x: getattr(x, 'shape', None) if hasattr(x, 'shape') else None, physics_state.data))
            
            try:
                jax.debug.print("[311.1] state_transition: about to get observation")
                obs = get_observation(physics_state, obs_rng, obs_generators=obs_generators)
                jax.debug.print("[311.2] state_transition: observation successful")
            except Exception as e:
                jax.debug.print("[ERR313] state_transition: Error in get_observation: {error}", error=str(e))
                raise
                
            try:
                jax.debug.print("[311.3] state_transition: about to get commands")
                command = get_commands(prev_command, physics_state, cmd_rng, command_generators=command_generators)
                jax.debug.print("[311.4] state_transition: commands successful")
            except Exception as e:
                jax.debug.print("[ERR314] state_transition: Error in get_commands: {error}", error=str(e))
                raise
                
            jax.debug.print("[311.5] state_transition: obs and command generated")

            try:
                jax.debug.print("[311.6] state_transition: about to normalize obs and command")
                normalized_obs = obs_normalizer(obs)
                normalized_cmd = cmd_normalizer(command)
                jax.debug.print("[311.7] state_transition: normalization successful")
            except Exception as e:
                jax.debug.print("[ERR315] state_transition: Error in normalization: {error}", error=str(e))
                raise
                
            try:
                jax.debug.print("[311.8] state_transition: about to call actor_model.forward")
                prediction, next_carry = agent.actor_model.forward(normalized_obs, normalized_cmd, carry)
                jax.debug.print("[311.9] state_transition: actor_model.forward successful")
            except Exception as e:
                jax.debug.print("[ERR316] state_transition: Error in actor_model.forward: {error}", error=str(e))
                raise
                
            try:
                jax.debug.print("[312.1] state_transition: about to sample action")
                action = agent.action_distribution.sample(prediction, act_rng)
                jax.debug.print("[312.2] state_transition: action sampling successful")
            except Exception as e:
                jax.debug.print("[ERR317] state_transition: Error in action sampling: {error}", error=str(e))
                raise
                
            jax.debug.print("[312.3] state_transition: prediction and action generated")
            
            # Debug before the engine step which might cause issues
            jax.debug.print("[313.1] state_transition: about to step engine")
            try:
                next_physics_state = engine.step(action, physics_state, physics_rng)
                jax.debug.print("[313.2] state_transition: engine stepped successfully")
            except Exception as e:
                jax.debug.print("[ERR318] state_transition: Error in engine.step: {error}", error=str(e))
                raise
            
            termination_components = get_terminations(next_physics_state, termination_generators=termination_generators)
            action_causes_termination = jax.tree_util.tree_reduce(jnp.logical_or, list(termination_components.values()))
            jax.debug.print("[315] state_transition: termination computed, action_causes_termination={term}", term=action_causes_termination)
            
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
            jax.debug.print("[316] state_transition: rewards computed")

            # resetting if nans or termination, resetting everything...
            next_data_has_nans = xax.pytree_has_nans(next_physics_state.data)
            do_reset = jnp.logical_or(action_causes_termination, next_data_has_nans)
            jax.debug.print("[317] state_transition: do_reset={do_reset}, next_data_has_nans={has_nans}", 
                         do_reset=do_reset, has_nans=next_data_has_nans)
            
            reset_state = engine.reset(reset_rng)
            jax.debug.print("[318] state_transition: engine reset completed")
            
            next_physics_state = xax.update_pytree(do_reset, reset_state, next_physics_state)
            next_carry = xax.update_pytree(do_reset, initial_carry, next_carry)
            command = xax.update_pytree(do_reset, initial_command, command)
            jax.debug.print("[319] state_transition: update_pytree calls completed")

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
            jax.debug.print("[320] state_transition: transition created")

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
            
            # Track if we have any NaNs
            any_nans = jax.tree_map(jnp.any, nan_mask)
            jax.debug.print("[321] state_transition: any_nans={any_nans}", any_nans=any_nans)
            
            # if is fine since condition will be static at runtime
            if return_intermediate_physics_data:
                return (command, next_carry, next_physics_state), (transition, nan_mask, next_physics_state)
            else:
                return (command, next_carry, next_physics_state), (transition, nan_mask, None)

        # Detailed debugging of the scan input values
        jax.debug.print("[302.1] unroll_trajectory: initial_carry type={type}", type=type(initial_carry).__name__)
        
        jax.debug.print("[302.2] unroll_trajectory: initial_command keys={keys}", 
                     keys=list(initial_command.keys()))
        jax.debug.print("[302.3] unroll_trajectory: physics_state model type={model_type}, data type={data_type}", 
                     model_type=type(physics_state.model).__name__,
                     data_type=type(physics_state.data).__name__)
        
        # Debug key properties of physics_state.data and model
        if hasattr(physics_state.data, "qpos"):
            jax.debug.print("[302.4] unroll_trajectory: physics_state.data.qpos shape={shape}", 
                         shape=physics_state.data.qpos.shape if hasattr(physics_state.data.qpos, 'shape') else None)
        
        # Debug the rng input to scan
        try:
            jax.debug.print("[302.5] unroll_trajectory: about to split rng for scan")
            scan_rngs = jax.random.split(rng, num_steps)
            jax.debug.print("[302.6] unroll_trajectory: split successful, scan_rngs shape={shape}", 
                         shape=scan_rngs.shape if hasattr(scan_rngs, 'shape') else None)
        except Exception as e:
            jax.debug.print("[ERR302] Error splitting RNG for scan: {error}", error=str(e))
            raise
        
        # Try to safely create the initial carry tuple
        try:
            jax.debug.print("[302.7] unroll_trajectory: creating initial carry tuple for scan")
            initial_scan_carry = (initial_command, initial_carry, physics_state)
            jax.debug.print("[302.8] unroll_trajectory: initial carry tuple created successfully")
        except Exception as e:
            jax.debug.print("[ERR303] Error creating initial carry tuple: {error}", error=str(e))
            raise

        # Try running scan with lots of error handling
        jax.debug.print("[302.9] unroll_trajectory: about to run scan over state_transition")
        try:
            scan_result = jax.lax.scan(
                state_transition,
                initial_scan_carry,
                scan_rngs,
            )
            jax.debug.print("[303.1] unroll_trajectory: scan executed successfully")
            (_, _, final_physics_state), (transition, nan_mask, intermediate_physics_data) = scan_result
            jax.debug.print("[303.2] unroll_trajectory: scan results unpacked successfully")
        except Exception as e:
            jax.debug.print("[ERR304] Error in jax.lax.scan: {error}", error=str(e))
            jax.debug.print("[ERR304.1] Error type: {error_type}", error_type=type(e).__name__)
            raise

        # post accumulating rewards (the only thing that makes sense to accumulate)
        post_accumulated_reward_components = post_accumulate_rewards(
            transition.reward_components,
            transition.done,
            reward_generators=reward_generators,
        )
        jax.debug.print("[304] unroll_trajectory: post_accumulate_rewards completed")

        post_accumulated_rewards = jax.tree_util.tree_reduce(jnp.add, list(post_accumulated_reward_components.values()))
        transition = replace(
            transition,
            reward_components=post_accumulated_reward_components,
            reward=post_accumulated_rewards,
        )
        has_nans_any = jax.tree_util.tree_map(jnp.any, nan_mask)
        jax.debug.print("[305] unroll_trajectory: unroll completed successfully")

        return transition, final_physics_state, has_nans_any, intermediate_physics_data
    
    except Exception as e:
        jax.debug.print("[ERR300] Exception in unroll_trajectory: {error}", error=str(e))
        raise


def get_observation(
    physics_state: PhysicsState,
    rng: PRNGKeyArray,
    *,
    obs_generators: Collection[Observation],
) -> FrozenDict[str, Array]:
    """Get the observation from the physics state."""
    jax.debug.print("[OBS1] get_observation: starting")
    
    try:
        jax.debug.print("[OBS2] get_observation: physics_state data type={type}", 
                    type=type(physics_state.data).__name__)
        
        # Check basic properties of physics_state.data that might be accessed by observation generators
        if hasattr(physics_state.data, "qpos"):
            jax.debug.print("[OBS3] get_observation: qpos shape={shape}", 
                        shape=physics_state.data.qpos.shape if hasattr(physics_state.data.qpos, 'shape') else None)
        else:
            jax.debug.print("[OBS3] get_observation: WARNING - data has no qpos attribute")
            
        if hasattr(physics_state.data, "qvel"):
            jax.debug.print("[OBS4] get_observation: qvel shape={shape}", 
                        shape=physics_state.data.qvel.shape if hasattr(physics_state.data.qvel, 'shape') else None)
        else:
            jax.debug.print("[OBS4] get_observation: WARNING - data has no qvel attribute")
            
        jax.debug.print("[OBS5] get_observation: initializing observations dict")
        observations = {}
        
        for i, observation in enumerate(obs_generators):
            obs_name = observation.observation_name
            jax.debug.print("[OBS6.{i}] Processing observation generator: {name}", i=i, name=obs_name)
            
            try:
                jax.debug.print("[OBS7.{i}] Splitting RNG for {name}", i=i, name=obs_name)
                rng, obs_rng = jax.random.split(rng)
                
                jax.debug.print("[OBS8.{i}] Calling observation generator {name}", i=i, name=obs_name)
                observation_value = observation(physics_state.data, obs_rng)
                
                jax.debug.print("[OBS9.{i}] Got value for {name}, shape={shape}", 
                             i=i, name=obs_name, 
                             shape=observation_value.shape if hasattr(observation_value, 'shape') else None)
                             
                observations[obs_name] = observation_value
                jax.debug.print("[OBS10.{i}] Added {name} to observations dict", i=i, name=obs_name)
            except Exception as e:
                jax.debug.print("[OBS_ERR{i}] Error processing observation {name}: {error}", 
                              i=i, name=obs_name, error=str(e))
                raise
        
        jax.debug.print("[OBS11] get_observation: creating FrozenDict")
        result = FrozenDict(observations)
        jax.debug.print("[OBS12] get_observation: returning result with keys {keys}", 
                      keys=list(result.keys()))
        return result
    except Exception as e:
        jax.debug.print("[OBS_ERR_MAIN] Unexpected error in get_observation: {error}", error=str(e))
        raise


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
    jax.debug.print("[CMD1] get_commands: starting")
    
    try:
        jax.debug.print("[CMD2] get_commands: prev_commands keys={keys}", 
                     keys=list(prev_commands.keys()))
        
        jax.debug.print("[CMD3] get_commands: physics_state.data.time={time}", 
                     time=physics_state.data.time if hasattr(physics_state.data, 'time') else "N/A")
                     
        jax.debug.print("[CMD4] get_commands: initializing commands dict")
        commands = {}
        
        for i, command_generator in enumerate(command_generators):
            command_name = command_generator.command_name
            jax.debug.print("[CMD5.{i}] Processing command generator: {name}", i=i, name=command_name)
            
            try:
                jax.debug.print("[CMD6.{i}] Getting prev_command for {name}", i=i, name=command_name)
                prev_command = prev_commands[command_name]
                
                jax.debug.print("[CMD7.{i}] prev_command shape for {name}={shape}", 
                             i=i, name=command_name, 
                             shape=prev_command.shape if hasattr(prev_command, 'shape') else None)
                
                assert isinstance(prev_command, Array)
                
                jax.debug.print("[CMD8.{i}] Calling command_generator for {name}", i=i, name=command_name)
                command_val = command_generator(prev_command, physics_state.data.time, rng)
                
                jax.debug.print("[CMD9.{i}] Got value for {name}, shape={shape}", 
                             i=i, name=command_name, 
                             shape=command_val.shape if hasattr(command_val, 'shape') else None)
                             
                commands[command_name] = command_val
                jax.debug.print("[CMD10.{i}] Added {name} to commands dict", i=i, name=command_name)
            except Exception as e:
                jax.debug.print("[CMD_ERR{i}] Error processing command {name}: {error}", 
                              i=i, name=command_name, error=str(e))
                raise
        
        jax.debug.print("[CMD11] get_commands: creating FrozenDict")
        result = FrozenDict(commands)
        jax.debug.print("[CMD12] get_commands: returning result with keys {keys}", 
                      keys=list(result.keys()))
        return result
    except Exception as e:
        jax.debug.print("[CMD_ERR_MAIN] Unexpected error in get_commands: {error}", error=str(e))
        raise


def get_initial_commands(
    rng: PRNGKeyArray,
    *,
    command_generators: Collection[Command],
) -> FrozenDict[str, Array]:
    """Get the initial commands from the physics state."""
    jax.debug.print("[ICMD1] get_initial_commands: starting")
    
    try:
        jax.debug.print("[ICMD2] get_initial_commands: initializing commands dict")
        commands = {}
        
        for i, command_generator in enumerate(command_generators):
            command_name = command_generator.command_name
            jax.debug.print("[ICMD3.{i}] Processing initial command generator: {name}", i=i, name=command_name)
            
            try:
                jax.debug.print("[ICMD4.{i}] Calling initial_command for {name}", i=i, name=command_name)
                command_val = command_generator.initial_command(rng)
                
                jax.debug.print("[ICMD5.{i}] Got value for {name}, shape={shape}", 
                             i=i, name=command_name, 
                             shape=command_val.shape if hasattr(command_val, 'shape') else None)
                             
                commands[command_name] = command_val
                jax.debug.print("[ICMD6.{i}] Added {name} to commands dict", i=i, name=command_name)
            except Exception as e:
                jax.debug.print("[ICMD_ERR{i}] Error processing initial command {name}: {error}", 
                              i=i, name=command_name, error=str(e))
                raise
        
        jax.debug.print("[ICMD7] get_initial_commands: creating FrozenDict")
        result = FrozenDict(commands)
        jax.debug.print("[ICMD8] get_initial_commands: returning result with keys {keys}", 
                      keys=list(result.keys()))
        return result
    except Exception as e:
        jax.debug.print("[ICMD_ERR_MAIN] Unexpected error in get_initial_commands: {error}", error=str(e))
        raise
