"""The environment used to run massively parallel rollouts.

Philosophy:
- Think of the environment as a function of the state, system, action, and rng.
- Rollouts are performed by vectorizing (vmap) the reset and step functions, with a final trajectory
  of shape (time, num_envs, *obs_shape/s).

Rollouts return a trajectory of shape (time, num_envs, ).
"""

import functools
import logging
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, TypeVar

import chex
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray
from kscale.web.gen.api import RobotURDFMetadataOutput
from mujoco import mjx

from ksim.actuators import Actuators, ActuatorsBuilder
from ksim.commands import Command, CommandBuilder
from ksim.env.base_env import BaseEnv, BaseEnvConfig
from ksim.env.types import EnvState
from ksim.model.base import Agent
from ksim.model.types import ModelInput
from ksim.normalization import Normalizer
from ksim.observation import Observation, ObservationBuilder
from ksim.resets import Reset, ResetBuilder
from ksim.rewards import Reward, RewardBuilder
from ksim.terminations import Termination, TerminationBuilder
from ksim.utils.data import BuilderData
from ksim.utils.mujoco import make_mujoco_mappings

logger = logging.getLogger(__name__)

T = TypeVar("T")


@xax.profile
@xax.jit()
def mjx_forward(mjx_model: mjx.Model, mjx_data: mjx.Data) -> mjx.Data:
    """Forward pass of the mjx model."""
    return mjx.forward(mjx_model, mjx_data)


@xax.profile
@xax.jit()
def mjx_step(mjx_model: mjx.Model, mjx_data: mjx.Data) -> mjx.Data:
    """Step pass of the mjx model."""
    return mjx.step(mjx_model, mjx_data)


def _unique_list(things: list[tuple[str, T]]) -> list[tuple[str, T]]:
    """Ensures that all names are unique."""
    names: set[str] = set()
    return_list: list[tuple[str, T]] = []
    for base_name, thing in things:
        name, idx = base_name, 1
        while name in names:
            idx += 1
            name = f"{base_name}_{idx}"
        names.add(name)
        return_list.append((name, thing))
    return return_list


@jax.tree_util.register_dataclass
@dataclass
class MjxEnvConfig(BaseEnvConfig):
    # Environment.
    debug_env: bool = xax.field(
        value=False,
        help="Whether to enable debug mode for the env.",
    )

    # Action.
    min_action_latency: float = xax.field(
        value=0.0,
        help="The minimum action latency.",
    )
    max_action_latency: float = xax.field(
        value=0.0,
        help="The maximum action latency.",
    )

    # Solver.
    solver_iterations: int = xax.field(
        value=6,
        help="Number of main solver iterations.",
    )
    solver_ls_iterations: int = xax.field(
        value=6,
        help="Number of line search iterations.",
    )
    disable_flags_bitmask: int = xax.field(
        value=mujoco.mjtDisableBit.mjDSBL_EULERDAMP.value,
        help="Bitmask of flags to disable.",
    )


Config = TypeVar("Config", bound=MjxEnvConfig)


class MjxEnv(BaseEnv[Config], ABC):
    """Wraps the MuJoCo MJX model.

    Args:
        config: The configuration for the environment.
        robot_model_path: The path to the robot model.
        actuators: The actuators for the environment.
        terminations: The terminations for the environment.
        resets: The resets for the environment.
        rewards: The rewards for the environment.
        observations: The observations for the environment.
    """

    config: Config
    actuators: Actuators

    def __init__(
        self,
        config: Config,
        robot_model_path: str | Path,
        robot_metadata_path: str | Path | None,
        actuators: Actuators | ActuatorsBuilder,
        terminations: Collection[Termination | TerminationBuilder],
        resets: Collection[Reset | ResetBuilder],
        rewards: Collection[Reward | RewardBuilder],
        observations: Collection[Observation | ObservationBuilder],
        commands: Collection[Command | CommandBuilder] = (),
    ) -> None:
        self.config = config
        if self.config.max_action_latency < self.config.min_action_latency:
            raise ValueError(
                f"Maximum action latency ({self.config.max_action_latency}) must be greater than "
                f"minimum action latency ({self.config.min_action_latency})"
            )
        if self.config.min_action_latency < 0:
            raise ValueError(
                f"Action latency ({self.config.min_action_latency}) must be non-negative"
            )

        self.min_action_latency_step = round(self.config.min_action_latency / self.config.dt)
        self.max_action_latency_step = round(self.config.max_action_latency / self.config.dt)

        # ML: we will probably overhaul all of this in due time.
        # right now loading model and metadata from kscale-assets submodule directly.
        # when we have clarity on where the assets should come from, we should rewrite.
        logger.info("Loading robot model %s", robot_model_path)
        mj_model = mujoco.MjModel.from_xml_path(str(robot_model_path))
        mj_model = self._override_model_settings(mj_model)
        if robot_metadata_path is not None:
            with open(robot_metadata_path, "r") as f:
                robot_metadata = RobotURDFMetadataOutput.model_validate_json(f.read())
        else:
            robot_metadata = None

        self.default_mj_model = mj_model
        self.default_mj_data = mujoco.MjData(mj_model)
        self.default_mjx_model = mjx.put_model(mj_model)
        self.default_mjx_data = mjx.make_data(self.default_mjx_model)

        self.mujoco_mappings = make_mujoco_mappings(self.default_mjx_model)

        # preparing builder data.
        data = BuilderData(
            robot_model=self.default_mjx_model,
            robot_metadata=robot_metadata,
            dt=self.config.dt,
            ctrl_dt=self.config.ctrl_dt,
            mujoco_mappings=self.mujoco_mappings,
        )

        # creating actuators with metadata (if required)
        if isinstance(actuators, ActuatorsBuilder):
            self.actuators = actuators(data)
        else:
            self.actuators = actuators

        # storing the termination, reset, reward, observation, and command builders
        terminations_v = [t(data) if isinstance(t, TerminationBuilder) else t for t in terminations]
        resets_v = [r(data) if isinstance(r, ResetBuilder) else r for r in resets]
        rewards_v = [r(data) if isinstance(r, RewardBuilder) else r for r in rewards]
        observations_v = [o(data) if isinstance(o, ObservationBuilder) else o for o in observations]
        commands_v = [c(data) if isinstance(c, CommandBuilder) else c for c in commands]

        self.terminations = _unique_list([(term.termination_name, term) for term in terminations_v])
        self.resets = _unique_list([(reset.reset_name, reset) for reset in resets_v])
        self.rewards = _unique_list([(reward.reward_name, reward) for reward in rewards_v])
        self.observations = _unique_list([(obs.observation_name, obs) for obs in observations_v])
        self.commands = _unique_list([(cmd.command_name, cmd) for cmd in commands_v])

        # For simplicity, assume integer (increase granularity if needed).
        assert self.config.ctrl_dt % self.config.dt == 0, "ctrl_dt must be a multiple of dt"
        self.physics_dt_per_ctrl_dt = int(self.config.ctrl_dt / self.config.dt)

    def _override_model_settings(self, mj_model: mujoco.MjModel) -> mujoco.MjModel:
        """Override default sim settings."""
        mj_model.opt.iterations = self.config.solver_iterations
        mj_model.opt.ls_iterations = self.config.solver_ls_iterations
        mj_model.opt.timestep = self.config.dt
        mj_model.opt.disableflags = self.config.disable_flags_bitmask
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        return mj_model

    @xax.profile
    @xax.jit(static_argnames=["self"])
    def get_observation(self, mjx_data_L: mjx.Data, rng: jax.Array) -> FrozenDict[str, Array]:
        """Compute observations from the pipeline state."""
        observations = {}
        for observation_name, observation in self.observations:
            rng, obs_rng = jax.random.split(rng)
            observation_value = observation(mjx_data_L, obs_rng)
            observations[observation_name] = observation_value
        return FrozenDict(observations)

    @xax.profile
    @xax.jit(static_argnames=["self"])
    def get_rewards(
        self,
        action_L_t_minus_1: Array,
        mjx_data_L_t: mjx.Data,
        command_L_t: FrozenDict[str, Array],
        action_L_t: Array,
        mjx_data_L_t_plus_1: mjx.Data,
    ) -> FrozenDict[str, Array]:
        """Compute rewards from the state transition."""
        rewards = {}
        for reward_name, reward in self.rewards:
            reward_val = (
                reward(
                    action_t_minus_1=action_L_t_minus_1,
                    mjx_data_t=mjx_data_L_t,
                    command_t=command_L_t,
                    action_t=action_L_t,
                    mjx_data_t_plus_1=mjx_data_L_t_plus_1,
                )
                * reward.scale
            )
            chex.assert_shape(
                reward_val,
                (),
                custom_message=f"Reward {reward_name} must be a scalar",
            )
            rewards[reward_name] = reward_val
        return FrozenDict(rewards)

    @xax.profile
    @xax.jit(static_argnames=["self"])
    def get_terminations(self, mjx_data_L_t_plus_1: mjx.Data) -> FrozenDict[str, Array]:
        """Compute termination conditions from the pipeline state."""
        terminations = {}
        for termination_name, termination in self.terminations:
            term_val = termination(mjx_data_L_t_plus_1)
            chex.assert_shape(
                term_val, (), custom_message=f"Termination {termination_name} must be a scalar"
            )
            terminations[termination_name] = term_val
        return FrozenDict(terminations)

    @xax.profile
    @xax.jit(static_argnames=["self"])
    def get_initial_commands(
        self, rng: PRNGKeyArray, initial_time: Array | None
    ) -> FrozenDict[str, Array]:
        """Compute initial commands from the pipeline state."""
        commands = {}
        if initial_time is None:
            initial_time = jnp.array(0.0)
        for command_name, command_def in self.commands:
            assert isinstance(command_def, Command)
            command_val = command_def(rng, initial_time)
            commands[command_name] = command_val
        return FrozenDict(commands)

    @xax.profile
    @xax.jit(static_argnames=["self"])
    def get_commands(
        self, prev_commands: FrozenDict[str, Array], rng: PRNGKeyArray, time: Array
    ) -> FrozenDict[str, Array]:
        """Compute commands from the pipeline state."""
        commands = {}
        for command_name, command_def in self.commands:
            assert isinstance(command_def, Command)
            prev_command = prev_commands[command_name]
            assert isinstance(prev_command, Array)
            command_val = command_def.update(prev_command, rng, time)
            commands[command_name] = command_val
        return FrozenDict(commands)

    #####################################
    # Stepping and Resetting Main Logic #
    #####################################

    @xax.profile
    def apply_physics_steps(
        self,
        mjx_model_L: mjx.Model,
        mjx_data_L: mjx.Data,
        previous_action_L: Array,
        current_action_L: Array,
        num_latency_steps: Array,
    ) -> mjx.Data:
        """A 'step' of the environment on a state composes multiple steps of the actual physics.

        We take num_latency_steps (continuing the feedback loop for the previous action), then we
        apply the feedback loop for the current action for the remainder of the physics steps.
        """
        n_steps = self.physics_dt_per_ctrl_dt

        def f(carry: tuple[mjx.Data, Array], _: None) -> tuple[tuple[mjx.Data, Array], None]:
            data, step_num = carry

            action_motor_sees = jax.lax.select(
                step_num >= num_latency_steps,
                current_action_L,
                previous_action_L,
            )
            torques = self.actuators.get_ctrl(data, action_motor_sees)
            data_with_ctrl = data.replace(ctrl=torques)
            data_with_ctrl = mjx_forward(mjx_model_L, data_with_ctrl)
            new_data = mjx_step(mjx_model_L, data_with_ctrl)
            return (new_data, step_num + 1.0), None

        final_data_L = jax.lax.scan(f, (mjx_data_L, jnp.array(0.0)), None, n_steps)[0][0]
        return final_data_L

    @xax.profile
    def get_init_physics_data(
        self,
        num_envs: int,
    ) -> mjx.Data:
        """Get initial mjx.Data (EL)."""

        def get_init_data(rng: jax.Array) -> mjx.Data:
            data = mjx.make_data(self.default_mjx_model)
            return data

        rngs = jax.random.split(jax.random.PRNGKey(0), num_envs)
        default_data_EL = jax.vmap(get_init_data)(rngs)

        mjx_data_EL_0 = jax.vmap(mjx_forward, in_axes=(None, 0))(
            self.default_mjx_model, default_data_EL
        )

        return mjx_data_EL_0

    def get_init_physics_model(self) -> mjx.Model:
        """Get the initial physics model for the environment (L)."""
        return self.default_mjx_model

    @xax.profile
    def get_dummy_env_states(
        self,
        num_envs: int,
    ) -> EnvState:
        """Get initial environment states (EL)."""
        rng = jax.random.PRNGKey(0)
        cmd_rng, obs_rng = jax.random.split(rng, 2)
        mjx_data_EL_0 = self.get_init_physics_data(num_envs)
        timestep_E = mjx_data_EL_0.time
        obs_rng_E = jax.random.split(obs_rng, num_envs)
        command_rng_E = jax.random.split(cmd_rng, num_envs)
        obs_dummy_EL = jax.vmap(self.get_observation)(mjx_data_EL_0, obs_rng_E)
        command_dummy_EL = jax.vmap(self.get_initial_commands)(command_rng_E, timestep_E)

        termination_components = {name: jnp.zeros((num_envs,)) for name, _ in self.terminations}
        reward_components = {name: jnp.zeros((num_envs,)) for name, _ in self.rewards}

        return EnvState(
            obs=obs_dummy_EL,
            command=command_dummy_EL,
            action=jnp.ones((num_envs, self.default_mjx_model.nu)),
            reward=jnp.ones((num_envs,)),
            done=jnp.zeros((num_envs,), dtype=jnp.bool_),
            timestep=timestep_E,
            termination_components=FrozenDict(termination_components),
            reward_components=FrozenDict(reward_components),
        )

    @xax.profile
    def reset(
        self,
        agent: Agent,
        rng: jax.Array,
        physics_model_L: mjx.Model,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
    ) -> tuple[EnvState, mjx.Data]:
        """Using agent, returns initial state and data (EL, EL).

        We couple the step and actor because we couple the actions with the rest
        of env state. This ultimately allows for extremely constrained `EnvState`s,
        which promote correct RL code.

        Additionally, using a carry state for mjx model and data allows for
        efficient unrolling of trajectories.
        """
        mjx_model_L = physics_model_L
        mjx_data_L_0 = mjx.make_data(mjx_model_L)

        for _, reset_func in self.resets:
            mjx_data_L_0 = reset_func(mjx_data_L_0, rng)
        assert isinstance(mjx_data_L_0, mjx.Data)

        rng, obs_rng = jax.random.split(rng, 2)
        timestep = jnp.array(0.0)

        mjx_data_L_0 = mjx_forward(mjx_model_L, mjx_data_L_0)
        obs_L_0 = self.get_observation(mjx_data_L_0, obs_rng)
        command_L_0 = self.get_initial_commands(rng, timestep)

        model_input = ModelInput(
            obs=obs_normalizer(obs_L_0),
            command=cmd_normalizer(command_L_0),
            action_history=None,
            recurrent_state=None,
        )

        prediction = agent.actor_model.forward(model_input)
        action_L_0 = agent.action_distribution.sample(prediction, rng)

        mjx_data_L_1 = self.apply_physics_steps(
            mjx_model_L=mjx_model_L,
            mjx_data_L=mjx_data_L_0,
            previous_action_L=action_L_0,  # NOTE: Effectively means no latency for first action.
            current_action_L=action_L_0,
            num_latency_steps=jnp.array(0),  # Enforced here as well...
        )

        term_components_L_0 = self.get_terminations(mjx_data_L_1)
        reward_components_L_0 = self.get_rewards(
            action_L_0, mjx_data_L_0, command_L_0, action_L_0, mjx_data_L_1
        )

        done_L_0 = jnp.stack([v for _, v in term_components_L_0.items()]).any()
        reward_L_0 = jnp.stack([v for _, v in reward_components_L_0.items()]).sum()

        env_state_L_0 = EnvState(
            obs=obs_L_0,
            command=command_L_0,
            action=action_L_0,
            reward=reward_L_0,
            done=done_L_0,
            timestep=timestep,
            termination_components=FrozenDict(term_components_L_0),
            reward_components=FrozenDict(reward_components_L_0),
        )
        return env_state_L_0, mjx_data_L_1

    @xax.profile
    def step(
        self,
        agent: Agent,
        env_state_L_t_minus_1: EnvState,
        rng: PRNGKeyArray,
        physics_data_L_t: mjx.Data,
        physics_model_L: mjx.Model,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
    ) -> tuple[EnvState, mjx.Data]:
        """Stepping the environment in a consistent, JIT-able manner. Works on a single environment.

        We couple the step and actor because we couple the actions with the rest of env state. This
        ultimately allows for extremely constrained `EnvState`s, which promote correct RL code.

        At t=t_0:
            - Action is sampled.
            - Latency steps are sampled.

        At t=t_i:
            - If i < latency_steps, apply the previous action. Otherwise, apply the current action.
            - Physics step is taken.
            - TODO: State perturbations are applied.

        At t=t_f:
            - The final state is returned.
            - Observations are computed.
            - Rewards are computed.
            - Terminations are computed.
        """
        mjx_model_L = physics_model_L
        mjx_data_L_t = physics_data_L_t

        rng, latency_rng, obs_rng = jax.random.split(rng, 3)
        timestep = mjx_data_L_t.time
        latency_steps = jax.random.randint(
            key=latency_rng,
            shape=(),
            minval=self.min_action_latency_step,
            maxval=self.max_action_latency_step,
        )

        obs_L_t = self.get_observation(mjx_data_L_t, obs_rng)
        command_L_t = self.get_commands(env_state_L_t_minus_1.command, rng, timestep)
        model_input = ModelInput(
            obs=obs_normalizer(obs_L_t),
            command=cmd_normalizer(command_L_t),
            action_history=None,
            recurrent_state=None,
        )
        prediction = agent.actor_model.forward(model_input)
        action_L_t = agent.action_distribution.sample(rng, prediction)

        mjx_data_L_t_plus_1 = self.apply_physics_steps(
            mjx_model_L=mjx_model_L,
            mjx_data_L=mjx_data_L_t,
            previous_action_L=env_state_L_t_minus_1.action,
            current_action_L=action_L_t,
            num_latency_steps=latency_steps,
        )

        term_components_L_t = self.get_terminations(mjx_data_L_t_plus_1)
        done_L_t = jnp.stack([v for _, v in term_components_L_t.items()]).any()
        reward_components_L_t = self.get_rewards(
            env_state_L_t_minus_1.action,
            mjx_data_L_t,
            command_L_t,
            action_L_t,
            mjx_data_L_t_plus_1,
        )
        reward_L_t = jnp.stack([v for _, v in reward_components_L_t.items()]).sum()

        env_state_L_t = EnvState(
            obs=obs_L_t,
            command=command_L_t,
            action=action_L_t,
            reward=reward_L_t,
            done=done_L_t,
            timestep=timestep,
            termination_components=term_components_L_t,
            reward_components=reward_components_L_t,
        )

        return env_state_L_t, mjx_data_L_t_plus_1

    @xax.profile
    @xax.jit(static_argnames=["self", "return_intermediate_data"])
    def scannable_step_with_automatic_reset(
        self,
        carry: tuple[EnvState, mjx.Data, PRNGKeyArray],
        _: None,
        *,
        agent: Agent,
        physics_model_L: mjx.Model,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
        return_intermediate_data: bool = False,
    ) -> tuple[tuple[EnvState, mjx.Data, PRNGKeyArray], tuple[EnvState, mjx.Data | None, Array]]:
        """Steps the environment and resets if needed."""
        env_state_L_t_minus_1, mjx_data_L_t, rng = carry
        reset_env_state_L_t, reset_mjx_data_L_t_plus_1 = self.reset(
            agent=agent,
            rng=rng,
            physics_model_L=physics_model_L,
            obs_normalizer=obs_normalizer,
            cmd_normalizer=cmd_normalizer,
        )

        step_env_state_L_t, step_mjx_data_L_t_plus_1 = self.step(
            agent=agent,
            env_state_L_t_minus_1=env_state_L_t_minus_1,
            rng=rng,
            physics_data_L_t=mjx_data_L_t,
            physics_model_L=physics_model_L,
            obs_normalizer=obs_normalizer,
            cmd_normalizer=cmd_normalizer,
        )

        data_has_nans = jax.tree_util.tree_reduce(
            lambda a, b: jnp.logical_or(a, b),
            jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), step_mjx_data_L_t_plus_1),
        )

        do_reset = jnp.logical_or(env_state_L_t_minus_1.done, data_has_nans)

        env_state_L_t = jax.tree_util.tree_map(
            lambda r, s: jax.lax.select(do_reset, r, s),
            reset_env_state_L_t,
            step_env_state_L_t,
        )
        mjx_data_L_t_plus_1 = jax.tree_util.tree_map(
            lambda r, s: jax.lax.select(do_reset, r, s),
            reset_mjx_data_L_t_plus_1,
            step_mjx_data_L_t_plus_1,
        )

        rng = jax.random.split(rng)[0]

        if return_intermediate_data:
            return (env_state_L_t, mjx_data_L_t_plus_1, rng), (
                env_state_L_t,
                mjx_data_L_t_plus_1,
                do_reset,
            )
        else:
            return (env_state_L_t, mjx_data_L_t_plus_1, rng), (env_state_L_t, None, data_has_nans)

    @xax.profile
    @xax.jit(static_argnames=["self"])
    def apply_post_accumulate(self, env_state_TL: EnvState) -> EnvState:
        """Apply post_accumulate to all reward components for a single trajectory."""

        # Create a new reward_components dict with post-accumulated values
        updated_reward_components = {}

        for reward_name, reward_func in self.rewards:
            # Extract the reward component for this reward function
            reward_component = env_state_TL.reward_components[reward_name]

            # Apply post_accumulate to the reward component
            updated_reward = reward_func.post_accumulate(reward_component, env_state_TL.done)

            # Store the updated reward
            updated_reward_components[reward_name] = updated_reward

        # Recalculate the total reward based on updated components
        updated_total_reward = jnp.stack([v for _, v in updated_reward_components.items()]).sum(
            axis=0
        )

        # Create a new EnvState with the updated reward components and total reward
        return EnvState(
            obs=env_state_TL.obs,
            command=env_state_TL.command,
            action=env_state_TL.action,
            reward=updated_total_reward,
            done=env_state_TL.done,
            timestep=env_state_TL.timestep,
            termination_components=env_state_TL.termination_components,
            reward_components=FrozenDict(updated_reward_components),
        )

    @xax.profile
    def unroll_trajectory(
        self,
        agent: Agent,
        rng: PRNGKeyArray,
        num_steps: int,
        env_state_L_t_minus_1: EnvState,
        physics_data_L_t: mjx.Data,
        physics_model_L: mjx.Model,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
        return_intermediate_data: bool = False,
    ) -> tuple[EnvState, mjx.Data, Array]:
        """Returns EnvState rollout, final mjx.Data, and mjx.Data rollout."""
        step_fn = functools.partial(
            self.scannable_step_with_automatic_reset,
            agent=agent,
            physics_model_L=physics_model_L,
            return_intermediate_data=return_intermediate_data,
            obs_normalizer=obs_normalizer,
            cmd_normalizer=cmd_normalizer,
        )

        carry = (env_state_L_t_minus_1, physics_data_L_t, rng)
        (_, final_mjx_data_L_f_plus_1, _), (env_state_TL, mjx_data_TL, has_nans_TL) = jax.lax.scan(
            f=step_fn,
            init=carry,
            xs=None,
            length=num_steps,
        )

        # Apply post accumulation to the trajectory
        env_state_TL = self.apply_post_accumulate(env_state_TL)

        if return_intermediate_data:
            assert isinstance(mjx_data_TL, mjx.Data)
            return env_state_TL, mjx_data_TL, has_nans_TL
        else:
            return env_state_TL, final_mjx_data_L_f_plus_1, has_nans_TL

    @xax.profile
    def unroll_trajectories(
        self,
        agent: Agent,
        rng: PRNGKeyArray,
        num_steps: int,
        num_envs: int,
        env_state_EL_t_minus_1: EnvState,
        physics_data_EL_t: mjx.Data,
        physics_model_L: mjx.Model,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
        return_intermediate_data: bool = False,
    ) -> tuple[EnvState, mjx.Data, Array]:
        """Returns EnvState rollout, final / stacked mjx.Data, and array of has_nans flags.

        1. The batched reset (using vmap) initializes a state for each environment.
        2. A vectorized (vmap-ed) env_step function is defined that calls step.
        3. A jax.lax.scan unrolls the trajectory for num_steps.
        4. The resulting trajectory has shape (num_steps, num_envs, ...).

        Note that if `carry_mjx_data` and `carry_mjx_model` are provided, they
        will be used as the initial state and model, respectively. Otherwise,
        the default model and data will be used.
        """
        rng_E = jax.random.split(rng, num_envs)

        env_state_ETL, physics_data_res, has_nans_ETL = jax.vmap(
            self.unroll_trajectory, in_axes=(None, 0, None, 0, 0, None, None, None, None)
        )(
            agent,
            rng_E,
            num_steps,
            env_state_EL_t_minus_1,
            physics_data_EL_t,
            physics_model_L,
            obs_normalizer=obs_normalizer,
            cmd_normalizer=cmd_normalizer,
            return_intermediate_data=return_intermediate_data,
        )

        # Transpose from (env, time, ...) to (time, env, ...)
        # TODO: update GAE to support (env, time, ...) to avoid an extra transpose here
        def transpose_time_and_env_dims(x: Array) -> Array:
            return jnp.transpose(x, (1, 0) + tuple(range(2, x.ndim)))

        env_state_TEL = jax.tree_util.tree_map(transpose_time_and_env_dims, env_state_ETL)

        if return_intermediate_data:
            # Only transpose physics data if it contains trajectory information
            physics_data_res = jax.tree_util.tree_map(transpose_time_and_env_dims, physics_data_res)

        has_nans = jnp.any(has_nans_ETL)

        return env_state_TEL, physics_data_res, has_nans

    @xax.profile
    def render_trajectory(
        self,
        agent: Agent,
        rng: PRNGKeyArray,
        obs_normalizer: Normalizer,
        cmd_normalizer: Normalizer,
        *,
        num_steps: int,
        width: int = 640,
        height: int = 480,
        camera: int | None = None,
    ) -> tuple[list[np.ndarray], EnvState]:
        """Render a trajectory of the environment."""
        physics_model_L = self.get_init_physics_model()
        reset_rngs = jax.random.split(rng, 1)

        env_state_1L_0, physics_data_1L_1 = jax.vmap(
            self.reset, in_axes=(None, 0, None, None, None)
        )(agent, reset_rngs, physics_model_L, obs_normalizer, cmd_normalizer)

        env_state_TEL, traj_data, _ = self.unroll_trajectories(
            agent=agent,
            rng=rng,
            num_steps=num_steps,
            num_envs=1,
            env_state_EL_t_minus_1=env_state_1L_0,
            physics_data_EL_t=physics_data_1L_1,
            physics_model_L=physics_model_L,
            obs_normalizer=obs_normalizer,
            cmd_normalizer=cmd_normalizer,
            return_intermediate_data=True,
        )

        mjx_data_traj = jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=1), traj_data)

        mjx_data_list = [
            jax.tree_util.tree_map(lambda x: x[i], mjx_data_traj) for i in range(num_steps)
        ]

        render_mj_data = mujoco.MjData(self.default_mj_model)

        def render_frame(renderer: mujoco.Renderer, mjx_data: mjx.Data, camera: int) -> np.ndarray:
            # Create fresh MjData for each frame
            render_mj_data.qpos, render_mj_data.qvel = mjx_data.qpos, mjx_data.qvel
            render_mj_data.mocap_pos, render_mj_data.mocap_quat = (
                mjx_data.mocap_pos,
                mjx_data.mocap_quat,
            )
            render_mj_data.xfrc_applied = mjx_data.xfrc_applied

            # Ensure physics state is fully updated
            mujoco.mj_forward(self.default_mj_model, render_mj_data)

            # Update scene and render
            renderer.update_scene(render_mj_data, camera=camera, scene_option=scene_option)
            return renderer.render()

        camera_id = camera or 0

        renderer = mujoco.Renderer(self.default_mj_model, height=height, width=width)
        scene_option = mujoco.MjvOption()
        frames = []
        for data in mjx_data_list:
            frame = render_frame(renderer, data, camera_id)
            frames.append(frame)
        renderer.close()

        return frames, env_state_TEL
