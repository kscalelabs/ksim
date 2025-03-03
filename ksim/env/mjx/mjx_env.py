"""The environment used to run massively parallel rollouts.

Philosophy:
- Think of the environment as a function of the state, system, action, and rng.
- Rollouts are performed by vectorizing (vmap) the reset and step functions, with a final trajectory
  of shape (time, num_envs, *obs_shape/s).

Rollouts return a trajectory of shape (time, num_envs, ).
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Tuple, TypeVar, cast, get_args

import chex
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import xax
from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray, PyTree
from mujoco import mjx
from mujoco_scenes.mjcf import load_mjmodel
from omegaconf import MISSING

from ksim.builders.commands import Command, CommandBuilder
from ksim.builders.observation import Observation, ObservationBuilder
from ksim.builders.resets import Reset, ResetBuilder
from ksim.builders.rewards import Reward, RewardBuilder
from ksim.builders.terminations import Termination, TerminationBuilder
from ksim.env.base_env import BaseEnv, BaseEnvConfig, EnvState
from ksim.env.mjx.actuators.mit_actuator import MITPositionActuators
from ksim.env.types import EnvState, KScaleActionModelType
from ksim.model.formulations import ActionModel, ActorCriticModel
from ksim.utils.data import BuilderData
from ksim.utils.jit import legit_jit
from ksim.utils.mujoco import make_mujoco_mappings
from ksim.utils.robot_model import get_model_and_metadata

logger = logging.getLogger(__name__)

T = TypeVar("T")


@legit_jit()
def step_mjx(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    ctrl: Array,
) -> mjx.Data:
    """Step the mujoco model."""
    data_with_ctrl = mjx_data.replace(ctrl=ctrl)
    # more logic if needed...
    return mjx.step(mjx_model, data_with_ctrl)


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


# TODO: add these back in.
@legit_jit()
def get_random_action(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    rng: PRNGKeyArray,
) -> tuple[jnp.ndarray, float]:
    """Get a random action."""
    ctrl_range = mjx_model.actuator_ctrlrange
    ctrl_min, ctrl_max = ctrl_range.T
    action_scale = jax.random.uniform(rng, shape=ctrl_min.shape, dtype=ctrl_min.dtype)
    ctrl = ctrl_min + (ctrl_max - ctrl_min) * action_scale
    return ctrl, 1.0


@legit_jit()
def get_midpoint_action(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    rng: PRNGKeyArray,
) -> tuple[jnp.ndarray, float]:
    """Get a midpoint action."""
    ctrl_range = mjx_model.actuator_ctrlrange
    ctrl_min, ctrl_max = ctrl_range.T
    ctrl = (ctrl_min + ctrl_max) / 2
    return ctrl, 1.0


@legit_jit()
def get_zero_action(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    rng: PRNGKeyArray,
) -> tuple[jnp.ndarray, float]:
    """Get a zero action."""
    ctrl = jnp.zeros_like(mjx_model.actuator_ctrlrange[..., 0])
    return ctrl, 1.0


def cast_action_type(action_type: str) -> KScaleActionModelType:
    """Cast the action type to the correct type."""
    options = get_args(KScaleActionModelType)
    if action_type not in options:
        raise ValueError(f"Invalid action type: {action_type} Choices are {options}")
    return cast(KScaleActionModelType, action_type)


def get_action_fn(
    action_type: KScaleActionModelType,
) -> Callable[[mjx.Model, mjx.Data, PRNGKeyArray], tuple[jnp.ndarray, float]]:
    """Get the action function for the given action type."""
    match action_type:
        case "random":
            return get_random_action
        case "midpoint":
            return get_midpoint_action
        case "zero":
            return get_zero_action
        case _:
            raise ValueError(f"Invalid action type: {action_type}")


@jax.tree_util.register_dataclass
@dataclass
class MjxEnvConfig(BaseEnvConfig):
    # environment configuration options
    dt: float = xax.field(value=0.004, help="Simulation time step.")
    ctrl_dt: float = xax.field(value=0.02, help="Control time step.")
    debug_env: bool = xax.field(value=False, help="Whether to enable debug mode for the env.")

    # action configuration options
    min_action_latency: float = xax.field(value=0.0, help="The minimum action latency.")
    max_action_latency: float = xax.field(value=0.0, help="The maximum action latency.")

    # solver configuration options
    solver_iterations: int = xax.field(value=6, help="Number of main solver iterations.")
    solver_ls_iterations: int = xax.field(value=6, help="Number of line search iterations.")

    # simulation artifact options
    ignore_cached_urdf: bool = xax.field(value=False, help="Whether to ignore the cached URDF.")


# The new stateless environment – note that we do not call any stateful methods.
class MjxEnv(BaseEnv):
    """An environment for massively parallel rollouts, stateless to obj state and system parameters.

    In this design:
      - All state (a EnvState) is passed in and returned by reset and step.
      - The underlying Mujoco model (here referred to as `mjx_model`) is provided to step/reset.
      - Rollouts are performed by vectorizing (vmap) the reset and step functions,
        with a final trajectory of shape (time, num_envs, ...).
      - The step wrapper only computes a reset (via jax.lax.cond) if the done flag is True.
    """

    def __init__(
        self,
        config: MjxEnvConfig,
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

        # getting the robot model and metadata
        robot_model_path, robot_model_metadata = asyncio.run(
            get_model_and_metadata(
                self.config.robot_model_name,
                cache=not self.config.ignore_cached_urdf,
            )
        )

        logger.info("Loading robot model %s", robot_model_path)
        mj_model = load_mjmodel(robot_model_path, self.config.robot_model_scene)
        self.default_mj_model = mj_model
        self.default_mj_data = mujoco.MjData(mj_model)
        self.default_mjx_model = mjx.put_model(mj_model)
        self.default_mjx_data = mjx.make_data(self.default_mjx_model)
        self.mujoco_mappings = make_mujoco_mappings(self.default_mjx_model)
        self.actuators = MITPositionActuators(
            actuators_metadata=robot_model_metadata.actuators,
            mujoco_mappings=self.mujoco_mappings,
        )

        # preparing builder data.
        data = BuilderData(
            model=self.default_mjx_model,
            dt=self.config.dt,
            ctrl_dt=self.config.ctrl_dt,
            mujoco_mappings=self.mujoco_mappings,
        )

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
        self._expected_dt_per_ctrl_dt = int(self.config.ctrl_dt / self.config.dt)

    ###################
    # Post Processing #
    ###################
    # TODO make these jittable...

    @legit_jit(static_argnames=["self"])
    def get_observation(self, mjx_data: mjx.Data, rng: jax.Array) -> FrozenDict[str, Array]:
        """Compute observations from the pipeline state."""
        observations = {}
        for observation_name, observation in self.observations:
            rng, obs_rng = jax.random.split(rng)
            observation_value = observation(mjx_data, obs_rng)
            observations[observation_name] = observation_value
        return FrozenDict(observations)

    def get_rewards(
        self,
        action_t_minus_1: Array,
        mjx_data_t: mjx.Data,
        command_t: FrozenDict[str, Array],
        action_t: Array,
        mjx_data_t_plus_1: mjx.Data,
    ) -> list[tuple[str, float]]:
        """Compute rewards (each as a scalar) from the state transition.

        ML: we might want to represent rewards as graphs (multiply and sum) or add flags...
        """
        rewards = []  # this ensures ordering...
        for reward_name, reward in self.rewards:
            reward_val = (
                reward(
                    action_t_minus_1=action_t_minus_1,
                    mjx_data_t=mjx_data_t,
                    command_t=command_t,
                    action_t=action_t,
                    mjx_data_t_plus_1=mjx_data_t_plus_1,
                )
                * reward.scale
            )
            chex.assert_shape(
                reward_val, (), custom_message=f"Reward {reward_name} must be a scalar"
            )
            rewards.append((reward_name, reward_val))
        return rewards

    def get_terminations(self, new_mjx_data: mjx.Data) -> list[tuple[str, float]]:
        """Compute termination conditions (each as a scalar) from the pipeline state."""
        terminations = []
        for termination_name, termination in self.terminations:
            term_val = termination(new_mjx_data)
            chex.assert_shape(
                term_val, (), custom_message=f"Termination {termination_name} must be a scalar"
            )
            terminations.append((termination_name, term_val))
        return terminations

    @legit_jit(static_argnames=["self"])
    def get_initial_commands(
        self, rng: PRNGKeyArray, initial_time: Array | None
    ) -> FrozenDict[str, Array]:
        """Compute initial commands from the pipeline state. Assumes consistent ordering."""
        commands = {}
        if initial_time is None:
            initial_time = jnp.array(0.0)
        for command_name, command_def in self.commands:
            assert isinstance(command_def, Command)
            command_val = command_def(rng, initial_time)
            commands[command_name] = command_val
        return FrozenDict(commands)

    @legit_jit(static_argnames=["self"])
    def get_commands(
        self, prev_commands: FrozenDict[str, Array], rng: PRNGKeyArray, time: Array
    ) -> FrozenDict[str, Array]:
        """Compute commands from the pipeline state. Assumes consistent ordering."""
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

    @legit_jit(static_argnames=["self"])
    def _apply_physics_steps(
        self,
        mjx_model: mjx.Model,
        mjx_data: mjx.Data,
        previous_action: Array,
        current_action: Array,
        num_latency_steps: Array,
    ) -> mjx.Data:
        """A 'step' of the environment on a state composes multiple steps of the actual physics.

        We take num_latency_steps (continuing the feedback loop for the previous action), then we
        apply the feedback loop for the current action for the remainder of the physics steps.
        """
        n_steps = self._expected_dt_per_ctrl_dt  # total number of pipeline steps to take.

        def f(carry: Tuple[mjx.Data, int], _: Any) -> Tuple[Tuple[mjx.Data, int], None]:
            state, step_num = carry

            action_motor_sees = jax.lax.select(
                step_num >= num_latency_steps, current_action, previous_action
            )
            torques = self.actuators.get_ctrl(state, action_motor_sees)

            # NOTE: can extend state to include anything from `mjx.Data` here...
            new_state = step_mjx(mjx_model, state, torques)
            return (new_state, step_num + 1), None

        (state, _), _ = jax.lax.scan(f, (mjx_data, 0), None, n_steps)
        return state

    @legit_jit(static_argnames=["self", "model"])
    def scannable_reset(
        self,
        model: ActorCriticModel,
        params: PyTree,
        rng: jax.Array,
        mjx_model: mjx.Model,
    ) -> tuple[EnvState, mjx.Data]:
        """A scannable reset function: returns an initial state computed solely from the inputs.

        NOTE: Reset actually takes a step. Trust this makes sense :)
        """
        reset_data = self.default_mjx_data

        for _, reset_func in self.resets:
            reset_data = reset_func(reset_data, rng)
        assert isinstance(reset_data, mjx.Data)

        rng, obs_rng = jax.random.split(rng, 2)
        mjx_data_0 = step_mjx(
            mjx_model=mjx_model,
            mjx_data=reset_data,
            ctrl=jnp.zeros_like(reset_data.ctrl),  # NOTE: the MJCF MUST be torque controlled.
        )

        timestep = mjx_data_0.time
        obs_0 = self.get_observation(mjx_data_0, obs_rng)
        command_0 = self.get_initial_commands(rng, timestep)

        # TODO: when we add in historical context to the model, we need to handle burn-in.
        action_0, action_log_prob_0 = model.apply(
            params, obs_0, command_0, rng, method="actor_sample_and_log_prob"
        )
        assert isinstance(action_log_prob_0, Array)

        mjx_data_1 = self._apply_physics_steps(
            mjx_model=mjx_model,
            mjx_data=mjx_data_0,
            previous_action=action_0,  # NOTE: Effectively means no latency for first action.
            current_action=action_0,
            num_latency_steps=jnp.array(0),  # Enforced here as well...
        )

        done_0 = jnp.array(False, dtype=jnp.bool_)
        reward_0 = jnp.array(0.0)

        return (
            EnvState(
                obs=obs_0,
                reward=reward_0,
                done=done_0,
                command=command_0,
                action=action_0,
                timestep=timestep,
            ),
            mjx_data_1,
        )

    @legit_jit(static_argnames=["self", "model"])
    def scannable_step(
        self,
        model: ActorCriticModel,
        params: PyTree,
        env_state_t_minus_1: EnvState,
        mjx_data_t: mjx.Data,
        mjx_model: mjx.Model,
        rng: PRNGKeyArray,
    ) -> tuple[EnvState, mjx.Data]:
        """A scannable step function: returns a new state computed solely from the inputs."""
        rng, latency_rng, obs_rng = jax.random.split(rng, 3)
        timestep = mjx_data_t.time
        latency_steps = jax.random.randint(
            key=latency_rng,
            shape=(),
            minval=self.min_action_latency_step,
            maxval=self.max_action_latency_step,
        )

        obs_t = self.get_observation(mjx_data_t, obs_rng)
        command_t = self.get_commands(env_state_t_minus_1.command, rng, timestep)
        action_t, _ = model.apply(params, obs_t, command_t, rng, method="actor_sample_and_log_prob")

        mjx_data_t_plus_1 = self._apply_physics_steps(
            mjx_model=mjx_model,
            mjx_data=mjx_data_t,
            previous_action=env_state_t_minus_1.action,
            current_action=action_t,
            num_latency_steps=latency_steps,
        )

        all_dones = self.get_terminations(mjx_data_t_plus_1)
        done_t = jnp.stack([v for _, v in all_dones]).any()
        all_rewards = self.get_rewards(
            env_state_t_minus_1.action, mjx_data_t, command_t, action_t, mjx_data_t_plus_1
        )
        reward_t = jnp.stack([v for _, v in all_rewards]).sum()

        env_state_t = EnvState(
            obs=obs_t,
            command=command_t,
            action=action_t,
            reward=reward_t,
            done=done_t,
            timestep=timestep,
        )

        return env_state_t, mjx_data_t_plus_1

    ###########################
    # Main API Implementation #
    ###########################

    @legit_jit(static_argnames=["self"])
    def get_dummy_env_state(
        self,
        rng: PRNGKeyArray,
    ) -> EnvState:
        """Get a dummy environment state for compilation purposes."""
        rng, obs_rng = jax.random.split(rng, 2)
        mjx_data_0 = step_mjx(
            mjx_model=self.default_mjx_model,
            mjx_data=self.default_mjx_data,
            ctrl=jnp.zeros_like(
                self.default_mjx_data.ctrl
            ),  # NOTE: the MJCF MUST be torque controlled.
        )

        timestep = mjx_data_0.time
        obs_dummy = self.get_observation(mjx_data_0, obs_rng)
        command_dummy = self.get_initial_commands(rng, timestep)
        return EnvState(
            obs=obs_dummy,
            command=command_dummy,
            action=jnp.ones(self.action_size),
            reward=jnp.ones(()),
            done=jnp.ones(()),
            timestep=jnp.ones(()),
        )

    @legit_jit(static_argnames=["self"])
    def reset(
        self,
        model: ActorCriticModel,
        params: PyTree,
        rng: jax.Array,
        *,
        mjx_model: mjx.Model,
    ) -> EnvState:
        """Pure reset function: returns an initial state computed solely from the inputs.

        We couple the step and actor because we couple the actions with the rest of env state. This
        ultimately allows for extremely constrained `EnvState`s, which promote correct RL code.
        """
        state, _ = self.scannable_reset(model=model, params=params, rng=rng, mjx_model=mjx_model)
        return state

    @legit_jit(static_argnames=["self", "model"])
    def step(
        self,
        model: ActorCriticModel,
        params: PyTree,
        prev_env_state: EnvState,
        rng: PRNGKeyArray,
        *,
        mjx_data: mjx.Data,
        mjx_model: mjx.Model,
    ) -> EnvState:
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
        new_state, _ = self.scannable_step(
            model=model,
            params=params,
            env_state_t_minus_1=prev_env_state,
            mjx_data_t=mjx_data,
            mjx_model=mjx_model,
            rng=rng,
        )
        return new_state

    @legit_jit(static_argnames=["self", "model", "num_steps", "num_envs", "return_data"])
    def unroll_trajectories(
        self,
        model: ActorCriticModel,
        params: PyTree,
        rng: PRNGKeyArray,
        num_steps: int,
        num_envs: int,
        return_data: bool = False,
        **kwargs: Any,
    ) -> tuple[EnvState, mjx.Data]:
        """Vectorized rollout of trajectories.

        1. The batched reset (using vmap) initializes a state for each environment.
        2. A vectorized (vmap-ed) env_step function is defined that calls step.
        3. A jax.lax.scan unrolls the trajectory for num_steps.
        4. The resulting trajectory has shape (num_steps, num_envs, ...).
        """

        if return_data:
            num_envs = 1

        init_rngs = jax.random.split(rng, num_envs)
        mjx_model = self.default_mjx_model
        # TODO: include logic to randomize environment parameters here...
        init_states, init_mjx_data = jax.vmap(
            lambda key: self.scannable_reset(
                model=model,
                params=params,
                rng=key,
                mjx_model=mjx_model,
            )
        )(init_rngs)
        rng, _ = jax.random.split(rng)

        # Define env_step as a pure function with all dependencies passed explicitly
        @legit_jit()
        def env_step(
            env_state: EnvState,
            mjx_data: mjx.Data,
            rng: Array,
        ) -> tuple[EnvState, mjx.Data]:
            reset_result = self.scannable_reset(
                model=model,
                params=params,
                rng=rng,
                mjx_model=mjx_model,
            )
            step_result = self.scannable_step(
                model=model,
                params=params,
                env_state_t_minus_1=env_state,
                mjx_data_t=mjx_data,
                mjx_model=mjx_model,
                rng=rng,
            )

            new_state = jax.tree_util.tree_map(
                lambda r, s: jax.lax.select(env_state.done, r, s), reset_result[0], step_result[0]
            )
            new_mjx_data = jax.tree_util.tree_map(
                lambda r, s: jax.lax.select(env_state.done, r, s), reset_result[1], step_result[1]
            )
            # ML: unintuitive but this is more efficient than `.cond` by keeping consistent path.

            return new_state, new_mjx_data

        # Create a partially applied version with fixed arguments
        env_step_partial = lambda state, data, key: env_step(state, data, key)

        @legit_jit()
        def scan_fn(
            carry: Tuple[EnvState, mjx.Data, Array], _: Any
        ) -> Tuple[Tuple[EnvState, mjx.Data, Array], Tuple[EnvState, mjx.Data]]:
            states, mjx_data, rng = carry
            rngs = jax.random.split(rng, num_envs + 1)
            new_states, new_mjx_data = jax.vmap(env_step_partial)(states, mjx_data, rngs[1:])

            if return_data:
                return (new_states, new_mjx_data, rngs[0]), (new_states, new_mjx_data)
            else:
                return (new_states, new_mjx_data, rngs[0]), (new_states, None)

        (_, _, _), traj = jax.lax.scan(
            f=scan_fn,
            init=(init_states, init_mjx_data, rng),
            xs=None,
            length=num_steps,
        )

        return traj  # Shape: (num_steps, num_envs, ...)

    def render_trajectory(
        self,
        trajectory: list[mjx.Data],
        width: int = 640,
        height: int = 480,
        camera: int | None = None,
    ) -> list[np.ndarray]:
        def render_frame(renderer: mujoco.Renderer, mjx_data: mjx.Data, camera: int) -> np.ndarray:
            # Create fresh MjData for each frame
            d = self.default_mj_data

            d.qpos, d.qvel = mjx_data.qpos, mjx_data.qvel
            d.mocap_pos, d.mocap_quat = mjx_data.mocap_pos, mjx_data.mocap_quat
            d.xfrc_applied = mjx_data.xfrc_applied

            # Ensure physics state is fully updated
            mujoco.mj_forward(self.default_mj_model, d)

            # Update scene and render
            renderer.update_scene(d, camera=camera, scene_option=scene_option)
            return renderer.render()

        camera_id = camera or 0

        renderer = mujoco.Renderer(self.default_mj_model, height=height, width=width)
        scene_option = mujoco.MjvOption()
        frames = []
        for _, data in trajectory:
            frame = render_frame(renderer, data, camera_id)
            frames.append(frame)
        renderer.close()
        return frames

    @property
    def observation_size(self) -> int:
        raise NotImplementedError("Not implemented yet... need to compile observations?")

    @property
    def action_size(self) -> int:
        return self.actuators.actuator_input_size
