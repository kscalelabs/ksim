"""Defines the default humanoid environment."""

import asyncio
import itertools
import logging
import pickle as pkl
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Literal, Protocol, TypeVar, cast, get_args

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mediapy
import numpy as np
import xax
from brax.base import State, System
from brax.envs.base import PipelineEnv, State as BraxState
from jaxtyping import PRNGKeyArray
from kscale import K
from mujoco_scenes.brax import load_model
from mujoco_scenes.mjcf import load_mjmodel
from omegaconf import MISSING

from ksim.observation.base import Observation, ObservationBuilder
from ksim.resets.base import Reset, ResetBuilder, ResetData
from ksim.rewards.base import Reward, RewardBuilder
from ksim.terminations.base import Termination, TerminationBuilder

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_CAMERA = "tracking_camera"


async def get_model_path(model_name: str, cache: bool = True) -> str | Path:
    async with K() as api:
        urdf_dir = await api.download_and_extract_urdf(model_name, cache=cache)

    try:
        mjcf_path = next(urdf_dir.glob("*.mjcf"))
    except StopIteration:
        raise ValueError(f"No MJCF file found for {model_name} (in {urdf_dir})")

    return mjcf_path


def _unique_dict(things: list[tuple[str, T]]) -> OrderedDict[str, T]:
    return_dict = OrderedDict()
    for base_name, thing in things:
        name, idx = base_name, 1
        while name in return_dict:
            idx += 1
            name = f"{base_name}_{idx}"
        return_dict[name] = thing
    return return_dict


class ActionModel(Protocol):
    def __call__(
        self,
        sys: System,
        state: BraxState,
        rng: PRNGKeyArray,
        carry: T | None,
    ) -> tuple[jnp.ndarray, T]: ...


def get_random_action(sys: System, state: BraxState, rng: PRNGKeyArray, carry: None) -> tuple[jnp.ndarray, None]:
    ctrl_range = sys.actuator.ctrl_range
    ctrl_min, ctrl_max = ctrl_range.T
    action_scale = jax.random.uniform(rng, shape=ctrl_min.shape, dtype=ctrl_min.dtype)
    ctrl = ctrl_min + (ctrl_max - ctrl_min) * action_scale
    return ctrl, None


def get_midpoint_action(sys: System, state: BraxState, rng: PRNGKeyArray, carry: None) -> tuple[jnp.ndarray, None]:
    ctrl_range = sys.actuator.ctrl_range
    ctrl_min, ctrl_max = ctrl_range.T
    ctrl = (ctrl_min + ctrl_max) / 2
    return ctrl, None


def get_zero_action(sys: System, state: BraxState, rng: PRNGKeyArray, carry: None) -> tuple[jnp.ndarray, None]:
    ctrl = jnp.zeros_like(sys.actuator.ctrl_range[..., 0])
    return ctrl, None


ActionModelType = Literal["random", "zero", "midpoint"]


def cast_action_type(action_type: str) -> ActionModelType:
    options = get_args(ActionModelType)
    if action_type not in options:
        raise ValueError(f"Invalid action type: {action_type} Choices are {options}")
    return cast(ActionModelType, action_type)


@jax.tree_util.register_dataclass
@dataclass
class KScaleEnvConfig(xax.Config):
    # Model configuration options.
    model_name: str = xax.field(
        value=MISSING,
        help="The name of the model to use.",
    )
    model_scene: str = xax.field(
        value="smooth",
        help="The scene to use for the model.",
    )

    # Environment configuration options.
    dt: float = xax.field(
        value=0.004,
        help="Simulation time step.",
    )
    ctrl_dt: float = xax.field(
        value=0.02,
        help="Control time step.",
    )
    debug_env: bool = xax.field(
        value=False,
        help="Whether to enable debug mode for the environment.",
    )
    backend: str = xax.field(
        value="mjx",
        help="The backend to use for the environment.",
    )

    # Solver configuration options.
    solver_iterations: int = xax.field(
        value=6,
        help="Number of main solver iterations.",
    )
    solver_ls_iterations: int = xax.field(
        value=6,
        help="Number of line search iterations.",
    )

    # Simulation artifact options.
    ignore_cached_urdf: bool = xax.field(
        value=False,
        help="Whether to ignore the cached URDF.",
    )


class KScaleEnv(PipelineEnv):
    """Defines a generic environment for interacting with K-Scale models."""

    def __init__(
        self,
        config: KScaleEnvConfig,
        terminations: Collection[Termination | TerminationBuilder],
        resets: Collection[Reset | ResetBuilder],
        rewards: Collection[Reward | RewardBuilder],
        observations: Collection[Observation | ObservationBuilder],
    ) -> None:
        self.config = config

        # Downloads the model from the K-Scale API and loads it into MuJoCo.
        model_path = str(
            asyncio.run(
                get_model_path(
                    model_name=self.config.model_name,
                    cache=not self.config.ignore_cached_urdf,
                )
            )
        )

        logger.info("Loading model %s", model_path)
        mj_model = load_mjmodel(model_path, self.config.model_scene)

        # Builds the terminations, resets, rewards, and observations.
        terminations_impl = [t(mj_model) if isinstance(t, TerminationBuilder) else t for t in terminations]
        resets_impl = [r(mj_model) if isinstance(r, ResetBuilder) else r for r in resets]
        rewards_impl = [r(mj_model) if isinstance(r, RewardBuilder) else r for r in rewards]
        observations_impl = [o(mj_model) if isinstance(o, ObservationBuilder) else o for o in observations]

        # Creates dictionaries of the unique terminations, resets, rewards, and observations.
        self.terminations = _unique_dict([(term.termination_name, term) for term in terminations_impl])
        self.resets = _unique_dict([(reset.reset_name, reset) for reset in resets_impl])
        self.rewards = _unique_dict([(reward.reward_name, reward) for reward in rewards_impl])
        self.observations = _unique_dict([(obs.observation_name, obs) for obs in observations_impl])

        logger.info("Converting model to Brax system")
        sys = load_model(mj_model)

        sys = sys.tree_replace(
            {
                "opt.timestep": self.config.dt,
                "opt.iterations": self.config.solver_iterations,
                "opt.ls_iterations": self.config.solver_ls_iterations,
            }
        )

        logger.info("Initializing pipeline")
        super().__init__(
            sys=sys,
            backend=self.config.backend,
            n_frames=round(self.config.ctrl_dt / self.config.dt),
            debug=self.config.debug_env,
        )

    def _pipeline_state_to_state(self, pipeline_state: State) -> BraxState:
        obs = self.get_observation(pipeline_state)
        all_rewards = self.get_rewards(pipeline_state)
        all_dones = self.get_terminations(pipeline_state)

        done = jnp.stack(list(all_dones.values()), axis=-1).any(axis=-1)
        reward = jnp.stack(list(all_rewards.values()), axis=-1).sum(axis=-1)

        # Keep track of all rewards separately.
        metrics = {**all_rewards, **all_dones}

        return BraxState(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    @eqx.filter_jit
    def reset(self, rng: PRNGKeyArray) -> BraxState:
        q = self.sys.init_q
        qd = jnp.zeros(self.sys.qd_size())
        pipeline_state = self.pipeline_init(q, qd)
        reset_data = ResetData(rng=rng, state=pipeline_state)
        for reset_func in self.resets.values():
            reset_data = reset_func(reset_data)
        return self._pipeline_state_to_state(reset_data.state)

    @eqx.filter_jit
    def step(self, state: BraxState, action: jnp.ndarray) -> BraxState:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        return self._pipeline_state_to_state(pipeline_state)

    @eqx.filter_jit
    def get_observation(self, pipeline_state: State) -> dict[str, jnp.ndarray]:
        observations: dict[str, jnp.ndarray] = {}
        for observation_name, observation in self.observations.items():
            observations[observation_name] = observation(pipeline_state)
        return observations

    @eqx.filter_jit
    def get_rewards(self, pipeline_state: State) -> dict[str, jnp.ndarray]:
        rewards: dict[str, jnp.ndarray] = {}
        for reward_name, reward in self.rewards.items():
            rewards[reward_name] = reward(pipeline_state)
        return rewards

    @eqx.filter_jit
    def get_terminations(self, pipeline_state: State) -> dict[str, jnp.ndarray]:
        terminations: dict[str, jnp.ndarray] = {}
        for termination_name, termination in self.terminations.items():
            terminations[termination_name] = termination(pipeline_state)
        return terminations

    @eqx.filter_jit
    def unroll_trajectory(
        self,
        num_steps: int,
        rng: PRNGKeyArray,
        model: ActionModel,
    ) -> BraxState:
        """Unrolls a trajectory for num_steps steps.

        Returns:
            A tuple of (initial_state, trajectory_states) where trajectory_states
            contains the states for steps 1 to num_steps.
        """
        init_state: BraxState = self.reset(rng)

        def identity_fn(state: BraxState, rng: PRNGKeyArray, carry_model: T | None) -> tuple[BraxState, T | None]:
            return state, rng, carry_model

        def step_fn(state: BraxState, rng: PRNGKeyArray, carry_model: T | None) -> tuple[BraxState, T | None]:
            rng, step_rng = jax.random.split(rng)
            action, carry_model = model(self.sys, state, step_rng, carry_model)
            next_state = self.step(state, action)
            return next_state, rng, carry_model

        def scan_fn(
            carry: tuple[BraxState, PRNGKeyArray, T | None],
            _: None,
        ) -> tuple[tuple[BraxState, PRNGKeyArray, T | None], BraxState]:
            state, rng, carry_model = carry
            next_state, rng, carry_model = jax.lax.cond(state.done.all(), identity_fn, step_fn, state, rng, carry_model)
            return (next_state, rng, carry_model), next_state

        # Initialize carry tuple with initial state, RNG, and None for model carry
        init_carry = (init_state, rng, None)

        # Runs the scan function.
        _, states = jax.lax.scan(scan_fn, init_carry, length=num_steps)

        return states

    def unroll_trajectory_and_render(
        self,
        num_steps: int,
        render_dir: str | Path | None = None,
        seed: int = 0,
        camera: str | None = DEFAULT_CAMERA,
        actions: ActionModelType | ActionModel = "zero",
    ) -> list[BraxState]:
        logger.info("Running test run for %d steps", num_steps)

        # Converts the shorthand function names to callable functions.
        if isinstance(actions, str):
            match actions:
                case "random":
                    actions = get_random_action
                case "zero":
                    actions = get_zero_action
                case "midpoint":
                    actions = get_midpoint_action
                case _:
                    raise ValueError(f"Invalid action type: {actions}")
        elif not isinstance(actions, ActionModel):
            raise ValueError(f"Invalid action type: {type(actions)}")

        # Run simulation
        rng = jax.random.PRNGKey(seed)
        trajectory = self.unroll_trajectory(num_steps, rng, actions)

        # Remove all the trajectory states after the episode finished.
        done = trajectory.done
        done = jnp.pad(done[:-1], (1, 0), mode="constant", constant_values=False)
        trajectory = jax.tree.map(lambda x: x[~done], trajectory)

        # Render if requested
        if render_dir is not None:
            (render_dir := Path(render_dir)).mkdir(parents=True, exist_ok=True)
            raw_trajectory = jax.tree.map(lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, trajectory)

            # Dumps the raw trajectory.
            with open(render_dir / "trajectory.pkl", "wb") as f:
                pkl.dump(raw_trajectory, f)

            metrics: dict[str, np.ndarray] = {
                key: raw_trajectory.metrics[key]
                for key in itertools.chain(self.terminations.keys(), self.rewards.keys())
            }

            # Plot against real-time.
            num_steps = len(raw_trajectory.done)
            t = np.arange(num_steps) * self.config.ctrl_dt

            # Plots the metrics.
            for key, metric in metrics.items():
                plt.figure()
                plt.plot(t, metric.astype(np.float32))
                plt.title(key)
                plt.xlabel("Time (s)")
                plt.ylabel(key)
                render_path = render_dir / f"{key}.png"
                plt.savefig(render_path)
                plt.close()
                logger.info("Saved %s", render_path)

            # Renders a video of the trajectory.
            render_path = render_dir / "render.mp4"
            fps = round(1 / self.config.ctrl_dt)
            pipeline_states = [jax.tree.map(lambda arr: arr[i], trajectory.pipeline_state) for i in range(num_steps)]
            frames = np.stack(self.render(pipeline_states, camera=camera), axis=0)
            mediapy.write_video(render_path, frames, fps=fps)

        return trajectory
