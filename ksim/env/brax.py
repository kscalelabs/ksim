"""Defines the default humanoid environment."""

import asyncio
import io
import logging
import pickle as pkl
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Iterator, Literal, Protocol, TypeVar, cast, get_args

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
from omegaconf import MISSING, DictConfig, OmegaConf
from PIL import Image

from ksim.commands import Command, CommandBuilder
from ksim.observation.base import Observation, ObservationBuilder
from ksim.resets.base import Reset, ResetBuilder, ResetData
from ksim.rewards.base import Reward, RewardBuilder
from ksim.terminations.base import Termination, TerminationBuilder
from ksim.utils.data import BuilderData

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
    def __call__(self, sys: System, state: BraxState, rng: PRNGKeyArray, carry: T) -> tuple[jnp.ndarray, T]: ...


def get_random_action(
    sys: System,
    state: BraxState,
    rng: PRNGKeyArray,
    carry: None,
) -> tuple[jnp.ndarray, None]:
    ctrl_range = sys.actuator.ctrl_range
    ctrl_min, ctrl_max = ctrl_range.T
    action_scale = jax.random.uniform(rng, shape=ctrl_min.shape, dtype=ctrl_min.dtype)
    ctrl = ctrl_min + (ctrl_max - ctrl_min) * action_scale
    return ctrl, None


def get_midpoint_action(
    sys: System,
    state: BraxState,
    rng: PRNGKeyArray,
    carry: None,
) -> tuple[jnp.ndarray, None]:
    ctrl_range = sys.actuator.ctrl_range
    ctrl_min, ctrl_max = ctrl_range.T
    ctrl = (ctrl_min + ctrl_max) / 2
    return ctrl, None


def get_zero_action(
    sys: System,
    state: BraxState,
    rng: PRNGKeyArray,
    carry: None,
) -> tuple[jnp.ndarray, None]:
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

    # Additional environment information. This is populated by the environment
    # when it is created and is only kept here for logging purposes.
    body_name_to_idx: dict[str, int] | None = xax.field(
        value=None,
        help="A mapping from body names to indices.",
    )
    joint_name_to_idx: dict[str, int] | None = xax.field(
        value=None,
        help="A mapping from joint names to indices.",
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
        commands: Collection[Command | CommandBuilder] = (),
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

        # Populates configuration joint information.
        self.body_name_to_idx = {mj_model.body(i).name: i for i in range(mj_model.nbody)}
        self.joint_name_to_idx = {mj_model.joint(i).name: i for i in range(mj_model.njnt)}

        # Gets the relevant data for building the components of the environment.
        data = BuilderData(
            model=mj_model,
            dt=self.config.dt,
            ctrl_dt=self.config.ctrl_dt,
            body_name_to_idx=self.body_name_to_idx,
            joint_name_to_idx=self.joint_name_to_idx,
        )

        # Builds the terminations, resets, rewards, and observations.
        terminations_impl = [t(data) if isinstance(t, TerminationBuilder) else t for t in terminations]
        resets_impl = [r(data) if isinstance(r, ResetBuilder) else r for r in resets]
        rewards_impl = [r(data) if isinstance(r, RewardBuilder) else r for r in rewards]
        observations_impl = [o(data) if isinstance(o, ObservationBuilder) else o for o in observations]
        commands_impl = [c(data) if isinstance(c, CommandBuilder) else c for c in commands]

        # Creates dictionaries of the unique terminations, resets, rewards, and observations.
        self.terminations = _unique_dict([(term.termination_name, term) for term in terminations_impl])
        self.resets = _unique_dict([(reset.reset_name, reset) for reset in resets_impl])
        self.rewards = _unique_dict([(reward.reward_name, reward) for reward in rewards_impl])
        self.observations = _unique_dict([(obs.observation_name, obs) for obs in observations_impl])
        self.commands = _unique_dict([(cmd.command_name, cmd) for cmd in commands_impl])

        logger.info("Converting model to Brax system")
        sys = load_model(mj_model)

        sys = sys.tree_replace(
            {
                "opt.timestep": self.config.dt,
                "opt.iterations": self.config.solver_iterations,
                "opt.ls_iterations": self.config.solver_ls_iterations,
            }
        )

        super().__init__(
            sys=sys,
            backend=self.config.backend,
            n_frames=round(self.config.ctrl_dt / self.config.dt),
            debug=self.config.debug_env,
        )

    def get_state(self) -> DictConfig:
        return OmegaConf.create(
            {
                "body_name_to_idx": self.body_name_to_idx,
                "joint_name_to_idx": self.joint_name_to_idx,
            },
        )

    @eqx.filter_jit
    def reset(self, rng: PRNGKeyArray) -> BraxState:
        q = self.sys.init_q
        qd = jnp.zeros(self.sys.qd_size())
        pipeline_state = self.pipeline_init(q, qd)
        reset_data = ResetData(rng=rng, state=pipeline_state)
        for reset_func in self.resets.values():
            reset_data = reset_func(reset_data)

        rng, obs_rng = jax.random.split(rng)
        obs = self.get_observation(pipeline_state, rng)
        all_dones = self.get_terminations(pipeline_state)
        all_rewards = OrderedDict([(key, jnp.zeros(())) for key in self.rewards.keys()])

        done = jnp.stack(list(all_dones.values()), axis=-1).any(axis=-1)
        reward = jnp.stack(list(all_rewards.values()), axis=-1)

        metrics = {
            "time": jnp.zeros(()),
            "rng": rng,
        }

        for cmd_name, cmd in self.commands.items():
            obs[cmd_name] = metrics[cmd_name] = cmd(rng)

        return BraxState(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )

    @eqx.filter_jit
    def step(self, prev_state: BraxState, action: jnp.ndarray) -> BraxState:
        pipeline_state = self.pipeline_step(prev_state.pipeline_state, action)

        # Update the metrics.
        time = prev_state.metrics["time"]
        rng = prev_state.metrics["rng"]

        rng, obs_rng = jax.random.split(rng)
        obs = self.get_observation(pipeline_state, obs_rng)
        all_dones = self.get_terminations(pipeline_state)
        all_rewards = self.get_rewards(prev_state, action, pipeline_state)

        done = jnp.stack(list(all_dones.values()), axis=-1).any(axis=-1)
        reward = jnp.stack(list(all_rewards.values()), axis=-1)

        for cmd_name, cmd in self.commands.items():
            rng, cmd_rng = jax.random.split(rng)
            prev_cmd = prev_state.metrics[cmd_name]
            next_cmd = cmd.update(prev_cmd, cmd_rng, time)
            obs[cmd_name] = prev_state.metrics[cmd_name] = next_cmd

        # Update with the new state.
        next_state = prev_state.tree_replace(
            {
                "pipeline_state": pipeline_state,
                "obs": obs,
                "reward": reward,
                "done": done,
            },
        )

        next_state.metrics["time"] = time + self.config.ctrl_dt
        next_state.metrics["rng"] = rng

        return next_state

    @eqx.filter_jit
    def get_observation(
        self,
        pipeline_state: State,
        rng: PRNGKeyArray,
    ) -> OrderedDict[str, jnp.ndarray]:
        observations: OrderedDict[str, jnp.ndarray] = OrderedDict()
        for observation_name, observation in self.observations.items():
            rng, obs_rng = jax.random.split(rng)
            observation_value = observation(pipeline_state)
            observation_value = observation.add_noise(observation_value, obs_rng)
            observations[observation_name] = observation_value
        return observations

    @eqx.filter_jit
    def get_rewards(
        self,
        prev_state: BraxState,
        action: jnp.ndarray,
        pipeline_state: State,
    ) -> OrderedDict[str, jnp.ndarray]:
        rewards: OrderedDict[str, jnp.ndarray] = OrderedDict()
        for reward_name, reward in self.rewards.items():
            reward_val = reward(prev_state, action, pipeline_state)
            rewards[reward_name] = reward_val
        return rewards

    @eqx.filter_jit
    def get_terminations(self, pipeline_state: State) -> OrderedDict[str, jnp.ndarray]:
        terminations: OrderedDict[str, jnp.ndarray] = OrderedDict()
        for termination_name, termination in self.terminations.items():
            term_val = termination(pipeline_state)
            assert term_val.shape == (), f"Termination {termination_name} must be a scalar, got {term_val.shape}"
            terminations[termination_name] = term_val
        return terminations

    @eqx.filter_jit
    def unroll_trajectory(
        self,
        num_steps: int,
        rng: PRNGKeyArray,
        init_carry: T,
        model: ActionModel,
    ) -> BraxState:
        """Unrolls a trajectory for num_st eps steps.

        Returns:
            A tuple of (initial_state, trajectory_states) where trajectory_states
            contains the states for steps 1 to num_steps.
        """
        rng, init_rng = jax.random.split(rng)
        init_state = self.reset(init_rng)

        def identity_fn(
            state: BraxState,
            rng: PRNGKeyArray,
            carry_model: T,
        ) -> tuple[BraxState, PRNGKeyArray, T]:
            # Ensure we return the exact same structure as step_fn
            return (state, rng, carry_model)

        def step_fn(
            state: BraxState,
            rng: PRNGKeyArray,
            carry_model: T,
        ) -> tuple[BraxState, PRNGKeyArray, T]:
            rng, step_rng = jax.random.split(rng)
            action, carry_model = model(sys=self.sys, state=state, rng=step_rng, carry=carry_model)

            # Clamps the action to the range of the action space.
            ctrl_range = self.sys.actuator.ctrl_range
            ctrl_min, ctrl_max = ctrl_range.T
            action = jnp.clip(action, ctrl_min, ctrl_max)

            next_state = self.step(state, action)
            return (next_state, rng, carry_model)  # Explicitly wrap in tuple

        def scan_fn(
            carry: tuple[BraxState, PRNGKeyArray, T],
            _: None,
        ) -> tuple[tuple[BraxState, PRNGKeyArray, T], BraxState]:
            state, rng, carry_model = carry
            # Check if done is a scalar or array
            done_condition = state.done.all()
            next_state, rng, carry_model = jax.lax.cond(
                done_condition,
                lambda x: identity_fn(*x),  # Unpack arguments
                lambda x: step_fn(*x),  # Unpack arguments
                (state, rng, carry_model),  # Pack arguments
            )
            return (next_state, rng, carry_model), next_state

        # Initialize carry tuple with initial state, RNG, and None for model carry
        init_carry = (init_state, rng, init_carry)

        # Runs the scan function.
        _, states = jax.lax.scan(scan_fn, init_carry, length=num_steps)

        # Apply post_accumulate and scale to rewards more efficiently
        rewards = jnp.stack(
            [
                reward_fn.post_accumulate(states.reward[:, i]) * reward_fn.scale
                for i, reward_fn in enumerate(self.rewards.values())
            ],
            axis=1,
        )
        states = states.tree_replace({"reward": rewards})

        return states

    def _plot_trajectory_data(
        self,
        t: np.ndarray,
        data: np.ndarray,
        title: str,
        ylabel: str,
        labels: list[str] | None = None,
        figsize: tuple[int, int] = (12, 12),
    ) -> Image.Image:
        """Helper function to create a plot and return it as a PIL Image."""
        plt.figure(figsize=figsize)
        data_2d = data.reshape(data.shape[0], -1)
        for j, value in enumerate(data_2d.T):
            label = labels[j] if labels is not None else f"Component {j}"
            plt.plot(t, value, label=label)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        if labels is not None:
            plt.legend()
        plt.tight_layout()

        # Convert to PIL image
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        return Image.open(buf)

    def generate_trajectory_plots(
        self,
        trajectory: BraxState,
        figsize: tuple[int, int] = (12, 12),
    ) -> Iterator[tuple[str, Image.Image]]:
        """Generate plots for trajectory data and yield (path, image) pairs.

        Args:
            trajectory: The trajectory to plot.
            dt: The time step of the trajectory.
            commands: The commands to plot.
            rewards: The rewards to plot.
            observations: The observations to plot.
            figsize: The size of the figure to plot.

        Returns:
            An iterator of (name, image) pairs.
        """
        raw_trajectory = jax.tree.map(lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, trajectory)
        num_steps = len(raw_trajectory.done)
        t = np.arange(num_steps) * self.config.ctrl_dt

        # Generate command plots
        for key in self.commands.keys():
            metric = raw_trajectory.metrics[key]
            img = self._plot_trajectory_data(
                t,
                metric,
                title=key,
                ylabel=key,
                labels=[f"Command {j}" for j in range(metric.shape[-1])],
                figsize=figsize,
            )
            yield f"commands/{key}.png", img

        # Generate reward plots
        for i, key in enumerate(self.rewards.keys()):
            reward_data = raw_trajectory.reward[:, i : i + 1].astype(np.float32)
            img = self._plot_trajectory_data(t, reward_data, title=key, ylabel=key, figsize=figsize)
            yield f"rewards/{key}.png", img

        # Generate observation plots
        for key in self.observations.keys():
            obs = raw_trajectory.obs[key]
            img = self._plot_trajectory_data(
                t,
                obs,
                title=key,
                ylabel=key,
                labels=[f"Observation {j}" for j in range(obs.shape[-1])],
                figsize=figsize,
            )
            yield f"observations/{key}.png", img

    def render_trajectory_video(
        self,
        trajectory: BraxState,
        camera: str | None,
    ) -> tuple[np.ndarray, int]:
        """Render trajectory as video frames with computed FPS."""
        num_steps = len(trajectory.done)
        fps = round(1 / self.config.ctrl_dt)
        pipeline_states = [jax.tree.map(lambda arr: arr[i], trajectory.pipeline_state) for i in range(num_steps)]
        frames = np.stack(self.render(pipeline_states, camera=camera), axis=0)
        return frames, fps

    def unroll_trajectory_and_render(
        self,
        num_steps: int,
        render_dir: str | Path | None = None,
        seed: int = 0,
        camera: str | None = DEFAULT_CAMERA,
        actions: ActionModelType | ActionModel = "zero",
        init_carry: T | None = None,
        figsize: tuple[int, int] = (12, 12),
    ) -> list[BraxState]:
        """Main function to unroll trajectory and optionally render results."""
        logger.info("Running test run for %d steps", num_steps)

        # Convert action type and run simulation
        if isinstance(actions, str):
            match actions:
                case "random":
                    actions, init_carry = get_random_action, None
                case "zero":
                    actions, init_carry = get_zero_action, None
                case "midpoint":
                    actions, init_carry = get_midpoint_action, None
                case _:
                    raise ValueError(f"Invalid action type: {actions}")
        elif not isinstance(actions, ActionModel):
            raise ValueError(f"Invalid action type: {type(actions)}")

        # Run simulation
        rng = jax.random.PRNGKey(seed)
        trajectory = self.unroll_trajectory(num_steps, rng, init_carry, actions)

        # Remove states after episode finished
        done = trajectory.done
        done = jnp.pad(done[:-1], (1, 0), mode="constant", constant_values=False)
        trajectory = jax.tree.map(lambda x: x[~done], trajectory)

        # Handle rendering if requested
        if render_dir is not None:
            render_dir = Path(render_dir)

            # Save raw trajectory
            render_dir.mkdir(parents=True, exist_ok=True)
            with open(render_dir / "trajectory.pkl", "wb") as f:
                raw_trajectory = jax.tree.map(lambda x: np.array(x) if isinstance(x, jnp.ndarray) else x, trajectory)
                pkl.dump(raw_trajectory, f)

            # Generate and save plots
            for plot_key, img in self.generate_trajectory_plots(trajectory, figsize):
                (full_path := render_dir / plot_key).parent.mkdir(parents=True, exist_ok=True)
                img.save(full_path)
                logger.info("Saved %s", full_path)

            # Generate and save video
            frames, fps = self.render_trajectory_video(trajectory, camera)
            video_path = render_dir / "render.mp4"
            mediapy.write_video(video_path, frames, fps=fps)

        return trajectory
