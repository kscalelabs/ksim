"""Defines the default humanoid environment."""

import asyncio
import io
import logging
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Iterator, Literal, Protocol, TypeVar, cast, get_args

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import xax
from brax.base import State, System
from brax.envs.base import PipelineEnv, State as BraxState
from jaxtyping import PRNGKeyArray
from kscale import K
from kscale.web.utils import get_robots_dir, should_refresh_file
from mujoco_scenes.brax import load_model
from mujoco_scenes.mjcf import load_mjmodel
from omegaconf import MISSING, DictConfig, OmegaConf
from PIL.Image import Image as PILImage

from ksim.commands import Command, CommandBuilder
from ksim.observation.base import Observation, ObservationBuilder
from ksim.resets.base import Reset, ResetBuilder, ResetData
from ksim.rewards.base import Reward, RewardBuilder
from ksim.terminations.base import Termination, TerminationBuilder
from ksim.utils.data import BuilderData

logger = logging.getLogger(__name__)

T = TypeVar("T")

STATE_KEY = "state"


@dataclass
class ActuatorMetadata:
    kp: float
    kd: float


@dataclass
class ModelMetadata:
    actuators: dict[str, ActuatorMetadata]
    control_frequency: float


async def get_model_path(model_name: str, cache: bool = True) -> str | Path:
    async with K() as api:
        urdf_dir = await api.download_and_extract_urdf(model_name, cache=cache)

    try:
        mjcf_path = next(urdf_dir.glob("*.mjcf"))
    except StopIteration:
        raise ValueError(f"No MJCF file found for {model_name} (in {urdf_dir})")

    return mjcf_path


async def get_model_metadata(model_name: str, cache: bool = True) -> ModelMetadata:
    metadata_path = get_robots_dir() / model_name / "metadata.yaml"

    # Downloads and caches the metadata if it doesn't exist.
    if not cache or not (metadata_path.exists() and not should_refresh_file(metadata_path)):
        async with K() as api:
            robot_class = await api.get_robot_class(model_name)
            if (metadata := robot_class.metadata) is None:
                raise ValueError(f"No metadata found for {model_name}")

        if (control_frequency := metadata.control_frequency) is None:
            raise ValueError(f"No control frequency found for {model_name}")
        if (actuators := metadata.joint_name_to_metadata) is None:
            raise ValueError(f"No actuators found for {model_name}")
        actuator_metadata = {k: ActuatorMetadata(kp=v.kp, kd=v.kd) for k, v in actuators.items()}
        model_metadata = ModelMetadata(actuators=actuator_metadata, control_frequency=control_frequency)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(model_metadata, metadata_path)

    config = OmegaConf.structured(ModelMetadata)
    return cast(ModelMetadata, OmegaConf.merge(config, OmegaConf.load(metadata_path)))


async def get_model_and_metadata(model_name: str, cache: bool = True) -> tuple[str, ModelMetadata]:
    return await asyncio.gather(
        get_model_path(
            model_name=model_name,
            cache=not cache,
        ),
        get_model_metadata(
            model_name=model_name,
            cache=not cache,
        ),
    )


def _unique_list(things: list[tuple[str, T]]) -> list[tuple[str, T]]:
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


class ActionModel(Protocol):
    def __call__(
        self,
        sys: System,
        state: BraxState,
        rng: PRNGKeyArray,
        carry: T,
    ) -> tuple[jnp.ndarray, T]: ...


@eqx.filter_jit
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


@eqx.filter_jit
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


@eqx.filter_jit
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
        value="patch",
        help="The scene to use for the model.",
    )
    render_camera: str = xax.field(
        value="tracking_camera",
        help="The camera to use for rendering.",
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
        commands: Collection[Command | CommandBuilder] = (),
    ) -> None:
        self.config = config

        # Downloads the model from the K-Scale API and loads it into MuJoCo.
        model_path, model_metadata = asyncio.run(
            get_model_and_metadata(
                self.config.model_name,
                cache=not self.config.ignore_cached_urdf,
            )
        )

        logger.info("Loading model %s", model_path)
        mj_model = load_mjmodel(model_path, self.config.model_scene)

        # Populates configuration joint information.
        self.body_name_to_idx = {mj_model.body(i).name: i for i in range(mj_model.nbody)}
        self.joint_name_to_idx = {mj_model.joint(i).name: i for i in range(mj_model.njnt)}
        self.actuator_name_to_idx = {mj_model.actuator(i).name: i for i in range(mj_model.nu)}
        self.geom_name_to_idx = {mj_model.geom(i).name: i for i in range(mj_model.ngeom)}
        self.site_name_to_idx = {mj_model.site(i).name: i for i in range(mj_model.nsite)}
        self.sensor_name_to_idx = {mj_model.sensor(i).name: i for i in range(mj_model.nsensor)}

        # Populates the Kp and Kd values.
        id_to_kp = {
            i: model_metadata.actuators[name].kp
            for name, i in self.joint_name_to_idx.items()
            if name in model_metadata.actuators
        }
        id_to_kd = {
            i: model_metadata.actuators[name].kd
            for name, i in self.joint_name_to_idx.items()
            if name in model_metadata.actuators
        }

        # Convert to list, skipping the root joint.
        kps = [id_to_kp[i] for i in range(1, mj_model.njnt)]
        kds = [id_to_kd[i] for i in range(1, mj_model.njnt)]

        self.kps = jnp.array(kps)
        self.kds = jnp.array(kds)

        # Gets the relevant data for building the components of the environment.
        data = BuilderData(
            model=mj_model,
            dt=self.config.dt,
            ctrl_dt=self.config.ctrl_dt,
            body_name_to_idx=self.body_name_to_idx,
            joint_name_to_idx=self.joint_name_to_idx,
            actuator_name_to_idx=self.actuator_name_to_idx,
            geom_name_to_idx=self.geom_name_to_idx,
            site_name_to_idx=self.site_name_to_idx,
            sensor_name_to_idx=self.sensor_name_to_idx,
        )

        # Builds the terminations, resets, rewards, and observations.
        terminations_v: list[Termination] = [t(data) if isinstance(t, TerminationBuilder) else t for t in terminations]
        resets_v: list[Reset] = [r(data) if isinstance(r, ResetBuilder) else r for r in resets]
        rewards_v: list[Reward] = [r(data) if isinstance(r, RewardBuilder) else r for r in rewards]
        observations_v: list[Observation] = [o(data) if isinstance(o, ObservationBuilder) else o for o in observations]
        commands_v: list[Command] = [c(data) if isinstance(c, CommandBuilder) else c for c in commands]

        # Creates dictionaries of the unique terminations, resets, rewards, and observations.
        self.terminations = _unique_list([(term.termination_name, term) for term in terminations_v])
        self.resets = _unique_list([(reset.reset_name, reset) for reset in resets_v])
        self.rewards = _unique_list([(reward.reward_name, reward) for reward in rewards_v])
        self.observations = _unique_list([(obs.observation_name, obs) for obs in observations_v])
        self.commands = _unique_list([(cmd.command_name, cmd) for cmd in commands_v])

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

        # Applies resets to the pipeline state.
        reset_data = ResetData(rng=rng, state=pipeline_state)
        for _, reset_func in self.resets:
            reset_data = reset_func(reset_data)
        pipeline_state = reset_data.state

        # Gets the observations, rewards, and terminations.
        rng, obs_rng = jax.random.split(rng)
        obs = self.get_observation(pipeline_state, obs_rng)
        all_dones = self.get_terminations(pipeline_state)
        all_rewards = [(key, jnp.zeros(())) for key, _ in self.rewards]

        done = jnp.stack([v for _, v in all_dones], axis=-1).any(axis=-1)
        reward = jnp.stack([v for _, v in all_rewards], axis=-1).sum(axis=-1)

        info = {
            "time": jnp.zeros(()),
            "rng": rng,
            "all_dones": {k: v for k, v in all_dones},
            "all_rewards": {k: v for k, v in all_rewards},
            "commands": {},
        }

        for cmd_name, cmd in self.commands:
            cmd_val = cmd(rng)
            obs.append((cmd_name, cmd_val))
            info["commands"][cmd_name] = cmd_val

        # Concatenate the observations into a single array, for convenience.
        obs.append((STATE_KEY, jnp.concatenate([o.reshape(-1) for _, o in obs], axis=-1)))

        return BraxState(
            pipeline_state=pipeline_state,
            obs={k: v for k, v in obs},
            reward=reward,
            done=done.astype(reward.dtype),
            info=info,
        )

    def actuator_model(self, pipeline_state: BraxState, actions: jnp.ndarray) -> BraxState:
        """Defines the actuator model, converting from actions to torques.

        K-Scale's MJCF files expect torque inputs, but our models provide
        position and velocity commands. This function specifies our own actuator
        model, taking the target position and velocity and computing the necessary
        torque to achieve that.

        Args:
            pipeline_state: The current state of the pipeline.
            actions: The actions to convert to torques.

        Returns:
            The updated pipeline state.
        """
        pipeline_num_actions = self.sys.actuator.ctrl_range.shape[0]

        if pipeline_num_actions == actions.shape[0]:
            # Position control.
            cur_pos, tar_pos = pipeline_state.q[7:], actions
            ctrl = self.kps * (tar_pos - cur_pos)
            return ctrl

        elif pipeline_num_actions == actions.shape[0] * 2:
            # Position and velocity control.
            cur_pos, tar_pos = pipeline_state.q[7:], actions[:pipeline_num_actions]
            cur_vel, tar_vel = pipeline_state.qd[6:], actions[pipeline_num_actions:]
            ctrl = self.kps * (tar_pos - cur_pos) + self.kds * (tar_vel - cur_vel)
            return ctrl

        else:
            raise ValueError(f"Invalid number of actions: {actions.shape[0]}")

    def pipeline_step(self, pipeline_state: BraxState, actions: jnp.ndarray) -> BraxState:
        """Takes a physics step using the physics pipeline."""

        def f(state: BraxState, _) -> tuple[BraxState, None]:
            torques = self.actuator_model(state, actions)
            return self._pipeline.step(self.sys, state, torques, self._debug), None

        return jax.lax.scan(f, pipeline_state, (), self._n_frames)[0]

    @eqx.filter_jit
    def step(self, prev_state: BraxState, action: jnp.ndarray) -> BraxState:
        pipeline_state = self.pipeline_step(prev_state.pipeline_state, action)

        # Update the metrics.
        time = prev_state.info["time"]
        rng = prev_state.info["rng"]

        rng, obs_rng = jax.random.split(rng)
        obs = self.get_observation(pipeline_state, obs_rng)
        all_dones = self.get_terminations(pipeline_state)
        all_rewards = self.get_rewards(prev_state, action, pipeline_state)

        done = jnp.stack([v for _, v in all_dones], axis=-1).any(axis=-1)
        reward = jnp.stack([v for _, v in all_rewards], axis=-1).sum(axis=-1)

        for cmd_name, cmd in self.commands:
            rng, cmd_rng = jax.random.split(rng)
            prev_cmd = prev_state.info["commands"][cmd_name]
            next_cmd = cmd.update(prev_cmd, cmd_rng, time)
            obs.append((cmd_name, next_cmd))
            prev_state.info["commands"][cmd_name] = next_cmd

        # Concatenate the observations into a single state vector, for convenience.
        obs.append((STATE_KEY, jnp.concatenate([o.reshape(-1) for _, o in obs], axis=-1)))

        # Update with the new state.
        next_state = prev_state.tree_replace(
            {
                "pipeline_state": pipeline_state,
                "obs": {k: v for k, v in obs},
                "reward": reward,
                "done": done.astype(reward.dtype),
            },
        )

        next_state.info["time"] = time + self.config.ctrl_dt
        next_state.info["rng"] = rng
        next_state.info["all_dones"] = {k: v for k, v in all_dones}
        next_state.info["all_rewards"] = {k: v for k, v in all_rewards}

        return next_state

    @eqx.filter_jit
    def get_observation(
        self,
        pipeline_state: State,
        rng: PRNGKeyArray,
    ) -> list[tuple[str, jnp.ndarray]]:
        observations: list[tuple[str, jnp.ndarray]] = []
        for observation_name, observation in self.observations:
            rng, obs_rng = jax.random.split(rng)
            observation_value = observation(pipeline_state)
            observation_value = observation.add_noise(observation_value, obs_rng)
            observations.append((observation_name, observation_value))
        return observations

    @eqx.filter_jit
    def get_rewards(
        self,
        prev_state: BraxState,
        action: jnp.ndarray,
        pipeline_state: State,
    ) -> list[tuple[str, jnp.ndarray]]:
        rewards: list[tuple[str, jnp.ndarray]] = []
        for reward_name, reward in self.rewards:
            reward_val = reward(prev_state, action, pipeline_state) * reward.scale
            chex.assert_shape(reward_val, ())
            rewards.append((reward_name, reward_val))
        return rewards

    @eqx.filter_jit
    def get_terminations(self, pipeline_state: State) -> list[tuple[str, jnp.ndarray]]:
        terminations: list[tuple[str, jnp.ndarray]] = []
        for termination_name, termination in self.terminations:
            term_val = termination(pipeline_state)
            assert term_val.shape == (), f"Termination {termination_name} must be a scalar, got {term_val.shape}"
            chex.assert_shape(term_val, ())
            terminations.append((termination_name, term_val))
        return terminations

    @eqx.filter_jit
    def unroll_trajectory(
        self,
        rng: PRNGKeyArray,
        num_steps: int,
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
            done_condition = state.done.astype(bool)
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
        all_rewards = states.info["all_rewards"]
        reward_list = [reward_fn.post_accumulate(all_rewards[name]) for name, reward_fn in self.rewards]
        rewards = jnp.stack(reward_list, axis=1)
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
    ) -> PILImage:
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
        return PIL.Image.open(buf)

    def generate_trajectory_plots(
        self,
        trajectory: BraxState,
        figsize: tuple[int, int] = (12, 12),
    ) -> Iterator[tuple[str, PILImage]]:
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
        num_steps = (~trajectory.done.astype(bool)).sum()
        raw_trajectory = jax.tree.map(
            lambda x: np.array(x[:num_steps]) if isinstance(x, jnp.ndarray) else x,
            trajectory,
        )
        t = np.arange(num_steps) * self.config.ctrl_dt

        # Generate command plots
        for i, (key, _) in enumerate(self.commands):
            metric = raw_trajectory.info["commands"][key]
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
        for i, (key, _) in enumerate(self.rewards):
            reward_data = raw_trajectory.reward[:, i : i + 1].astype(np.float32)
            img = self._plot_trajectory_data(t, reward_data, title=key, ylabel=key, figsize=figsize)
            yield f"rewards/{key}.png", img

        # Generate observation plots
        for key, _ in self.observations:
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

    def render_trajectory_video(self, trajectory: BraxState) -> tuple[np.ndarray, int]:
        """Render trajectory as video frames with computed FPS."""
        num_steps = (~trajectory.done.astype(bool)).sum()
        fps = round(1 / self.config.ctrl_dt)
        pipeline_states = [jax.tree.map(lambda arr: arr[i], trajectory.pipeline_state) for i in range(num_steps)]
        frames = np.stack(self.render(pipeline_states, camera=self.config.render_camera), axis=0)
        return frames, fps

    def unroll_trajectory_and_render(
        self,
        rng: PRNGKeyArray,
        num_steps: int,
        render_dir: str | Path | None = None,
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
        trajectory = self.unroll_trajectory(
            rng=rng,
            num_steps=num_steps,
            init_carry=init_carry,
            model=actions,
        )

        # Remove states after episode finished
        done = trajectory.done.astype(bool)
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
            frames, fps = self.render_trajectory_video(trajectory)
            video_path = render_dir / "render.gif"
            images = [PIL.Image.fromarray(frame) for frame in frames]
            images[0].save(video_path, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0)
            logger.info("Saved video to %s", video_path)

        return trajectory
