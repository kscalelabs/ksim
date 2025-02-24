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
from typing import (
    Any,
    Callable,
    Collection,
    Literal,
    Tuple,
    TypeVar,
    cast,
    get_args,
)

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import xax
from mujoco import mjx
from jaxtyping import Array, PRNGKeyArray
from kscale import K
from kscale.web.utils import get_robots_dir, should_refresh_file
from mujoco_scenes.mjcf import load_mjmodel
from omegaconf import MISSING, OmegaConf

from ksim.commands import Command, CommandBuilder
from ksim.env.base_env import BaseEnv
from ksim.observation import Observation, ObservationBuilder
from ksim.resets import Reset, ResetBuilder, ResetData
from ksim.rewards import Reward, RewardBuilder
from ksim.terminations import Termination, TerminationBuilder
from ksim.utils.data import BuilderData
from ksim.utils.mujoco import make_mujoco_mappings
from ksim.env.actuators.mit_controller import MITPositionActuators
logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ActuatorMetadata:
    kp: float
    kd: float


@dataclass
class ModelMetadata:
    actuators: dict[str, ActuatorMetadata]
    control_frequency: float


@dataclass
class EnvState:
    """The state of the environment.
    
    Attributes (inheriteds):
        model: Handles physics and model definition (latter shouldn't be touched).
        data: Includes current state of the robot.
        obs: The post-processed observations of the environment.
        reward: The reward of the environment.
        done: Whether the episode is done.
        info: Additional information about the environment.
    """

    mjx_model: mjx.Model
    mjx_data: mjx.Data # making this non-optional.
    obs: dict[str, Array]
    reward: Array
    done: Array
    info: dict[str, Any]


def step_mjx(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    ctrl: Array,
) -> mjx.Data:
    """Step the mujoco model."""
    data_with_ctrl = mjx_data.replace(ctrl=ctrl)
    # more logic if needed...
    mjx.step(mjx_model, data_with_ctrl)
    return data_with_ctrl

async def get_model_path(model_name: str, cache: bool = True) -> str | Path:
    """Downloads and caches the model URDF."""
    async with K() as api:
        urdf_dir = await api.download_and_extract_urdf(model_name, cache=cache)

    try:
        mjcf_path = next(urdf_dir.glob("*.mjcf"))
    except StopIteration:
        raise ValueError(f"No MJCF file found for {model_name} (in {urdf_dir})")

    return mjcf_path


async def get_model_metadata(model_name: str, cache: bool = True) -> ModelMetadata:
    """Downloads and caches the model metadata."""
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
        model_metadata = ModelMetadata(
            actuators=actuator_metadata, control_frequency=control_frequency
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(model_metadata, metadata_path)

    config = OmegaConf.structured(ModelMetadata)
    return cast(ModelMetadata, OmegaConf.merge(config, OmegaConf.load(metadata_path)))


async def get_model_and_metadata(model_name: str, cache: bool = True) -> tuple[str, ModelMetadata]:
    """Downloads and caches the model URDF and metadata."""
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
@eqx.filter_jit
def get_random_action(
    env_state: EnvState,
    rng: PRNGKeyArray,
    carry: None,
) -> tuple[jnp.ndarray, None]:
    """Get a random action."""
    ctrl_range = env_state.mjx_model.actuator_ctrlrange
    ctrl_min, ctrl_max = ctrl_range.T
    action_scale = jax.random.uniform(rng, shape=ctrl_min.shape, dtype=ctrl_min.dtype)
    ctrl = ctrl_min + (ctrl_max - ctrl_min) * action_scale
    return ctrl, None


@eqx.filter_jit
def get_midpoint_action(
    env_state: EnvState,
    rng: PRNGKeyArray,
    carry: None,
) -> tuple[jnp.ndarray, None]:
    """Get a midpoint action."""
    ctrl_range = env_state.mjx_model.actuator_ctrlrange
    ctrl_min, ctrl_max = ctrl_range.T
    ctrl = (ctrl_min + ctrl_max) / 2
    return ctrl, None


@eqx.filter_jit
def get_zero_action(
    env_state: EnvState,
    rng: PRNGKeyArray,
    carry: None,
) -> tuple[jnp.ndarray, None]:
    """Get a zero action."""
    ctrl = jnp.zeros_like(env_state.mjx_model.actuator_ctrlrange[..., 0])
    return ctrl, None


KScaleActionModelType = Literal["random", "zero", "midpoint"]


def cast_action_type(action_type: str) -> KScaleActionModelType:
    """Cast the action type to the correct type."""
    options = get_args(KScaleActionModelType)
    if action_type not in options:
        raise ValueError(f"Invalid action type: {action_type} Choices are {options}")
    return cast(KScaleActionModelType, action_type)


@jax.tree_util.register_dataclass
@dataclass
class KScaleEnvConfig(xax.Config):
    # robot model configuration options
    robot_model_name: str = xax.field(value=MISSING, help="The name of the model to use.")
    robot_model_scene: str = xax.field(value="patch", help="The scene to use for the model.")
    render_camera: str = xax.field(value="tracking_camera", help="The camera to use for rendering.")
    render_width: int = xax.field(value=640, help="The width of the rendered image.")
    render_height: int = xax.field(value=480, help="The height of the rendered image.")

    # environment configuration options
    dt: float = xax.field(value=0.004, help="Simulation time step.")
    ctrl_dt: float = xax.field(value=0.02, help="Control time step.")
    debug_env: bool = xax.field(value=False, help="Whether to enable debug mode for the environment.")

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
      - All state (a BraxState) is passed in and returned by reset and step.
      - The underlying Brax system (here referred to as `brax_sys`) is provided to step/reset.
      - Rollouts are performed by vectorizing (vmap) the reset and step functions,
        with a final trajectory of shape (time, num_envs, ...).
      - The step wrapper only computes a reset (via jax.lax.cond) if the done flag is True.
    """

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
        self.mujoco_mappings = make_mujoco_mappings(mj_model)
        self.actuators = MITPositionActuators(
            actuators_metadata=robot_model_metadata.actuators,
            mujoco_mappings=self.mujoco_mappings,
        )

        # building mappings from mj_model parts to indices
        # self.body_name_to_idx = {mj_model.body(i).name: i for i in range(mj_model.nbody)}
        # self.joint_name_to_idx = {mj_model.joint(i).name: i for i in range(mj_model.njnt)}
        # self.actuator_name_to_idx = {mj_model.actuator(i).name: i for i in range(mj_model.nu)}
        # self.geom_name_to_idx = {mj_model.geom(i).name: i for i in range(mj_model.ngeom)}
        # self.site_name_to_idx = {mj_model.site(i).name: i for i in range(mj_model.nsite)}
        # self.sensor_name_to_idx = {mj_model.sensor(i).name: i for i in range(mj_model.nsensor)}

        # populating KP and KD.
        # id_to_kp = {
        #     i: robot_model_metadata.actuators[name].kp
        #     for name, i in self.joint_name_to_idx.items()
        #     if name in robot_model_metadata.actuators
        # }
        # id_to_kd = {
        #     i: robot_model_metadata.actuators[name].kd
        #     for name, i in self.joint_name_to_idx.items()
        #     if name in robot_model_metadata.actuators
        # }

        # skipping the root joint.
        # kps = [id_to_kp[i] for i in range(1, mj_model.njnt)]
        # kds = [id_to_kd[i] for i in range(1, mj_model.njnt)]
        # self.kps = jnp.array(kps)
        # self.kds = jnp.array(kds)

        # preparing builder data.
        # data = BuilderData(
        #     model=mj_model,
        #     dt=self.config.dt,
        #     ctrl_dt=self.config.ctrl_dt,
        #     body_name_to_idx=self.body_name_to_idx,
        #     joint_name_to_idx=self.joint_name_to_idx,
        #     actuator_name_to_idx=self.actuator_name_to_idx,
        #     geom_name_to_idx=self.geom_name_to_idx,
        #     site_name_to_idx=self.site_name_to_idx,
        #     sensor_name_to_idx=self.sensor_name_to_idx,
        # )

        # storing the termination, reset, reward, observation, and command builders
        # terminations_v = [t(data) if isinstance(t, TerminationBuilder) else t for t in terminations]
        # resets_v = [r(data) if isinstance(r, ResetBuilder) else r for r in resets]
        # rewards_v = [r(data) if isinstance(r, RewardBuilder) else r for r in rewards]
        # observations_v = [o(data) if isinstance(o, ObservationBuilder) else o for o in observations]
        # commands_v = [c(data) if isinstance(c, CommandBuilder) else c for c in commands]

        # self.terminations = _unique_list([(term.termination_name, term) for term in terminations_v])
        # self.resets = _unique_list([(reset.reset_name, reset) for reset in resets_v])
        # self.rewards = _unique_list([(reward.reward_name, reward) for reward in rewards_v])
        # self.observations = _unique_list([(obs.observation_name, obs) for obs in observations_v])
        # self.commands = _unique_list([(cmd.command_name, cmd) for cmd in commands_v])

        logger.info("Converting model to Brax system")

        # For simplicity, assume integer (increase granularity if needed).
        assert self.config.ctrl_dt % self.config.dt == 0, "ctrl_dt must be a multiple of dt"
        self._expected_dt_per_ctrl_dt = int(self.config.ctrl_dt / self.config.dt)

    ###################
    # Post Processing #
    ###################

    def get_observation(self, mjx_data: mjx.Data, rng: jax.Array) -> dict:
        """Compute observations from the pipeline state."""
        observations = []
        for observation_name, observation in self.observations:
            rng, obs_rng = jax.random.split(rng)
            observation_value = observation(mjx_data)
            observation_value = observation.add_noise(observation_value, obs_rng)
            observations.append((observation_name, observation_value))
        return {k: v for k, v in observations}

    def get_rewards(self, prev_state: EnvState, action: jnp.ndarray, new_mjx_data: mjx.Data):
        """Compute rewards (each as a scalar) from the state transition."""
        rewards = []
        for reward_name, reward in self.rewards:
            reward_val = reward(prev_state, action, new_mjx_data) * reward.scale
            chex.assert_shape(
                reward_val, (), custom_message=f"Reward {reward_name} must be a scalar"
            )
            rewards.append((reward_name, reward_val))
        return rewards

    def get_terminations(self, new_mjx_data: mjx.Data):
        """Compute termination conditions (each as a scalar) from the pipeline state."""
        terminations = []
        for termination_name, termination in self.terminations:
            term_val = termination(new_mjx_data)
            chex.assert_shape(
                term_val, (), custom_message=f"Termination {termination_name} must be a scalar"
            )
            terminations.append((termination_name, term_val))
        return terminations

    ###########################
    # Main API Implementation #
    ###########################

    def reset(self, env_state: EnvState, rng: jax.Array) -> EnvState:
        """Pure reset function: returns an initial state computed solely from the inputs."""
        # Apply any reset functions.
        reset_data = ResetData(rng=rng, state=env_state)
        for _, reset_func in self.resets:
            reset_data = reset_func(reset_data)
        updated_state = reset_data.state

        rng, obs_rng = jax.random.split(rng)

        new_data = step_mjx(
            mjx_model=updated_state.mjx_model,
            mjx_data=updated_state.mjx_data,
            ctrl=jnp.zeros_like(updated_state.mjx_data.ctrl),
        )

        done = jnp.array(False, dtype=jnp.bool_)
        reward = jnp.array(0.0)
        info = {
            "time": jnp.array(0.0),
            "rng": rng,
            "prev_ctrl": jnp.zeros_like(new_data.ctrl),
        }
        obs = self.get_observation(new_data, obs_rng)
        return EnvState(
            mjx_model=updated_state.mjx_model,
            mjx_data=new_data,
            obs=obs,
            reward=reward,
            done=done,
            info=info,
        )
    
    def _apply_physics_steps(
        self,
        env_state: EnvState,
        ctrl: Array,
        prev_ctrl: Array,
        num_latency_steps: int,
    ) -> mjx.Data:
        """A 'step' of the environment on a state composes multiple steps of the actual physics.
        
        We take num_latency_steps (applying the previous action), then apply the control signal for
        the remainder of the physics steps.
        """
        n_steps = self._expected_dt_per_ctrl_dt # total number of pipeline steps to take.
        mjx_data = env_state.mjx_data

        def f(carry: Tuple[mjx.Data, int], _: Any) -> Tuple[Tuple[mjx.Data, int], None]:
            state, step_num = carry
            torques = jax.lax.select(step_num >= num_latency_steps, ctrl, prev_ctrl)

            # NOTE: can extend state to include anything from `mjx.Data` here...
            new_state = step_mjx(env_state.mjx_model, state, torques) # type: ignore
            return (new_state, step_num + 1), None

        (state, _), _ = jax.lax.scan(f, (mjx_data, 0), None, n_steps)
        return state


    def step(
        self, env_state: EnvState, action: Array, rng: Array
    ) -> EnvState:
        """Stepping the environment in a consistent, JIT-able manner.

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
        rng, latency_rng, obs_rng = jax.random.split(rng, 3)
        latency_steps = jax.random.randint(
            key=latency_rng,
            shape=(),
            minval=self.min_action_latency_step,
            maxval=self.max_action_latency_step,
        )
        prev_ctrl = env_state.info["prev_ctrl"]
        torque_ctrl = self.actuators.get_ctrl(env_state, action)

        new_mjx_data = self._apply_physics_steps(
            env_state, torque_ctrl, prev_ctrl, latency_steps.item(),
        )

        obs = self.get_observation(new_mjx_data, obs_rng)
        all_dones = self.get_terminations(new_mjx_data)
        done = jnp.stack([v for _, v in all_dones], axis=-1).any(axis=-1)
        all_rewards = self.get_rewards(env_state, action, new_mjx_data)
        reward = jnp.stack([v for _, v in all_rewards], axis=-1).sum(axis=-1)

        new_info = dict(
            time=env_state.info["time"] + self.config.ctrl_dt,
            rng=rng,
            prev_ctrl=new_mjx_data.ctrl,
        )

        new_state = EnvState(
            mjx_model=env_state.mjx_model,
            mjx_data=new_mjx_data,
            obs=obs,
            reward=reward,
            done=done,
            info=new_info,
        )
        
        return new_state

    def rollout_trajectories(
        self,
        env_state: EnvState,
        rng: Array,
        num_steps: int,
        num_envs: int,
        action_fn: Callable[[EnvState], Array],
    ) -> EnvState:
        """
        Vectorized rollout of trajectories.
        
        1. The batched reset (using vmap) initializes a state for each environment.
        2. A vectorized (vmap-ed) env_step function is defined that calls step.
        3. A jax.lax.scan unrolls the trajectory for num_steps.
        4. The resulting trajectory has shape (num_steps, num_envs, ...).
        """
        init_rngs = jax.random.split(rng, num_envs)
        init_states = jax.vmap(lambda key: self.reset(env_state, key))(init_rngs)
        rng, _ = jax.random.split(rng)

        def env_step(env_state: EnvState, rng: Array):
            action = action_fn(env_state)
            new_state = self.step(env_state, action, rng)
            return new_state

        def scan_fn(carry: Tuple[EnvState, Array], _):
            states, rng = carry
            rng, step_rng = jax.random.split(rng)
            rngs = jax.random.split(step_rng, num_envs)
            new_states = jax.vmap(env_step)(states, rngs)
            return (new_states, rng), new_states

        (final_states, _), traj = jax.lax.scan(
            f=scan_fn,
            init=(init_states, rng),
            xs=None,
            length=num_steps,
        )
        return traj  # Shape: (num_steps, num_envs, ...)

