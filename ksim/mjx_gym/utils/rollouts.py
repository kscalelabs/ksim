"""Defines some utility functions for rollouts."""

import logging
from typing import Callable

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.mjx.base import State as mjxState
from mujoco import mjx
from tqdm import tqdm

logger = logging.getLogger(__name__)

InferenceFn = Callable[[jp.ndarray, jp.ndarray], tuple[jp.ndarray, jp.ndarray]]


def mjx_rollout(
    env: mujoco.MjModel,
    inference_fn: InferenceFn,
    n_steps: int = 1000,
    render_every: int = 2,
    seed: int = 0,
) -> list[mjxState]:
    """Rollout a trajectory using MJX.

    It is worth noting that env, a Brax environment, is expected to implement MJX
    in the background. See default_humanoid_env for reference.

    Args:
        env: Brax environment
        inference_fn: Inference function
        n_steps: Number of steps to rollout
        render_every: Render every nth step
        seed: Random seed

    Returns:
        A list of pipeline states of the policy rollout
    """
    # print(f"Rolling out {n_steps} steps with MJX")
    logger.info("Rolling out %d steps with MJX", n_steps)
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    inference_fn = jax.jit(inference_fn)
    rng = jax.random.PRNGKey(seed)

    state = reset_fn(rng)
    rollout = [state.pipeline_state]
    for i in tqdm(range(n_steps)):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = inference_fn(state.obs, act_rng)
        state = step_fn(state, ctrl)
        rollout.append(state.pipeline_state)

        if state.done:
            state = reset_fn(rng)

    return rollout


def render_mjx_rollout(
    env: mujoco.MjModel,
    inference_fn: InferenceFn,
    n_steps: int = 1000,
    render_every: int = 2,
    seed: int = 0,
    width: int = 320,
    height: int = 240,
) -> np.ndarray:
    """Rollout a trajectory using MuJoCo and render it.

    Args:
        env: Brax environment
        inference_fn: Inference function
        n_steps: Number of steps to rollout
        render_every: Render every nth step
        seed: Random seed
        width: width of rendered frame in pixels
        height: height of rendered frame in pixels

    Returns:
        A list of renderings of the policy rollout with dimensions (T, H, W, C)
    """
    rollout = mjx_rollout(env, inference_fn, n_steps, render_every, seed)
    images = env.render(rollout[::render_every], camera="side", width=width, height=height)

    return np.array(images)


def render_mujoco_rollout(
    env: mujoco.MjModel,
    inference_fn: InferenceFn,
    n_steps: int = 1000,
    render_every: int = 2,
    seed: int = 0,
    width: int = 320,
    height: int = 240,
) -> np.ndarray:
    """Rollout a trajectory using MuJoCo.

    Args:
        env: Brax environment
        inference_fn: Inference function
        n_steps: Number of steps to rollout
        render_every: Render every nth step
        seed: Random seed
        width: width of rendered frame in pixels
        height: height of rendered frame in pixels

    Returns:
        A list of images of the policy rollout (T, H, W, C)
    """
    print(f"Rolling out {n_steps} steps with MuJoCo")
    model = env.sys.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    ctrl = jp.zeros(model.nu)

    images: list[np.ndarray] = []
    rng = jax.random.PRNGKey(seed)
    for step in tqdm(range(n_steps)):
        act_rng, seed = jax.random.split(rng)
        obs = env._get_obs(mjx.put_data(model, data), ctrl)
        # TODO: implement methods in envs that avoid having to use mjx in a hacky way...
        # print(obs)
        ctrl, _ = inference_fn(obs, act_rng)
        data.ctrl = ctrl
        for _ in range(env._n_frames):
            mujoco.mj_step(model, data)

        if step % render_every == 0:
            renderer.update_scene(data, camera="side")
            images.append(renderer.render())

    return np.array(images)


def render_random_rollout(
    env: mujoco.MjModel,
    n_steps: int = 1000,
    render_every: int = 2,
    seed: int = 0,
    width: int = 320,
    height: int = 240,
) -> np.ndarray:
    """Rollout a trajectory using MuJoCo.

    Args:
        env: Brax environment
        inference_fn: Inference function
        n_steps: Number of steps to rollout
        render_every: Render every nth step
        seed: Random seed
        width: width of rendered frame in pixels
        height: height of rendered frame in pixels

    Returns:
        A list of images of the policy rollout (T, H, W, C)
    """
    print(f"Rolling out {n_steps} steps with MuJoCo")
    model = env.sys.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    ctrl = jp.zeros(model.nu)

    images: list[np.ndarray] = []
    seed = jax.random.PRNGKey(seed)
    env.reset(seed)
    for step in tqdm(range(n_steps)):
        # collect random actions
        ctrl = jax.random.uniform(seed, (model.nu,), minval=-1, maxval=1)
        data.ctrl = ctrl
        for _ in range(env._n_frames):
            mujoco.mj_step(model, data)

        if step % render_every == 0:
            renderer.update_scene(data, camera="side")
            images.append(renderer.render())

    return np.array(images)
