from typing import List

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.mjx.base import State as mjxState
from mujoco import mjx
from tqdm import tqdm


def mjx_rollout(env, inference_fn, n_steps=1000, render_every=2, seed=0) -> List[mjxState]:
    """
    Rollout a trajectory using MJX
    Args:
        env: Brax environment
        inference_fn: Inference function
        n_steps: Number of steps to rollout
        render_every: Render every nth step
        seed: Random seed
    Returns:
        A list of pipeline states of the policy rollout

    It is worth noting that env, a Brax environment, is expected to implement MJX
    in the background. See default_humanoid_env for reference.
    """
    print(f"Rolling out {n_steps} steps with MJX")
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


def render_mjx_rollout(env, inference_fn, n_steps=1000, render_every=2, seed=0) -> np.ndarray:
    """
    Rollout a trajectory using MuJoCo and render it

    Args:
        env: Brax environment
        inference_fn: Inference function
        n_steps: Number of steps to rollout
        render_every: Render every nth step
        seed: Random seed
    Returns:
        A list of renderings of the policy rollout with dimensions (T, H, W, C)
    """
    rollout = mjx_rollout(env, inference_fn, n_steps, render_every, seed)
    images = env.render(rollout[::render_every], camera="side")

    return np.array(images)


def render_mujoco_rollout(env, inference_fn, n_steps=1000, render_every=2, seed=0):
    """
    Rollout a trajectory using MuJoCo
    Args:
        env: Brax environment
        inference_fn: Inference function
        n_steps: Number of steps to rollout
        render_every: Render every nth step
        seed: Random seed
    Returns:
        A list of images of the policy rollout (T, H, W, C)
    """
    print(f"Rolling out {n_steps} steps with MuJoCo")
    model = env.sys.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model)
    ctrl = jp.zeros(model.nu)

    images = []
    rng = jax.random.PRNGKey(seed)
    for step in tqdm(range(n_steps)):
        act_rng, seed = jax.random.split(rng)
        obs = env._get_obs(mjx.put_data(model, data), ctrl)
        # TODO: implement methods in envs that avoid having to use mjx in a hacky way...
        ctrl, _ = inference_fn(obs, act_rng)
        data.ctrl = ctrl
        for _ in range(env._n_frames):
            mujoco.mj_step(model, data)

        if step % render_every == 0:
            renderer.update_scene(data, camera="side")
            images.append(renderer.render())

    return images
