"""Defines some utility functions for rollouts."""

import logging
import subprocess
from typing import Callable

import cv2
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

# def stream_mjx_rollout_mujoco_renderer_bad(
#     env: mujoco.MjModel,
#     inference_fn: InferenceFn,
#     render_every: int = 1,  # Render every frame for smoother video
#     seed: int = 0,
#     width: int = 1280,  # Increased resolution
#     height: int = 720,  # Increased resolution
#     fps: int = 60,  # Higher frame rate
#     twitch_stream_key: str = None
# ):
#     print("Streaming MJX rollout to Twitch. Press Ctrl+C to stop.")
#     reset_fn = jax.jit(env.reset)
#     step_fn = jax.jit(env.step)
#     inference_fn = jax.jit(inference_fn)


#     model = env.sys.mj_model
#     data = mujoco.MjData(model)
#     renderer = mujoco.Renderer(model, width=width, height=height)
#     rng = jax.random.PRNGKey(seed)

#     # Set up FFmpeg command with improved encoding settings
#     command = [
#         'ffmpeg',
#         '-y',
#         '-f', 'rawvideo',
#         '-vcodec', 'rawvideo',
#         '-pix_fmt', 'bgr24',
#         '-s', f'{width}x{height}',
#         '-r', str(fps),
#         '-i', '-',
#         '-c:v', 'libx264',
#         '-preset', 'veryfast',  # Balance between speed and quality
#         '-crf', '23',  # Lower CRF for better quality (range: 0-51, lower is better)
#         '-maxrate', '6000k',
#         '-bufsize', '12000k',
#         '-pix_fmt', 'yuv420p',
#         '-g', '60',  # Set keyframe interval to 60
#         '-f', 'flv',
#         f'rtmp://live.twitch.tv/app/{twitch_stream_key}'
#     ]

#     # Start FFmpeg process
#     process = subprocess.Popen(command, stdin=subprocess.PIPE)

#     state = reset_fn(rng)
#     step = 0

#     try:
#         while True:
#             loop_start = time.time()

#             # Inference
#             inference_start = time.time()
#             act_rng, rng = jax.random.split(rng)
#             ctrl, _ = inference_fn(state.obs, act_rng)
#             data.ctrl = ctrl
#             inference_time = time.time() - inference_start

#             # Step
#             step_start = time.time()
#             state = step_fn(state, ctrl)
#             step_time = time.time() - step_start

#             # for _ in range(render_every):
#                 # mujoco.mj_step(model, data)

#             # Render and stream
#             render_time = 0
#             stream_time = 0
#             if step % render_every == 0:
#                 # render_start = time.time()
#                 # frame = env.render(state.pipeline_state, camera="side", width=width, height=height)
#                 # render_time = time.time() - render_start
                
#                 # stream_start = time.time()
#                 # # Convert frame to BGR for FFmpeg
#                 # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
#                 # # Write frame to FFmpeg process
#                 # process.stdin.write(frame_bgr.tobytes())
#                 # stream_time = time.time() - stream_start

#                 # renderer.update_scene(data)
#                 renderer.update_scene(data, camera="side")
#                 frame = renderer.render()
                
#                 # Convert frame to BGR for FFmpeg
#                 frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
#                 # Write frame to FFmpeg process
#                 process.stdin.write(frame_bgr.tobytes())

#             if state.done:
#                 state = reset_fn(rng)
            
#             loop_time = time.time() - loop_start
            
#             # NOTE: Rendering takes majority of time 

#             # print(f"Step {step}: Total {loop_time:.4f}s | "
#             #       f"Inference {inference_time:.4f}s | "
#             #       f"Step {step_time:.4f}s | "
#             #       f"Render {render_time:.4f}s | "
#             #       f"Stream {stream_time:.4f}s")

#             step += 1

#     except KeyboardInterrupt:
#         print("Streaming interrupted by user.")

#     finally:
#         process.stdin.close()
#         process.wait()
#         print(f"Streaming ended after {step} steps.")

def stream_mjx_rollout(
    env: mujoco.MjModel,
    inference_fn: InferenceFn,
    render_every: int = 2,
    seed: int = 0,
    width: int = 320,
    height: int = 240,
    fps: int = 30,
    twitch_stream_key: str = None
):
    print("Streaming MJX rollout to Twitch. Press Ctrl+C to stop.")
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    inference_fn = jax.jit(inference_fn)
    rng = jax.random.PRNGKey(seed)

    # Set up FFmpeg command
    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-b:v', '100k',
        '-preset', 'ultrafast',
        '-f', 'flv',
        f'rtmp://live.twitch.tv/app/{twitch_stream_key}'
    ]

    # Start FFmpeg process
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    state = reset_fn(rng)
    step = 0
    try:
        while True:
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = inference_fn(state.obs, act_rng)
            state = step_fn(state, ctrl)

            if step % render_every == 0:
                frame = env.render(state.pipeline_state, camera="side", width=width, height=height)
                
                # Convert frame to BGR for FFmpeg
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Write frame to FFmpeg process
                process.stdin.write(frame_bgr.tobytes())
            
            if state.done:
                state = reset_fn(rng)
            
            step += 1

    except KeyboardInterrupt:
        print("Streaming interrupted by user.")

    finally:
        # Close FFmpeg process
        process.stdin.close()
        process.wait()
        print(f"Streaming ended after {step} steps.")

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


def stream_mujoco_rollout(
    env: mujoco.MjModel,
    inference_fn: callable,
    render_every: int = 2,
    seed: int = 0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    twitch_stream_key: str = None
):
    print("Streaming MuJoCo rollout to Twitch. Press Ctrl+C to stop.")
    model = env.sys.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    ctrl = jp.zeros(model.nu)
    rng = jax.random.PRNGKey(seed)

    # Set up FFmpeg command
    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'slow',  # Better quality preset
        '-b:v', '4500k',  # Increased bitrate
        '-preset', 'ultrafast',
        '-f', 'flv',
        f'rtmp://live.twitch.tv/app/{twitch_stream_key}'
    ]

    # Start FFmpeg process
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    step = 0
    try:
        while True:
            act_rng, seed = jax.random.split(rng)
            obs = env._get_obs(mjx.put_data(model, data), ctrl)
            ctrl, *_ = inference_fn(obs, act_rng)
            data.ctrl = ctrl
            
            for _ in range(env._n_frames):
                mujoco.mj_step(model, data)
            
            if step % render_every == 0:
                # renderer.update_scene(data)
                renderer.update_scene(data, camera="side")
                frame = renderer.render()
                
                # Convert frame to BGR for FFmpeg
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Write frame to FFmpeg process
                process.stdin.write(frame_bgr.tobytes())
            
            step += 1

    except KeyboardInterrupt:
        print("Streaming interrupted by user.")

    finally:
        # Close FFmpeg process
        process.stdin.close()
        process.wait()
        print(f"Streaming ended after {step} steps.")


def render_random_rollout(
    env: mujoco.MjModel,
    n_steps: int = 100,
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
    for step in tqdm(range(n_steps)):
        # collect random actions
        ctrl = jax.random.uniform(seed, (model.nu,), minval=-1, maxval=1)
        print(ctrl)
        data.ctrl = ctrl
        for _ in range(env._n_frames):
            mujoco.mj_step(model, data)

        if step % render_every == 0:
            renderer.update_scene(data, camera="side")
            images.append(renderer.render())

    return np.array(images)
