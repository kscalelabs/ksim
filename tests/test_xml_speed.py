# Throw a ball at 100 different velocities.

import jax
import mujoco
import numpy as np
from mujoco import mjx
from pathlib import Path
import os
import glfw
from PIL import Image
import time
from functools import partial


XML=r"""
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <camera pos="3 3 3" xyaxes="1 1 0 -1 1 2"/>
    <body>
      <freejoint/>
      <geom size=".15" mass="1" type="sphere" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
mjx_model = mjx.put_model(model)

@partial(jax.jit, static_argnums=(1,))
def simulate_steps(init_vel, num_steps):
    mjx_data = mjx.make_data(mjx_model)
    # Initialize with floating point values to ensure consistent types.
    init_qpos = jax.numpy.array([-2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=jax.numpy.float32)
    init_qvel = jax.numpy.array([init_vel, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=jax.numpy.float32)
    mjx_data = mjx_data.replace(qpos=init_qpos, qvel=init_qvel)
    
    mjx_data = mjx.forward(mjx_model, mjx_data)
    
    def step_fn(carry, _):
        mjx_data = carry
        next_data = mjx.step(mjx_model, mjx_data)
        return next_data, (next_data.qpos, next_data.qvel)
    
    final_state, (qpos_hist, qvel_hist) = jax.lax.scan(
        step_fn, mjx_data, None, length=num_steps
    )
    
    return qpos_hist, qvel_hist

def main():
    # Initialize GLFW and create OpenGL context
    if not glfw.init():
        raise RuntimeError('Failed to initialize GLFW')
    
    # Create an invisible window to get OpenGL context
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(640, 480, "hidden", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError('Failed to create GLFW window')
    
    glfw.make_context_current(window)

    # Setup output directory
    output_dir = Path("ball_trajectory")
    frames_dir = output_dir / "frames" / time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    try:
        # Initialize MuJoCo visualization
        cam = mujoco.MjvCamera()
        cam.trackbodyid = -1  # Disable automatic tracking
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE  # Use free camera mode
        # Set the camera position and lookat target so the ball is visible.
        # cam.pos = np.array([2.0, 2.0, 2.5])
        cam.lookat = np.array([-1.0, 0.0, 1.0])
        
        opt = mujoco.MjvOption()
        scene = mujoco.MjvScene(model, maxgeom=10000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # Simulate with different initial velocities
        velocities = [2.0, 4.0, 6.0, 8.0]
        num_steps = 50

        frame_idx = 0
        for vel in velocities:
            qpos_hist, qvel_hist = simulate_steps(vel, num_steps)
            
            # Render frames for this trajectory
            for qpos, qvel in zip(qpos_hist, qvel_hist):
                # Create fresh MjData for rendering
                render_data = mujoco.MjData(model)
                render_data.qpos = np.array(qpos)
                render_data.qvel = np.array(qvel)

                # Update scene
                mujoco.mj_forward(model, render_data)
                mujoco.mjv_updateScene(
                    model,
                    render_data,
                    opt,
                    None,
                    cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    scene,
                )

                # Render and save frame
                viewport = mujoco.MjrRect(0, 0, 640, 480)
                mujoco.mjr_render(viewport, scene, context)
                
                # Read pixels
                rgb_arr = np.empty((480, 640, 3), dtype=np.uint8)
                mujoco.mjr_readPixels(rgb_arr, None, viewport, context)
                
                # Save frame as PNG
                frame_path = frames_dir / f"vel_{vel}_frame_{frame_idx:06d}.png"
                rgb_arr = np.flipud(rgb_arr)  # Flip image vertically
                Image.fromarray(rgb_arr).save(frame_path)
                frame_idx += 1

        print(f"Saved frames to {frames_dir}")

    finally:
        # Cleanup
        glfw.destroy_window(window)
        glfw.terminate()

if __name__ == "__main__":
    main()