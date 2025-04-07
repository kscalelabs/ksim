"""MuJoCo viewer implementation with interactive visualization capabilities."""

import time
from threading import Lock
from typing import Callable, Literal, get_args, overload

import glfw
import mujoco
import numpy as np

from ksim.vis import configure_scene

RenderMode = Literal["window", "offscreen"]
Callback = Callable[[mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None]


class MujocoViewer:
    """Main viewer class for MuJoCo environments.

    This class provides a complete visualization interface for MuJoCo environments,
    including interactive camera control, real-time physics visualization, and
    various display options.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        mode: RenderMode = "window",
        title: str = "ksim",
        width: int | None = None,
        height: int | None = None,
        shadow: bool = False,
        reflection: bool = False,
        contact_force: bool = False,
        contact_point: bool = False,
        inertia: bool = False,
    ) -> None:
        """Initialize the MuJoCo viewer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            mode: Rendering mode ("window" or "offscreen")
            title: Window title for window mode
            width: Window width (optional)
            height: Window height (optional)
            shadow: Whether to show shadow
            reflection: Whether to show reflection
            contact_force: Whether to show contact force
            contact_point: Whether to show contact point
            inertia: Whether to show inertia
        """
        super().__init__()

        self._gui_lock = Lock()
        self._render_every_frame = True
        self._time_per_render = 1 / 60.0
        self._loop_count = 0
        self._advance_by_one_step = False

        self.model = model
        self.data = data
        self.render_mode = mode
        if self.render_mode not in get_args(RenderMode):
            raise NotImplementedError(f"Invalid mode: {self.render_mode}")

        self.is_alive = True

        # Initialize GLFW
        glfw.init()

        # Get window dimensions if not provided.
        if width is None:
            width, _ = glfw.get_video_mode(glfw.get_primary_monitor()).size
        if height is None:
            _, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
        assert width is not None and height is not None

        # Create window
        if self.render_mode == "offscreen":
            glfw.window_hint(glfw.VISIBLE, 0)
        self.window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # Get framebuffer dimensions
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)

        # Initialize MuJoCo visualization objects
        self.vopt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.pert = mujoco.MjvPerturb()
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        configure_scene(
            self.scn,
            self.vopt,
            shadow=shadow,
            reflection=reflection,
            contact_force=contact_force,
            contact_point=contact_point,
            inertia=inertia,
        )

        # Set up viewport
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)

        # Mouse interaction variables
        self._button_left = False
        self._button_right = False
        self._button_middle = False
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0

        # Set up mouse and keyboard callbacks
        if self.render_mode == "window":
            glfw.set_cursor_pos_callback(self.window, self._mouse_move)
            glfw.set_mouse_button_callback(self.window, self._mouse_button)
            glfw.set_scroll_callback(self.window, self._scroll)
            glfw.set_key_callback(self.window, self._keyboard)

    def _mouse_move(self, window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:
        """Mouse motion callback."""
        dx = xpos - self._last_mouse_x
        dy = ypos - self._last_mouse_y

        # Update mouse position
        self._last_mouse_x = xpos
        self._last_mouse_y = ypos

        # Left button: rotate camera
        if self._button_left:
            self.cam.azimuth -= dx * 0.5
            self.cam.elevation -= dy * 0.5

        # Right button: pan camera
        elif self._button_right:
            forward = np.array(
                [
                    np.cos(np.deg2rad(self.cam.azimuth)) * np.cos(np.deg2rad(self.cam.elevation)),
                    np.sin(np.deg2rad(self.cam.azimuth)) * np.cos(np.deg2rad(self.cam.elevation)),
                    np.sin(np.deg2rad(self.cam.elevation)),
                ]
            )
            right = np.array([-np.sin(np.deg2rad(self.cam.azimuth)), np.cos(np.deg2rad(self.cam.azimuth)), 0])
            up = np.cross(right, forward)

            # Scale pan speed with distance
            scale = self.cam.distance * 0.001

            self.cam.lookat[0] += right[0] * dx * scale - up[0] * dy * scale
            self.cam.lookat[1] += right[1] * dx * scale - up[1] * dy * scale
            self.cam.lookat[2] += right[2] * dx * scale - up[2] * dy * scale

    def _mouse_button(self, window: glfw._GLFWwindow, button: int, act: int, mods: int) -> None:
        """Mouse button callback."""
        self._button_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._button_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self._button_middle = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS

        # Update cursor position
        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = x
        self._last_mouse_y = y

    def _scroll(self, window: glfw._GLFWwindow, xoffset: float, yoffset: float) -> None:
        """Mouse scroll callback."""
        self.cam.distance *= 0.9 if yoffset > 0 else 1.1

    def _keyboard(self, window: glfw._GLFWwindow, key: int, scancode: int, act: int, mods: int) -> None:
        """Keyboard callback."""
        if act == glfw.PRESS and key == glfw.KEY_ESCAPE:
            self._handle_quit()

    def _handle_quit(self) -> None:
        glfw.set_window_should_close(self.window, True)

    @overload
    def read_pixels(self, depth: Literal[True], callback: Callback | None = None) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def read_pixels(self, depth: Literal[False] = False, callback: Callback | None = None) -> np.ndarray: ...

    def read_pixels(
        self,
        depth: bool = False,
        callback: Callback | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Read pixel data from the scene.

        Args:
            depth: Whether to also return depth buffer
            callback: Callback to call after rendering

        Returns:
            RGB image array, or tuple of (RGB, depth) arrays if depth=True
        """
        if self.render_mode == "window":
            raise NotImplementedError("Use 'render()' in 'window' mode.")

        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scn,
        )
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        if callback is not None:
            callback(self.model, self.data, self.scn)
        mujoco.mjr_render(self.viewport, self.scn, self.ctx)
        shape = glfw.get_framebuffer_size(self.window)

        if depth:
            rgb_img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            depth_img = np.zeros((shape[1], shape[0], 1), dtype=np.float32)
            mujoco.mjr_readPixels(rgb_img, depth_img, self.viewport, self.ctx)
            return np.flipud(rgb_img), np.flipud(depth_img)
        else:
            img = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
            mujoco.mjr_readPixels(img, None, self.viewport, self.ctx)
            return np.flipud(img)

    def render(self, callback: Callback | None = None) -> None:
        """Render a frame of the simulation."""
        if self.render_mode == "offscreen":
            raise NotImplementedError("Use 'read_pixels()' for 'offscreen' mode.")
        if not self.is_alive:
            raise Exception("GLFW window does not exist but you tried to render.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        def update() -> None:
            render_start = time.time()
            width, height = glfw.get_framebuffer_size(self.window)
            self.viewport.width, self.viewport.height = width, height

            with self._gui_lock:
                # Update and render scene
                mujoco.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.scn,
                )

                if callback is not None:
                    callback(self.model, self.data, self.scn)

                mujoco.mjr_render(self.viewport, self.scn, self.ctx)

                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (time.time() - render_start)

        # Handle running in real-time.
        self._loop_count += self.model.opt.timestep / self._time_per_render
        if self._render_every_frame:
            self._loop_count = 1
        while self._loop_count > 0:
            update()
            self._loop_count -= 1

    def close(self) -> None:
        """Close the viewer and clean up resources."""
        self.is_alive = False
        glfw.terminate()
        self.ctx.free()
