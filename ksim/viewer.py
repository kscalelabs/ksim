"""MuJoCo viewer implementation with interactive visualization capabilities."""

__all__ = [
    "DefaultMujocoViewer",
    "GlfwMujocoViewer",
]

import time
from threading import Lock
from typing import Callable, Literal, get_args

import glfw
import mujoco
import numpy as np

from ksim.vis import configure_scene

RenderMode = Literal["window", "offscreen"]
Callback = Callable[[mujoco.MjModel, mujoco.MjData, mujoco.MjvScene], None]


class DefaultMujocoViewer:
    """MuJoCo viewer implementation using offscreen OpenGL context."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        width: int = 320,
        height: int = 240,
        max_geom: int = 10000,
    ) -> None:
        """Initialize the default MuJoCo viewer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            width: Width of the viewer
            height: Height of the viewer
            max_geom: Maximum number of geoms to render
        """
        super().__init__()

        if data is None:
            data = mujoco.MjData(model)

        self.model = model
        self.data = data
        self.width = width
        self.height = height

        # Validate framebuffer size
        if width > model.vis.global_.offwidth or height > model.vis.global_.offheight:
            raise ValueError(
                f"Image size ({width}x{height}) exceeds offscreen buffer size "
                f"({model.vis.global_.offwidth}x{model.vis.global_.offheight}). "
                "Increase `offwidth`/`offheight` in the XML model."
            )

        # Offscreen rendering context
        self._gl_context = mujoco.gl_context.GLContext(width, height)
        self._gl_context.make_current()

        # MuJoCo scene setup
        self.scn = mujoco.MjvScene(model, maxgeom=max_geom)
        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()
        self.rect = mujoco.MjrRect(0, 0, width, height)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.cam)

        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)

    def set_camera(self, id: int | str) -> None:
        """Set the camera to use."""
        if isinstance(id, int):
            if id < -1 or id >= self.model.ncam:
                raise ValueError(f"Camera ID {id} is out of range [-1, {self.model.ncam}).")
            # Set up camera
            self.cam.fixedcamid = id
            if id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                mujoco.mjv_defaultFreeCamera(self.model, self.cam)
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        elif isinstance(id, str):
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, id)
            if camera_id == -1:
                raise ValueError(f'The camera "{id}" does not exist.')
            # Set up camera
            self.cam.fixedcamid = camera_id
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        else:
            raise ValueError(f"Invalid camera ID: {id}")

    def read_pixels(self, callback: Callback | None = None) -> np.ndarray:
        self._gl_context.make_current()

        # Update scene.
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

        # Render.
        mujoco.mjr_render(self.rect, self.scn, self.ctx)

        # Read pixels.
        rgb_array = np.empty((self.height, self.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb_array, None, self.rect, self.ctx)
        return np.flipud(rgb_array)

    def render(self, callback: Callback | None = None) -> None:
        raise NotImplementedError("Default viewer does not support rendering.")

    def close(self) -> None:
        if self._gl_context:
            self._gl_context.free()
            self._gl_context = None
        if self.ctx:
            self.ctx.free()
            self.ctx = None


class GlfwMujocoViewer:
    """Main viewer class for MuJoCo environments.

    This class provides a complete visualization interface for MuJoCo environments,
    including interactive camera control, real-time physics visualization, and
    various display options.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
        mode: RenderMode = "window",
        title: str = "ksim",
        width: int | None = None,
        height: int | None = None,
        shadow: bool = False,
        reflection: bool = False,
        contact_force: bool = False,
        contact_point: bool = False,
        inertia: bool = False,
        max_geom: int = 10000,
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
            max_geom: Maximum number of geoms to render
        """
        super().__init__()

        self._gui_lock = Lock()
        self._render_every_frame = True
        self._time_per_render = 1 / 60.0
        self._loop_count = 0
        self._advance_by_one_step = False

        if data is None:
            data = mujoco.MjData(model)

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
        self.scn = mujoco.MjvScene(self.model, maxgeom=max_geom)
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
        self.rect = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)

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

    def set_camera(self, id: int | str) -> None:
        """Set the camera to use."""
        if isinstance(id, int):
            if id < -1 or id >= self.model.ncam:
                raise ValueError(f"Camera ID {id} is out of range [-1, {self.model.ncam}).")
            # Set up camera
            self.cam.fixedcamid = id
            if id == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                mujoco.mjv_defaultFreeCamera(self.model, self.cam)
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        elif isinstance(id, str):
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, id)
            if camera_id == -1:
                raise ValueError(f'The camera "{id}" does not exist.')
            # Set up camera
            self.cam.fixedcamid = camera_id
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        else:
            raise ValueError(f"Invalid camera ID: {id}")

    def _mouse_move(self, window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:
        """Mouse motion callback."""
        dx = xpos - self._last_mouse_x
        dy = ypos - self._last_mouse_y

        # Update mouse position
        self._last_mouse_x = xpos
        self._last_mouse_y = ypos

        # If already applying a perturbation force
        if self.pert.active:
            # Get framebuffer size to normalize mouse movement.
            width, height = glfw.get_framebuffer_size(window)
            mod_shift = (
                glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
            )
            if self._button_right:
                action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
            elif self._button_left:
                action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            else:
                action = mujoco.mjtMouse.mjMOUSE_ZOOM

            with self._gui_lock:
                mujoco.mjv_movePerturb(
                    self.model,
                    self.data,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.pert,
                )
            return

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

        # Apply perturbation forces
        ctrl_pressed = mods & glfw.MOD_CONTROL

        if act == glfw.PRESS and ctrl_pressed:
            # Get framebuffer size to normalize mouse movement.
            width, height = glfw.get_framebuffer_size(window)
            aspectratio = width / height
            relx = x / width
            rely = (height - y) / height

            # Select a body to perturb
            selpnt = np.zeros(3)
            selgeom = np.zeros(1, dtype=np.int32)
            selskin = np.zeros(1, dtype=np.int32)

            # Calls the MuJoCo's select function to find the relevant body
            selbody = mujoco.mjv_select(
                self.model,
                self.data,
                self.vopt,
                aspectratio,
                relx,
                rely,
                self.scn,
                selpnt,
                selgeom,
                np.zeros(1, dtype=np.int32),
                selskin,
            )

            if selbody >= 0:
                self.pert.select = selbody
                self.pert.skinselect = int(selskin[0])
                vec = selpnt - self.data.xpos[selbody]
                self.pert.localpos = self.data.xmat[selbody].reshape(3, 3).dot(vec)

            # Set the perturbation type
            newperturb = 0
            if selbody >= 0:
                if button == glfw.MOUSE_BUTTON_RIGHT:
                    newperturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
                elif button == glfw.MOUSE_BUTTON_LEFT:
                    newperturb = mujoco.mjtPertBit.mjPERT_ROTATE

            # Initialize the perturbation
            if newperturb and not self.pert.active:
                mujoco.mjv_initPerturb(self.model, self.data, self.scn, self.pert)
            self.pert.active = newperturb

        # On release, stop perturbation
        if act == glfw.RELEASE:
            self.pert.active = 0

    def _scroll(self, window: glfw._GLFWwindow, xoffset: float, yoffset: float) -> None:
        """Mouse scroll callback."""
        self.cam.distance *= 0.9 if yoffset > 0 else 1.1

    def _keyboard(self, window: glfw._GLFWwindow, key: int, scancode: int, act: int, mods: int) -> None:
        """Keyboard callback."""
        if act == glfw.PRESS and key == glfw.KEY_ESCAPE:
            self._handle_quit()

    def _handle_quit(self) -> None:
        glfw.set_window_should_close(self.window, True)

    def read_pixels(self, callback: Callback | None = None) -> np.ndarray:
        if self.render_mode == "window":
            raise NotImplementedError("Use 'render()' in 'window' mode.")

        # Update scene.
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

        # Render.
        mujoco.mjr_render(self.rect, self.scn, self.ctx)

        # Read pixels.
        rgb_array = np.empty((self.rect.height, self.rect.width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb_array, None, self.rect, self.ctx)
        return np.flipud(rgb_array)

    def render(self, callback: Callback | None = None) -> None:
        if self.render_mode == "offscreen":
            raise NotImplementedError("Use 'read_pixels()' for 'offscreen' mode.")
        if not self.is_alive:
            raise Exception("GLFW window does not exist but you tried to render.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        def update() -> None:
            render_start = time.monotonic()
            width, height = glfw.get_framebuffer_size(self.window)
            self.rect.width, self.rect.height = width, height

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

                mujoco.mjr_render(self.rect, self.scn, self.ctx)

                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (time.monotonic() - render_start)

        # Handle running in real-time.
        self._loop_count += self.model.opt.timestep / self._time_per_render
        if self._render_every_frame:
            self._loop_count = 1
        while self._loop_count > 0:
            update()
            self._loop_count -= 1

        self.apply_perturbations()

    def close(self) -> None:
        """Close the viewer and clean up resources."""
        self.is_alive = False
        glfw.terminate()
        self.ctx.free()

    def apply_perturbations(self) -> None:
        """Apply user perturbations (via Ctrl+click and drag) to the simulation."""
        self.data.xfrc_applied[:] = 0
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)
