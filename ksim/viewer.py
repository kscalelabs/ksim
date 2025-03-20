"""MuJoCo viewer implementation with interactive visualization capabilities."""

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, get_args

import glfw
import mujoco
import numpy as np

RenderMode = Literal["window", "offscreen"]


@dataclass
class ViewerConfig:
    hide_menus: bool = False
    render_mode: RenderMode = "window"
    title: str = "ksim"
    width: int | None = None
    height: int | None = None


class OverlayManager:
    """Manages the overlay text display in the viewer."""

    def __init__(self) -> None:
        """Initialize the overlay manager."""
        self._overlay: dict[int, list[str]] = {}
        self._grid_positions = {
            "topleft": mujoco.mjtGridPos.mjGRID_TOPLEFT,
            "topright": mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "bottomleft": mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            "bottomright": mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
        }

    def add_text(self, gridpos: int, text1: str, text2: str, newline: bool = True) -> None:
        """Add text to the overlay.

        Args:
            gridpos: Grid position to add text to
            text1: Primary text
            text2: Secondary text
            newline: Whether to add a newline to the text
        """
        if newline:
            text1 += "\n"
            text2 += "\n"

        if gridpos not in self._overlay:
            self._overlay[gridpos] = ["", ""]
        self._overlay[gridpos][0] += text1
        self._overlay[gridpos][1] += text2

    def clear(self) -> None:
        """Clear all overlay text."""
        self._overlay.clear()

    def get_overlay(self) -> dict[int, list[str]]:
        """Get the current overlay text.

        Returns:
            Dictionary mapping grid positions to text pairs
        """
        return self._overlay


class MarkerManager:
    """Manages visualization markers in the scene."""

    def __init__(self, scene: mujoco.MjvScene) -> None:
        """Initialize the marker manager.

        Args:
            scene: MuJoCo scene object
        """
        self._scene = scene
        self._markers: list[dict[str, Any]] = []  # noqa: ANN401

    def add_marker(self, **marker_params: Any) -> None:  # noqa: ANN401
        """Add a marker to the scene.

        Args:
            **marker_params: Parameters for the marker
        """
        self._markers.append(marker_params)

    def clear(self) -> None:
        """Clear all markers."""
        self._markers[:] = []

    def render(self) -> None:
        """Render all markers in the scene."""
        for marker in self._markers:
            self._add_marker_to_scene(marker)

    def _add_marker_to_scene(self, marker: dict[str, Any]) -> None:  # noqa: ANN401
        """Add a marker to the current scene.

        Args:
            marker: Marker parameters
        """
        if self._scene.ngeom >= self._scene.maxgeom:
            raise RuntimeError(f"Ran out of geoms. maxgeom: {self._scene.maxgeom}")

        g = self._scene.geoms[self._scene.ngeom]
        # Set default values
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.matid = -1
        # g.texuniform = 0
        # g.texrepeat[0] = 1
        # g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        # Apply marker parameters
        for key, value in marker.items():
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, bytes):
                setattr(g, key, value)
            else:
                raise ValueError(
                    f"Invalid type for attribute '{key}': {type(value)}"
                )

        self._scene.ngeom += 1


class PlotManager:
    """Manages plotting figures in the viewer."""

    def __init__(self, width: int, height: int) -> None:
        """Initialize the plot manager.

        Args:
            width: Window width
            height: Window height
        """
        self._width = width
        self._height = height
        self._figs: list[mujoco.MjvFigure] = []
        self._setup_figures()

    def _setup_figures(self) -> None:
        """Set up plotting figures."""
        max_num_figs = 3
        for _ in range(max_num_figs):
            fig = mujoco.MjvFigure()
            mujoco.mjv_defaultFigure(fig)
            fig.flg_extend = 1
            self._figs.append(fig)

    def add_line(self, line_name: str, fig_idx: int = 0) -> None:
        """Add a new line to a figure.

        Args:
            line_name: Name of the line
            fig_idx: Index of the figure to add the line to
        """
        assert isinstance(line_name, str), "Line name must be a string."

        fig = self._figs[fig_idx]
        if line_name.encode("utf8") == b"":
            raise Exception("Line name cannot be empty.")
        if line_name.encode("utf8") in fig.linename:
            raise Exception("Line name already exists in this plot.")

        linecount = fig.linename.tolist().index(b"")
        fig.linename[linecount] = line_name

        for i in range(mujoco.mjMAXLINEPNT):
            fig.linedata[linecount][2 * i] = -float(i)

    def add_data(self, line_name: str, line_data: float, fig_idx: int = 0) -> None:
        """Add data point to an existing line.

        Args:
            line_name: Name of the line
            line_data: Data point to add
            fig_idx: Index of the figure containing the line
        """
        fig = self._figs[fig_idx]

        try:
            _line_name = line_name.encode("utf8")
            linenames = fig.linename.tolist()
            line_idx = linenames.index(_line_name)
        except ValueError:
            raise Exception("line name is not valid, add it to list before calling update")

        pnt = min(mujoco.mjMAXLINEPNT, fig.linepnt[line_idx] + 1)
        for i in range(pnt - 1, 0, -1):
            fig.linedata[line_idx][2 * i + 1] = fig.linedata[line_idx][2 * i - 1]

        fig.linepnt[line_idx] = pnt
        fig.linedata[line_idx][1] = line_data

    def render(self, ctx: mujoco.MjrContext, hide_graph: bool) -> None:
        """Render all figures.

        Args:
            ctx: MuJoCo rendering context
            hide_graph: Whether to hide the graph
        """
        if hide_graph:
            return

        for idx, fig in enumerate(self._figs):
            width_adjustment = self._width % 4
            x = int(3 * self._width / 4) + width_adjustment
            y = idx * int(self._height / 4)
            viewport = mujoco.MjrRect(x, y, int(self._width / 4), int(self._height / 4))

            has_lines = len([i for i in fig.linename if i != b""])
            if has_lines:
                mujoco.mjr_figure(viewport, fig, ctx)


class Callbacks:
    """Handles user input callbacks for the MuJoCo viewer.

    This class manages keyboard, mouse, and scroll input events and their effects
    on the visualization, such as camera movement, simulation control, and display options.
    """

    def __init__(self, hide_menus: bool = False) -> None:
        """Initialize the callbacks handler.

        Args:
            hide_menus: Whether to hide overlay menus by default
        """
        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        self._last_left_click_time: float | None = None
        self._last_right_click_time: float | None = None
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._hide_graph = False
        self._transparent = False
        self._contacts = False
        self._joints = False
        self._shadows = True
        self._wire_frame = False
        self._convex_hull_rendering = False
        self._inertias = False
        self._com = False
        self._render_every_frame = True
        self._image_idx = 0
        self._image_path = "/tmp/frame_%07d.png"
        self._time_per_render = 1 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        self._advance_by_one_step = False
        self._hide_menus = hide_menus

    def _key_callback(
        self,
        window: glfw._GLFWwindow,
        key: int,
        scancode: int,
        action: int,
        mods: int,
    ) -> None:
        """Handle keyboard input events.

        Args:
            window: GLFW window handle
            key: Key code
            scancode: Platform-specific key code
            action: Key action (press/release)
            mods: Modifier keys state
        """
        if action != glfw.RELEASE:
            return

        match key:
            case glfw.KEY_ESCAPE:
                self._handle_quit()
            case _:
                pass

    def _cursor_pos_callback(self, window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:
        """Handle mouse movement events.

        Args:
            window: GLFW window handle
            xpos: Mouse x position
            ypos: Mouse y position
        """
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        if self._button_right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self._gui_lock:
            if self.pert.active:
                mujoco.mjv_movePerturb(self.model, self.data, action, dx / height, dy / height, self.scn, self.pert)
            else:
                mujoco.mjv_moveCamera(self.model, action, dx / height, dy / height, self.scn, self.cam)

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window: glfw._GLFWwindow, button: int, act: int, mods: int) -> None:
        """Handle mouse button events.

        Args:
            window: GLFW window handle
            button: Mouse button code
            act: Button action (press/release)
            mods: Modifier keys state
        """
        self._button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        self._button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

        # Detect double clicks
        self._detect_double_clicks(button, act)

        # Handle perturbation
        self._handle_perturbation(button, act, mods)

        # Handle double click selection
        if self._left_double_click_pressed or self._right_double_click_pressed:
            self._handle_double_click_selection(x, y)

        # Reset perturbation on release
        if act == glfw.RELEASE:
            self.pert.active = 0

    def _scroll_callback(self, window: glfw._GLFWwindow, x_offset: float, y_offset: float) -> None:
        """Handle mouse scroll events.

        Args:
            window: GLFW window handle
            x_offset: Horizontal scroll offset
            y_offset: Vertical scroll offset
        """
        with self._gui_lock:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset, self.scn, self.cam)

    def _detect_double_clicks(self, button: int, act: int) -> None:
        """Detect double click events.

        Args:
            button: Mouse button code
            act: Button action (press/release)
        """
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        time_now = glfw.get_time()

        if self._button_left_pressed:
            if self._last_left_click_time is None:
                self._last_left_click_time = time_now
            time_diff = time_now - self._last_left_click_time
            if 0.01 < time_diff < 0.3:
                self._left_double_click_pressed = True
            self._last_left_click_time = time_now

        if self._button_right_pressed:
            if self._last_right_click_time is None:
                self._last_right_click_time = time_now
            time_diff = time_now - self._last_right_click_time
            if 0.01 < time_diff < 0.2:
                self._right_double_click_pressed = True
            self._last_right_click_time = time_now

    def _handle_perturbation(self, button: int, act: int, mods: int) -> None:
        """Handle perturbation of selected objects.

        Args:
            button: Mouse button code
            act: Button action (press/release)
            mods: Modifier keys state
        """
        key = mods == glfw.MOD_CONTROL
        newperturb = 0
        if key and self.pert.select > 0:
            if self._button_right_pressed:
                newperturb = mujoco.mjtPerturbBit.mjPERT_TRANSLATE
            if self._button_left_pressed:
                newperturb = mujoco.mjtPerturbBit.mjPERT_ROTATE

            if newperturb and not self.pert.active:
                mujoco.mjv_initPerturb(self.model, self.data, self.scn, self.pert)
        self.pert.active = newperturb

    def _handle_double_click_selection(self, x: float, y: float) -> None:
        """Handle object selection on double click.

        Args:
            x: Mouse x position
            y: Mouse y position
        """
        # Determine selection mode
        selmode = 0
        if self._left_double_click_pressed:
            selmode = 1
        if self._right_double_click_pressed:
            selmode = 2
        if self._right_double_click_pressed and self.pert.active:
            selmode = 3

        # Find selected object
        width, height = self.viewport.width, self.viewport.height
        aspectratio = width / height
        relx = x / width
        rely = (self.viewport.height - y) / height
        selpnt = np.zeros((3, 1), dtype=np.float64)
        selgeom = np.zeros((1, 1), dtype=np.int32)
        selflex = np.zeros((1, 1), dtype=np.int32)
        selskin = np.zeros((1, 1), dtype=np.int32)

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
            selflex,
            selskin,
        )

        # Handle selection
        if selmode in (2, 3):
            if selbody >= 0:
                self.cam.lookat = selpnt.flatten()
            if selmode == 3 and selbody > 0:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.cam.trackbodyid = selbody
                self.cam.fixedcamid = -1
        elif selbody >= 0:
            self.pert.select = selbody
            self.pert.skinselect = selskin
            vec = selpnt.flatten() - self.data.xpos[selbody]
            mat = self.data.xmat[selbody].reshape(3, 3)
            self.pert.localpos = mat.dot(vec)
        else:
            self.pert.select = 0
            self.pert.skinselect = -1

        self.pert.active = 0


class MujocoViewer(Callbacks):
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
        title: str = "mujoco-python-viewer",
        width: int | None = None,
        height: int | None = None,
        hide_menus: bool = False,
    ) -> None:
        """Initialize the MuJoCo viewer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            mode: Rendering mode ("window" or "offscreen")
            title: Window title for window mode
            width: Window width (optional)
            height: Window height (optional)
            hide_menus: Whether to hide overlay menus by default
        """
        super().__init__(hide_menus)

        self.model = model
        self.data = data
        self.render_mode = mode
        if self.render_mode not in get_args(RenderMode):
            raise NotImplementedError(f"Invalid mode: {self.render_mode}")

        self.is_alive = True

        # Initialize GLFW
        glfw.init()

        # Set window dimensions
        if not width:
            width, _ = glfw.get_video_mode(glfw.get_primary_monitor()).size
        if not height:
            _, height = glfw.get_video_mode(glfw.get_primary_monitor()).size

        # Create window
        if self.render_mode == "offscreen":
            glfw.window_hint(glfw.VISIBLE, 0)
        self.window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # Get framebuffer dimensions
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)

        # Set up callbacks for window mode
        if self.render_mode == "window":
            window_width, _ = glfw.get_window_size(self.window)
            self._scale = framebuffer_width * 1.0 / window_width

            glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
            glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
            glfw.set_key_callback(self.window, self._key_callback)

        # Initialize MuJoCo visualization objects
        self.vopt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.pert = mujoco.MjvPerturb()
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # Set up viewport
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)

        # Initialize managers
        self.overlay_manager = OverlayManager()
        self.marker_manager = MarkerManager(self.scn)
        self.plot_manager = PlotManager(width, height)

    def add_line_to_fig(self, line_name: str, fig_idx: int = 0) -> None:
        """Add a new line to a figure.

        Args:
            line_name: Name of the line
            fig_idx: Index of the figure to add the line to
        """
        self.plot_manager.add_line(line_name, fig_idx)

    def add_data_to_line(self, line_name: str, line_data: float, fig_idx: int = 0) -> None:
        """Add data point to an existing line.

        Args:
            line_name: Name of the line
            line_data: Data point to add
            fig_idx: Index of the figure containing the line
        """
        self.plot_manager.add_data(line_name, line_data, fig_idx)

    def add_marker(self, **marker_params: Any) -> None:  # noqa: ANN401
        """Add a marker to the scene.

        Args:
            **marker_params: Parameters for the marker
        """
        self.marker_manager.add_marker(**marker_params)

    def add_direct_geom(self, geom_type, size, pos, rgba) -> None:
        """Add a geometry directly to the scene, bypassing the marker manager.
        
        This is a simpler approach that directly manipulates the scene geometry.
        
        Args:
            geom_type: Type of geometry (e.g., mujoco.mjtGeom.mjGEOM_SPHERE)
            size: Size array for the geometry
            pos: Position [x, y, z]
            rgba: Color and transparency [r, g, b, a]
        """
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(f"Ran out of geoms. maxgeom: {self.scn.maxgeom}")
            
        g = self.scn.geoms[self.scn.ngeom]
        # Set default values
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.matid = -1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = geom_type
        
        # Set size, position and color
        g.size[:] = np.asarray(size).reshape(g.size.shape)
        g.pos[:] = np.asarray(pos).reshape(g.pos.shape)
        g.rgba[:] = np.asarray(rgba).reshape(g.rgba.shape)
        
        # Set default orientation (identity matrix)
        g.mat[:] = np.eye(3)
        
        # Increment the scene's geometry count
        self.scn.ngeom += 1

    def _create_overlay(self) -> None:
        """Create overlay text for the current frame."""
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT

        def add_overlay(gridpos: int, text1: str, text2: str, newline: bool = True) -> None:
            self.overlay_manager.add_text(gridpos, text1, text2, newline)

        # Add performance metrics
        add_overlay(bottomleft, "FPS", f"{int(1 / self._time_per_render)}")
        add_overlay(bottomleft, "Max solver iters", str(max(self.data.solver_niter) + 1))
        add_overlay(bottomleft, "Step", str(round(self.data.time / self.model.opt.timestep)))
        add_overlay(bottomleft, "timestep", f"{self.model.opt.timestep:.5f}", newline=False)

    def apply_perturbations(self) -> None:
        """Apply accumulated perturbations to the model."""
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def read_pixels(
        self,
        camid: int | None = None,
        depth: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Read pixel data from the scene.

        Args:
            camid: Camera ID to use (-1 for free camera)
            depth: Whether to also return depth buffer

        Returns:
            RGB image array, or tuple of (RGB, depth) arrays if depth=True
        """
        if self.render_mode == "window":
            raise NotImplementedError("Use 'render()' in 'window' mode.")

        if camid is not None:
            if camid == -1:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camid

        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
        mujoco.mjv_updateScene(
            self.model, self.data, self.vopt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scn
        )
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

    def render(self) -> None:
        """Render a frame of the simulation."""
        if self.render_mode == "offscreen":
            raise NotImplementedError("Use 'read_pixels()' for 'offscreen' mode.")
        if not self.is_alive:
            raise Exception("GLFW window does not exist but you tried to render.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        def update() -> None:
            self._create_overlay()
            render_start = time.time()

            width, height = glfw.get_framebuffer_size(self.window)
            self.viewport.width, self.viewport.height = width, height

            with self._gui_lock:
                # Update and render scene
                mujoco.mjv_updateScene(
                    self.model, self.data, self.vopt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value, self.scn
                )
                self.marker_manager.render()
                mujoco.mjr_render(self.viewport, self.scn, self.ctx)

                # Render overlay
                for gridpos, [t1, t2] in self.overlay_manager.get_overlay().items():
                    menu_positions = [mujoco.mjtGridPos.mjGRID_TOPLEFT, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT]
                    if gridpos in menu_positions and self._hide_menus:
                        continue

                    mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, gridpos, self.viewport, t1, t2, self.ctx)

                # Render figures
                self.plot_manager.render(self.ctx, self._hide_graph)

                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (time.time() - render_start)

            self.overlay_manager.clear()

        # Handle running state
        self._loop_count += self.model.opt.timestep / (self._time_per_render * self._run_speed)
        if self._render_every_frame:
            self._loop_count = 1
        while self._loop_count > 0:
            update()
            self._loop_count -= 1

        # Clear markers and apply perturbations
        self.marker_manager.clear()
        self.apply_perturbations()

    def close(self) -> None:
        """Close the viewer and clean up resources."""
        self.is_alive = False
        glfw.terminate()
        self.ctx.free()
