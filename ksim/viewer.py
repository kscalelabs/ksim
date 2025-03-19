"""MuJoCo viewer implementation with interactive visualization capabilities.

This module provides a viewer for MuJoCo environments with support for:
- Interactive camera control
- Real-time visualization of physics simulation
- Overlay information display
- Plotting capabilities
- Screenshot capture
- Custom marker visualization
"""

from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import glfw
import mujoco
import numpy as np
import yaml

MUJOCO_VERSION = tuple(map(int, mujoco.__version__.split(".")))


@dataclass
class ViewerConfig:
    """Configuration for the MuJoCo viewer.

    Attributes:
        hide_menus: Whether to hide the overlay menus by default
        render_mode: Either "window" or "offscreen" rendering mode
        title: Window title for window mode
        width: Window width (optional)
        height: Window height (optional)
        config_path: Path to save/load camera configuration
    """

    hide_menus: bool = False
    render_mode: str = "window"
    title: str = "mujoco-python-viewer"
    width: Optional[int] = None
    height: Optional[int] = None
    config_path: pathlib.Path = pathlib.Path.home() / ".config/mujoco_viewer/config.yaml"


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
        self._last_left_click_time: Optional[float] = None
        self._last_right_click_time: Optional[float] = None
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._paused = False
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
        window: Any,
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
            if key == glfw.KEY_LEFT_ALT:
                self._hide_menus = False
            return

        match key:
            case glfw.KEY_TAB:
                self._handle_camera_switch()
            case glfw.KEY_SPACE if self._paused is not None:
                self._paused = not self._paused
            case glfw.KEY_RIGHT if self._paused is not None:
                self._advance_by_one_step = True
                self._paused = True
            case glfw.KEY_S if mods != glfw.MOD_CONTROL:
                self._run_speed /= 2.0
            case glfw.KEY_F:
                self._run_speed *= 2.0
            case glfw.KEY_D:
                self._render_every_frame = not self._render_every_frame
            case glfw.KEY_T:
                self._capture_screenshot()
            case glfw.KEY_C:
                self._toggle_contacts()
            case glfw.KEY_J:
                self._toggle_joints()
            case glfw.KEY_E:
                self._cycle_frame_visualization()
            case glfw.KEY_LEFT_ALT:
                self._hide_menus = True
            case glfw.KEY_H:
                self._hide_menus = not self._hide_menus
            case glfw.KEY_R:
                self._toggle_transparency()
            case glfw.KEY_G:
                self._hide_graph = not self._hide_graph
            case glfw.KEY_I:
                self._toggle_inertia()
            case glfw.KEY_M:
                self._toggle_com()
            case glfw.KEY_O:
                self._toggle_shadows()
            case glfw.KEY_V:
                self._toggle_convex_hull()
            case glfw.KEY_W:
                self._toggle_wireframe()
            case glfw.KEY_S if mods == glfw.MOD_CONTROL:
                self._save_camera_config()
            case glfw.KEY_ESCAPE:
                self._handle_quit()
            case _ if key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5):
                self._toggle_geom_group(key)

    def _cursor_pos_callback(self, window: Any, xpos: float, ypos: float) -> None:
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

    def _mouse_button_callback(self, window: Any, button: int, act: int, mods: int) -> None:
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

    def _scroll_callback(self, window: Any, x_offset: float, y_offset: float) -> None:
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
                newperturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if self._button_left_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_ROTATE

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

        if MUJOCO_VERSION >= (3, 0, 0):
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
        else:
            selbody = mujoco.mjv_select(
                self.model, self.data, self.vopt, aspectratio, relx, rely, self.scn, selpnt, selgeom, selskin
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
        mode: str = "window",
        title: str = "mujoco-python-viewer",
        width: Optional[int] = None,
        height: Optional[int] = None,
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
        if self.render_mode not in ["offscreen", "window"]:
            raise NotImplementedError("Invalid mode. Only 'offscreen' and 'window' are supported.")

        self.is_alive = True
        self.CONFIG_PATH = pathlib.Path.joinpath(pathlib.Path.home(), ".config/mujoco_viewer/config.yaml")

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

        # Initialize plotting figures
        self._setup_figures(width, height)

        # Load camera configuration
        self._load_camera_config()

        # Initialize overlay and markers
        self._overlay: Dict[int, List[str]] = {}
        self._markers: List[Dict[str, Any]] = []

    def _setup_figures(self, width: int, height: int) -> None:
        """Set up plotting figures.

        Args:
            width: Window width
            height: Window height
        """
        max_num_figs = 3
        self.figs = []
        width_adjustment = width % 4
        fig_w, fig_h = int(width / 4), int(height / 4)
        for idx in range(max_num_figs):
            fig = mujoco.MjvFigure()
            mujoco.mjv_defaultFigure(fig)
            fig.flg_extend = 1
            self.figs.append(fig)

    def _load_camera_config(self) -> None:
        """Load camera configuration from file."""
        pathlib.Path(self.CONFIG_PATH.parent).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.CONFIG_PATH).touch(exist_ok=True)
        with open(self.CONFIG_PATH, "r") as f:
            try:
                cam_config = {
                    "type": self.cam.type,
                    "fixedcamid": self.cam.fixedcamid,
                    "trackbodyid": self.cam.trackbodyid,
                    "lookat": self.cam.lookat.tolist(),
                    "distance": self.cam.distance,
                    "azimuth": self.cam.azimuth,
                    "elevation": self.cam.elevation,
                }
                load_config = yaml.safe_load(f)
                if isinstance(load_config, dict):
                    for key, val in load_config.items():
                        if key in cam_config:
                            cam_config[key] = val
                if cam_config["type"] == mujoco.mjtCamera.mjCAMERA_FIXED:
                    if cam_config["fixedcamid"] < self.model.ncam:
                        self.cam.type = cam_config["type"]
                        self.cam.fixedcamid = cam_config["fixedcamid"]
                if cam_config["type"] == mujoco.mjtCamera.mjCAMERA_TRACKING:
                    if cam_config["trackbodyid"] < self.model.nbody:
                        self.cam.type = cam_config["type"]
                        self.cam.trackbodyid = cam_config["trackbodyid"]
                self.cam.lookat = np.array(cam_config["lookat"])
                self.cam.distance = cam_config["distance"]
                self.cam.azimuth = cam_config["azimuth"]
                self.cam.elevation = cam_config["elevation"]
            except yaml.YAMLError as e:
                print(e)

    def add_line_to_fig(self, line_name: str, fig_idx: int = 0) -> None:
        """Add a new line to a figure.

        Args:
            line_name: Name of the line
            fig_idx: Index of the figure to add the line to
        """
        assert isinstance(line_name, str), "Line name must be a string."

        fig = self.figs[fig_idx]
        if line_name.encode("utf8") == b"":
            raise Exception("Line name cannot be empty.")
        if line_name.encode("utf8") in fig.linename:
            raise Exception("Line name already exists in this plot.")

        linecount = fig.linename.tolist().index(b"")
        fig.linename[linecount] = line_name

        for i in range(mujoco.mjMAXLINEPNT):
            fig.linedata[linecount][2 * i] = -float(i)

    def add_data_to_line(self, line_name: str, line_data: float, fig_idx: int = 0) -> None:
        """Add data point to an existing line.

        Args:
            line_name: Name of the line
            line_data: Data point to add
            fig_idx: Index of the figure containing the line
        """
        fig = self.figs[fig_idx]

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

    def add_marker(self, **marker_params: Any) -> None:
        """Add a marker to the scene.

        Args:
            **marker_params: Parameters for the marker
        """
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker: Dict[str, Any]) -> None:
        """Add a marker to the current scene.

        Args:
            marker: Marker parameters
        """
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(f"Ran out of geoms. maxgeom: {self.scn.maxgeom}")

        g = self.scn.geoms[self.scn.ngeom]
        # Set default values
        g.dataid = -1
        g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mujoco.mjtCatBit.mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
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
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(f"mjtGeom has attr {key} but type {type(value)} is invalid")
            else:
                raise ValueError(f"mjtGeom doesn't have field {key}")

        self.scn.ngeom += 1

    def _create_overlay(self) -> None:
        """Create overlay text for the current frame."""
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        topright = mujoco.mjtGridPos.mjGRID_TOPRIGHT
        bottomleft = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT
        bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT

        def add_overlay(gridpos: int, text1: str, text2: str) -> None:
            if gridpos not in self._overlay:
                self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1 + "\n"
            self._overlay[gridpos][1] += text2 + "\n"

        # Add simulation control info
        if self._render_every_frame:
            add_overlay(topleft, "", "")
        else:
            add_overlay(topleft, f"Run speed = {self._run_speed:.3f} x real time", "[S]lower, [F]aster")
        add_overlay(topleft, "Ren[d]er every frame", "On" if self._render_every_frame else "Off")
        add_overlay(
            topleft,
            f"Switch camera (#cams = {self.model.ncam + 1})",
            f"[Tab] (camera ID = {self.cam.fixedcamid})",
        )

        # Add visualization toggles
        add_overlay(topleft, "[C]ontact forces", "On" if self._contacts else "Off")
        add_overlay(topleft, "[J]oints", "On" if self._joints else "Off")
        add_overlay(topleft, "[G]raph Viewer", "Off" if self._hide_graph else "On")
        add_overlay(topleft, "[I]nertia", "On" if self._inertias else "Off")
        add_overlay(topleft, "Center of [M]ass", "On" if self._com else "Off")
        add_overlay(topleft, "Shad[O]ws", "On" if self._shadows else "Off")
        add_overlay(topleft, "T[r]ansparent", "On" if self._transparent else "Off")
        add_overlay(topleft, "[W]ireframe", "On" if self._wire_frame else "Off")
        add_overlay(
            topleft,
            "Con[V]ex Hull Rendering",
            "On" if self._convex_hull_rendering else "Off",
        )

        # Add simulation state info
        if self._paused is not None:
            if not self._paused:
                add_overlay(topleft, "Stop", "[Space]")
            else:
                add_overlay(topleft, "Start", "[Space]")
                add_overlay(topleft, "Advance simulation by one step", "[right arrow]")

        # Add additional info
        add_overlay(
            topleft,
            "Toggle geomgroup visibility (0-5)",
            ",".join(["On" if g else "Off" for g in self.vopt.geomgroup]),
        )
        add_overlay(topleft, "Referenc[e] frames", mujoco.mjtFrame(self.vopt.frame).name)
        add_overlay(topleft, "[H]ide Menus", "")

        # Add capture frame info
        if self._image_idx > 0:
            fname = self._image_path % (self._image_idx - 1)
            add_overlay(topleft, "Cap[t]ure frame", f"Saved as {fname}")
        else:
            add_overlay(topleft, "Cap[t]ure frame", "")

        # Add performance metrics
        add_overlay(bottomleft, "FPS", f"{int(1 / self._time_per_render)}")

        if MUJOCO_VERSION >= (3, 0, 0):
            add_overlay(bottomleft, "Max solver iters", str(max(self.data.solver_niter) + 1))
        else:
            add_overlay(bottomleft, "Solver iterations", str(self.data.solver_iter + 1))

        add_overlay(bottomleft, "Step", str(round(self.data.time / self.model.opt.timestep)))
        add_overlay(bottomleft, "timestep", f"{self.model.opt.timestep:.5f}")

    def apply_perturbations(self) -> None:
        """Apply accumulated perturbations to the model."""
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def read_pixels(
        self, camid: Optional[int] = None, depth: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                mujoco.mjr_render(self.viewport, self.scn, self.ctx)

                # Render overlay
                for gridpos, [t1, t2] in self._overlay.items():
                    menu_positions = [mujoco.mjtGridPos.mjGRID_TOPLEFT, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT]
                    if gridpos in menu_positions and self._hide_menus:
                        continue

                    mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, gridpos, self.viewport, t1, t2, self.ctx)

                # Render figures
                if not self._hide_graph:
                    for idx, fig in enumerate(self.figs):
                        width_adjustment = width % 4
                        x = int(3 * width / 4) + width_adjustment
                        y = idx * int(height / 4)
                        viewport = mujoco.MjrRect(x, y, int(width / 4), int(height / 4))

                        has_lines = len([i for i in fig.linename if i != b""])
                        if has_lines:
                            mujoco.mjr_figure(viewport, fig, self.ctx)

                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (time.time() - render_start)

            self._overlay.clear()

        # Handle paused state
        if self._paused:
            while self._paused:
                update()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            # Handle running state
            self._loop_count += self.model.opt.timestep / (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        # Clear markers and apply perturbations
        self._markers[:] = []
        self.apply_perturbations()

    def close(self) -> None:
        """Close the viewer and clean up resources."""
        self.is_alive = False
        glfw.terminate()
        self.ctx.free()
