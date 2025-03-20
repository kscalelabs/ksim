""" Utilities for rendering the environment. """
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image

class MujocoViewerHandler:
    def __init__(self, handle: mujoco.viewer.Handle, capture_pixels: bool = False, render_width: int = 640, render_height: int = 480):
        self.handle = handle
        self._markers = []
        
        # Initialize renderer for pixel capture if requested
        self._capture_pixels = capture_pixels
        self._render_width = render_width
        self._render_height = render_height
        self._renderer = None
        
        # If we're going to capture pixels, initialize the renderer now
        if self._capture_pixels and self.handle.m is not None:
            self._renderer = mujoco.Renderer(self.handle.m, width=render_width, height=render_height)

    def setup_camera(self, config: dict):
        """Setup the camera with the given configuration."""
        self.handle.cam.distance = config.render_distance
        self.handle.cam.azimuth = config.render_azimuth
        self.handle.cam.elevation = config.render_elevation
        self.handle.cam.lookat[:] = config.render_lookat
        if config.render_track_body_id is not None:
            self.handle.cam.trackbodyid = config.render_track_body_id
            self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        
    def copy_data(self, dst: mujoco.MjData, src: mujoco.MjData):
        """Copy the data from the source to the destination."""
        dst.ctrl[:] = src.ctrl[:]
        dst.act[:] = src.act[:]
        dst.xfrc_applied[:] = src.xfrc_applied[:]
        dst.qpos[:] = src.qpos[:]
        dst.qvel[:] = src.qvel[:]
        dst.time = src.time
        
    def clear_markers(self):
        """Clear all markers from the scene."""
        if self.handle._user_scn is not None:
            # Reset the geom counter to effectively clear all markers
            self.handle._user_scn.ngeom = 0
            self._markers = []
    
    def add_marker(self, pos, size=(0.1, 0, 0), rgba=(1, 0, 0, 1), type=mujoco.mjtGeom.mjGEOM_SPHERE, mat=None, label=""):
        """Add a marker to be rendered in the scene."""
        self._markers.append({
            'pos': pos, 
            'size': size, 
            'rgba': rgba, 
            'type': type,
            'mat': np.eye(3) if mat is None else mat,
            'label': label if isinstance(label, bytes) else label.encode('utf8') if label else b''
        })
    
    def add_commands(self, commands):
        """Add visual representations of commands to the scene.
        
        Args:
            commands: A dictionary of commands from the engine variables.
        """
        # Handle linear velocity command
        if "linear_velocity_command" in commands:
            command_vel = commands["linear_velocity_command"]
            
            # Check if it's a numpy array-like with at least 2 elements
            if hasattr(command_vel, "shape") and len(command_vel) >= 2:
                # Draw X velocity arrow (forward/backward)
                self.add_velocity_arrow(
                    command_vel[0], 
                    base_pos=(0, 0, 1.7), 
                    rgba=(1.0, 0.0, 0.0, 0.8),  # Red for X
                    direction=[1.0, 0.0, 0.0],
                    label=f"X Cmd: {command_vel[0]:.2f}"
                )
                
                # Draw Y velocity arrow (left/right)
                self.add_velocity_arrow(
                    command_vel[1], 
                    base_pos=(0, 0, 1.5), 
                    rgba=(0.0, 1.0, 0.0, 0.8),  # Green for Y
                    direction=[0.0, 1.0, 0.0],
                    label=f"Y Cmd: {command_vel[1]:.2f}"
                )
    
    def add_velocity_arrow(self, command_velocity, base_pos=(0, 0, 1.7), scale=0.1, rgba=(0, 1.0, 0, 1.0), direction=None, label=None):
        """Add an arrow showing command velocity.
        
        Args:
            command_velocity: The velocity magnitude
            base_pos: Position for the arrow base
            scale: Scale factor for arrow length
            rgba: Color of the arrow
            direction: Optional direction vector [x,y,z]. If provided, arrow points in this direction
        """
        # Default to x-axis if no direction provided
        if direction is None:
            direction = [1.0, 0.0, 0.0]
            
        # For negative velocity, flip the direction
        if command_velocity < 0:
            direction = [-d for d in direction]
            
        # Get rotation matrix for the direction
        mat = rotation_matrix_from_direction(direction)
        
        # Scale the arrow length by the velocity magnitude
        length = abs(command_velocity) * scale
        
        # Add the arrow marker
        self.add_marker(
            pos=base_pos,
            mat=mat,
            size=[0.02, 0.02, max(0.001, length)],
            rgba=rgba,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            label=label if label is not None else f"Cmd: {command_velocity:.2f}"
        )
    
    def _update_scene_markers(self):
        """Add all current markers to the scene."""
        if self.handle._user_scn is None:
            return
            
        for marker in self._markers:
            if self.handle._user_scn.ngeom < self.handle._user_scn.maxgeom:
                g = self.handle._user_scn.geoms[self.handle._user_scn.ngeom]
                
                # Set basic properties
                g.type = marker['type']
                g.size[:] = marker['size']
                g.pos[:] = marker['pos']
                g.mat[:] = marker['mat']
                g.rgba[:] = marker['rgba']
                g.label = marker['label']
                
                # Set other rendering properties
                g.dataid = -1
                g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                g.objid = -1
                g.category = mujoco.mjtCatBit.mjCAT_DECOR
                g.emission = 0
                g.specular = 0.5
                g.shininess = 0.5
                
                self.handle._user_scn.ngeom += 1
    
    def sync(self):
        """Sync the viewer with current state."""
        self.handle.sync()

    def read_pixels(self) -> np.ndarray:
        """Read the current viewport pixels as a numpy array.
        
        Returns:
            A numpy array of shape (height, width, 3) containing RGB pixel values.
        """
        # Force a sync to ensure the current state is displayed
        self.handle.sync()
        
        # Get the current model and data from the handle
        model = self.handle.m
        data = self.handle.d
        
        if model is None or data is None:
            # If model or data is not available, return empty array with render dimensions
            return np.zeros((self._render_height, self._render_width, 3), dtype=np.uint8)
        
        # Initialize or update the renderer if needed
        if self._renderer is None:
            self._renderer = mujoco.Renderer(model, height=self._render_height, width=self._render_width)
        
        # Get the current camera settings from the viewer
        camera = mujoco.MjvCamera()
        camera.type = self.handle.cam.type
        camera.fixedcamid = self.handle.cam.fixedcamid
        camera.trackbodyid = self.handle.cam.trackbodyid
        camera.lookat[:] = self.handle.cam.lookat
        camera.distance = self.handle.cam.distance
        camera.azimuth = self.handle.cam.azimuth
        camera.elevation = self.handle.cam.elevation
        
        # Update the scene with the current physics state
        self._renderer.update_scene(data, camera=camera)
        
        # Add markers to the scene manually through the scene object
        scene = self._renderer.scene
        for marker in self._markers:
            if scene.ngeom < scene.maxgeom:
                g = scene.geoms[scene.ngeom]
                
                # Set marker properties
                g.type = marker['type']
                g.size[:] = marker['size']
                g.pos[:] = marker['pos']
                g.mat[:] = marker['mat'] 
                g.rgba[:] = marker['rgba']
                
                # Convert label if needed
                if isinstance(marker['label'], bytes):
                    label = marker['label']
                else:
                    label = str(marker['label']).encode('utf-8') if marker['label'] else b''
                g.label = label
                
                # Set additional properties
                g.dataid = -1
                g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                g.objid = -1
                g.category = mujoco.mjtCatBit.mjCAT_DECOR
                
                # Increment the geom count
                scene.ngeom += 1
        
        # Render the scene
        pixels = self._renderer.render()
        
        return pixels

    def update_and_sync(self):
        """Update the marks, sync with viewer, and clear the markers."""
        self._update_scene_markers()
        self.sync()
        self.clear_markers()

            
def launch_passive(model, data, show_left_ui=False, show_right_ui=False, capture_pixels=False, **kwargs):
    """Drop-in replacement for viewer.launch_passive."""
    # Launch the standard viewer without specifying dimensions
    handle = mujoco.viewer.launch_passive(model, data, show_left_ui=show_left_ui, show_right_ui=show_right_ui, **kwargs)
    # Wrap it with our enhanced version
    return MujocoViewerHandlerContext(handle, capture_pixels=capture_pixels)

class MujocoViewerHandlerContext:
    
    def __init__(self, handle: mujoco.viewer.Handle, capture_pixels=False):
        self.handle = handle
        self.capture_pixels = capture_pixels

    def __enter__(self):
        return MujocoViewerHandler(self.handle, capture_pixels=self.capture_pixels)

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.close()


def rotation_matrix_from_direction(direction: np.ndarray, reference: np.ndarray = np.array([0, 0, 1])) -> np.ndarray:
    """Compute a rotation matrix that aligns the reference vector with the direction vector.
    
    This is particularly useful for orienting objects (like arrows) in MuJoCo
    to point in a specific direction.
    
    Args:
        direction: The target direction vector [x, y, z].
        reference: The reference vector that will be aligned with direction.
                  Default is [0, 0, 1] (MuJoCo's default arrow direction).
                  
    Returns:
        A 3x3 rotation matrix that rotates the reference vector to point 
        in the direction of the direction vector.
    """
    # Normalize direction vector
    dir_vec = np.array(direction, dtype=float)
    norm = np.linalg.norm(dir_vec)
    if norm < 1e-10:  # Avoid division by zero
        return np.eye(3)
        
    dir_vec = dir_vec / norm
    
    # Normalize reference vector
    ref_vec = np.array(reference, dtype=float)
    ref_vec = ref_vec / np.linalg.norm(ref_vec)
    
    # Simple case: vectors are nearly aligned
    if np.abs(np.dot(dir_vec, ref_vec) - 1.0) < 1e-10:
        return np.eye(3)
        
    # Simple case: vectors are nearly opposite
    if np.abs(np.dot(dir_vec, ref_vec) + 1.0) < 1e-10:
        # Flip around x-axis for [0,0,1] reference
        if np.allclose(ref_vec, [0, 0, 1]):
            return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        # General case
        else:
            # Find an axis perpendicular to the reference
            perp = np.cross(ref_vec, [1, 0, 0])
            if np.linalg.norm(perp) < 1e-10:
                perp = np.cross(ref_vec, [0, 1, 0])
            perp = perp / np.linalg.norm(perp)
            
            # Rotate 180 degrees around this perpendicular axis
            c = -1  # cos(π)
            s = 0   # sin(π)
            t = 1 - c
            x, y, z = perp
            
            return np.array([
                [t*x*x + c, t*x*y - z*s, t*x*z + y*s],
                [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
                [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
            ])
    
    # General case: use cross product to find rotation axis
    axis = np.cross(ref_vec, dir_vec)
    axis = axis / np.linalg.norm(axis)
    
    # Angle between vectors
    angle = np.arccos(np.clip(np.dot(ref_vec, dir_vec), -1.0, 1.0))
    
    # Rodrigues rotation formula
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    
    return np.array([
        [t*x*x + c, t*x*y - z*s, t*x*z + y*s],
        [t*x*y + z*s, t*y*y + c, t*y*z - x*s],
        [t*x*z - y*s, t*y*z + x*s, t*z*z + c]
    ])
