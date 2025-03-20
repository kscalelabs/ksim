""" Utilities for rendering the environment. """
import mujoco
import mujoco.viewer

class MujocoViewerHandler:
    def __init__(self, handle: mujoco.viewer.Handle):
        self.handle = handle
        # self._setup_features()

    def setup_camera(self, config: dict):
        self.handle.cam.distance = config.render_distance
        self.handle.cam.azimuth = config.render_azimuth
        self.handle.cam.elevation = config.render_elevation
        self.handle.cam.lookat[:] = config.render_lookat
        if config.render_track_body_id is not None:
            self.handle.cam.trackbodyid = config.render_track_body_id
            self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    def sync(self):
        self.handle.sync()
        
    def copy_data(self, dst: mujoco.MjData, src: mujoco.MjData):
        dst.ctrl[:] = src.ctrl[:]
        dst.act[:] = src.act[:]
        dst.xfrc_applied[:] = src.xfrc_applied[:]
        dst.qpos[:] = src.qpos[:]
        dst.qvel[:] = src.qvel[:]
        dst.time = src.time
            
def launch_passive(model, data, show_left_ui=False, show_right_ui=False, **kwargs):
    """Drop-in replacement for viewer.launch_passive."""
    # Launch the standard viewer
    handle = mujoco.viewer.launch_passive(model, data, show_left_ui=show_left_ui, show_right_ui=show_right_ui, **kwargs)
    # Wrap it with our enhanced version
    return MujocoViewerHandlerContext(handle)

class MujocoViewerHandlerContext:
    
    def __init__(self, handle: mujoco.viewer.Handle):
        self.handle = handle

    def __enter__(self):
        return MujocoViewerHandler(self.handle)

    def __exit__(self, exc_type, exc_value, traceback):
        self.handle.close()
