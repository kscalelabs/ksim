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
        
    def sync_engine_to_viewer(self, viewer_data: mujoco.MjData, engine_data: mujoco.MjData):
        engine_data.ctrl[:] = viewer_data.ctrl[:]
        engine_data.act[:] = viewer_data.act[:]
        engine_data.xfrc_applied[:] = viewer_data.xfrc_applied[:]
        engine_data.qpos[:] = viewer_data.qpos[:]
        engine_data.qvel[:] = viewer_data.qvel[:]
        engine_data.time = viewer_data.time
        
    def sync_viewer_to_engine(self, viewer_data: mujoco.MjData, engine_data: mujoco.MjData):
        viewer_data.ctrl[:] = engine_data.ctrl[:]
        viewer_data.act[:] = engine_data.act[:]
        viewer_data.xfrc_applied[:] = engine_data.xfrc_applied[:]
        viewer_data.qpos[:] = engine_data.qpos[:]
        viewer_data.qvel[:] = engine_data.qvel[:]
        viewer_data.time = engine_data.time
            
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
