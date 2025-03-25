# Default Humanoid

This is a simple example of a humanoid robot walking task.

To create this example, we took the `humanoid.xml` model from: https://github.com/google-deepmind/mujoco/blob/main/mjx/mujoco/mjx/test_data/humanoid/humanoid.xml and did the following:
- Added `_ctrl` suffix to all the motor names
- Removed the floor geom