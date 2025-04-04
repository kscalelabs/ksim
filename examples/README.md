# Default Humanoid

This is a simple example of a humanoid robot walking task.

To create this example, we took the `humanoid.xml` model from: https://github.com/google-deepmind/mujoco/blob/main/mjx/mujoco/mjx/test_data/humanoid/humanoid.xml and did the following:

- Added `_ctrl` suffix to all the motor names
- Removed the floor geom

# AMP / Gait Matching

## Setup

First, make sure to install bvhio

```
pip install bvhio
```

To create the actorcore BVH file, we ran a simple converter from FBX to BVH and took the "looped" version.

To map the actorcore motion to default_humanoid update offsets (after converting from .fbx to .bvh):

- Base_Spine01: add 3 on the Z offset
- Base_L/R_Forearm: add 5.5 to the X offset
- Base_L/R_Hand: add 5.5 in the -X offset
- Base_L/R_Calf: subtract 7 in the -Z offset
- Base_L/R_Foot: subtract 7.5 in the -Z offset
