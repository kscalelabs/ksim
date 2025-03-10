# ksim

This is K-Scale's library for running simulation experiments.

### Notes

When you try to render a trajectory while on a headless system, you may get an error like the following:

```bash
mujoco.FatalError: an OpenGL platform library has not been loaded into this process, this most likely means that a valid OpenGL context has not been created before mjr_makeContext was called
```

The fix is to create a virtual display:

```bash
Xvfb :100 -ac &
PID1=$!
export DISPLAY=:100.0
```

### Possible sources of NaNs

- The XLA Triton gemm kernel is buggy. To fix, try disabling with `export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"`


### Terminology
The following terminology is relevant to understanding RL tasks.
- `EnvState`: includes obs, command, action, reward, and done. The latter two are
  conditioned on the transition produced by the action.
- Dataset: a set of `EnvState`s used for training an RL update. Fully defined by
  `num_env_states_per_minibatch * num_minibatches`.
- Minibatch: a subset of the dataset used for training an RL update. Updates are
  performed per minibatch.
- Minibatch size: the number of environment states in each minibatch.
- Epoch: number of full passes through the current training dataset.
- Num Envs: the amount of parallel environments to run. Because of automatic
  resetting, this should not affect the batch math (in expectation).

### Variable Naming Conventions
Please use these units in the suffixes of variable names. For PyTrees, assume 
consistency of all dimensions except `L`. If including the timestampe would
help someone understand the variable, do the dimension suffix first, then the
timestamp suffix. (e.g. `mjx_data_L_0`). If it helps, specify return units in
function docstrings.

Dimension suffixes:
- `D`: dimension of environment states in the dataset.
- `B`: dimension of environment states in each minibatch.
- `T`: the time dimension during rollout.
- `E`: the env dimension during rollout.
- `L`: leaf dimension of a pytree (e.g. joint position vector size in an obs), 
       should not be used if the variable's final dimension is a scalar.

Timestamp suffixes:
- `t`: current timestep
- `t_plus_1`: next timestep
- `t_minus_1`: previous timestep
- `t_0`: initial timestep
- `t_f`: final timestep

These should absolutely be annotated:
- `mjx.Data`
- `mjx.Model`
- Everything relevant to `EnvState` (e.g. `obs`, `command`, `action`, etc.)

### Sharp Bits
- Add all sharp bits or unorthodox (yet correct) design decisions here.

### Development Tools

#### Reward Visualization

The reward visualizer is a tool for visualizing the rewards of an RL task. It is
designed to be used in conjunction with the `ksim.utils.reward_visualization.base.RewardVisualizer`
base class.

To use the reward visualizer, you need to subclass the base class and implement the
`setup_environment` and `run` methods.

The `setup_environment` method should return an instance of the environment
that you want to visualize the rewards of.

The `run` method should contain the logic for running the reward visualizer.

To see an example of how to use the reward visualizer, see the `ksim.utils.reward_visualization.mujoco.MujocoRewardVisualizer`
class. This class is used in the `examples/kbot/viz_standing.py` example.

Currently, the live plot for the reward visualizer is saved to a file which can be specified in the `RewardVisualizerConfig` class. It is by default saved to `/tmp/rewards_plots`. To see the live plot, open it in a viewer that supports live updates (e.g. opening in a VS Code tab)

Key Commands:
- `Space`: Pause/Resume the simulation
- `S`: Suspend the model in place
- Arrow Keys: Modify the model position in place
  - `up`: increase x position
  - `down`: decrease x position
  - `right`: increase y position
  - `left`: decrease y position
- `P`: increase z position
- `L`: decrease z position
- `N`: Step the simulation forward
- `R`: Reset joint positions and robot orientation to initial conditions
