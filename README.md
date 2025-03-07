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
       shoould not be used if the variable's final dimension is a scalar.

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
