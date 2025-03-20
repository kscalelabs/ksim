# ksim

This is K-Scale's library for running simulation experiments.

## Installation

Clone the repo

```bash
git clone git@github.com:kscalelabs/ksim.git
```

Also, you will need git lfs for the kscale-assets submodule, so run

```bash
sudo apt-get install git-lfs # Optional
cd ksim
git submodule update --init --recursive
```

Create a new python environment. We reccomend using [conda](https://www.anaconda.com/docs/getting-started/miniconda/main) for this.

```bash
conda create -n ksim python=3.11
conda activate ksim
pip install -U "jax[cuda12]"
```

Optional: Verify GPU backend: `python -c "import jax; print(jax.default_backend())"` should print gpu
Next, install the ksim python package locally

```bash
cd ksim # make sure you are in the root folder of this repo (ls should show a pyproect.toml file)
pip install -e .\[all\]
```

You should be all set! See the troubleshooting section below for tips if something isn't working.

## Usage

### Training

Run training on any of the example environments by their task name, for example:

```bash
python -m examples.kbot.standing
```

### Evaluation

To evaluate (save a video of) a saved checkpoint use `action=env` and `pretrained=path/to/run` and optionally `checkpoint_num=n`, for example:

```bash
python -m examples.kbot.standing action=env pretrained=examples/kbot/kbot_standing_task/run_6 checkpoint_num=5
```

### Interactive Visualization

To run the interactive visualizer, use the following command:

```bash
python -m examples.kbot.viz_standing
# or
python -m examples.kbot.viz_standing --physics-backend mujoco # default is mjx
```

Use `mjpython` if using a Mac, see
[mujoco passive-viewer docs](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)

The interactive visualizer is a tool for visualizing the state of an RL task. It is
designed to be used in conjunction with the `ksim.utils.interactive.base.InteractiveVisualizer`
base class.

To use the interactive visualizer, you need to subclass the base class and implement the
`setup_environment` and `run` methods.

The `setup_environment` method should return an instance of the environment
that you want to visualize the state of.

The `run` method should contain the logic for running the interactive visualizer.

To see an example of how to use the interactive visualizer, see the `ksim.utils.interactive.mujoco.MujocoInteractiveVisualizer`
class. This class is used in the `examples/kbot/viz_standing.py` example.

Currently, the live plot for the reward is saved to a file which can be specified in the `InteractiveVisualizerConfig` class. It is by default saved to `/tmp/rewards_plots`. To see the live plot, open it in a viewer that supports live updates (e.g. opening in a VS Code tab)

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

## Terminology

The following terminology is relevant to understanding RL tasks.

- `Trajector`: includes obs, command, action, and done. The latter is
  conditioned on the trajectory produced by the action.
- Dataset: a set of `Trajectory`s used for training an RL update. Fully defined by
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
- Everything relevant to `Trajectory` (e.g. `obs`, `command`, `action`, etc.)

### Sharp Bits

- Add all sharp bits or unorthodox (yet correct) design decisions here.

## Troubleshooting

### Headless Systems

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

You may also need to tell MuJoCo to use GPU accelerated off-screen rendering via

```
export MUJOCO_GL="egl"
```

### Possible sources of NaNs

- The XLA Triton gemm kernel is buggy. To fix, try disabling with `export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"`

### Long run / wait times

Prefix your commands with `JIT_PROFILE=1` to enable prints for what is taking long to compile and run.

### Clear cache

We've often found that jax reuses caches when its not supposed to. We recommend clearing your Jax cache after changing any function

```bash
rm -rf ~/.cache/jax/jaxcache
```
