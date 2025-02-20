# ksim

This is K-Scale's library for running simulation experiments.

## Getting Started

First, create a new Conda environment (note that there is a [known issue](https://github.com/Genesis-Embodied-AI/Genesis/issues/11) with running Genesis in uv, so it's recommended to use Conda instead):

```bash
conda create -n ksim python=3.11
```

Then, install the dependencies:

```bash
pip install -e .
```

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
