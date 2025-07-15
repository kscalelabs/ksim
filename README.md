<div align="center">
<h1>K-Sim</h1>
<p>Welcome to <code>ksim</code>, a modular and easy-to-use framework for training policies in simulation.</p>
<h3>
  <a href="https://url.kscale.dev/docs">Docs</a> ·
  <a href="https://url.kscale.dev/discord">Discord</a>
</h3>
<img src="./assets/policy.gif" alt="Policy" />
</div>

## Installation

To install the framework:

```
pip install ksim
```

Make sure to install [JAX](https://github.com/google/jax#installation) correctly for your hardware (CPU or GPU). We recommend using `conda` rather than `uv` to avoid compatibility issues with MuJoCo on macOS.

## Getting Started

We suggest getting started by following the [examples](examples/).

## Acknowledgements

Many of the design decisions in `ksim` are heavily influenced other reinforcement learning training libraries like [Mujoco Playground](https://github.com/google-deepmind/mujoco_playground), [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) and [Maniskill](https://github.com/haosulab/ManiSkill). In particular, we are very grateful to the Mujoco and MJX maintainers for building a great, cross-platform simulator, and we hope `ksim` will help contribute to making it the ecosystem standard.
