# Examples

This directory contains example scripts for training policies in K-Sim.

While not strictly necessary, we recommend using `ksim` with a GPU - we do most of our development on 4090s, but other GPUs should work as well, although you might encounter minor bugs.

To get started, follow these instructions:

1. Clone this repository:

```bash
git clone git@github.com:kscalelabs/ksim.git
cd ksim
```

2. Create a new Python environment (requires Python 3.11 or later). We recommend using [uv](https://docs.astral.sh/uv/).

3. Install `ksim`:

```bash
pip install ksim  # To install the public version
pip install -e '.'  # To install your local copy
```

4. Run an example script:

```bash
python -m examples.walking
```

## Creating reference motion data

To create reference motion data, run the following command (from the root directory):

```
ksim-generate-reference --config examples/data/walk_normal.yaml
```

This will generate a `humanoid_amp_walk_ref.npz` file in the `examples/data` directory.

To visualize the reference motion, run the following command:

```
ksim-visualize-reference examples/data/humanoid_amp_walk_ref.npz --model examples/data/scene.mjcf
```
