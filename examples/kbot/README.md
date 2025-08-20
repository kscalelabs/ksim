# K-Bot Training Example

This example shows how to use K-Sim to train and deploy a policy to run on the K-Bot.

To get started, first install the additional requirements:

```bash
pip install requirements.txt
```

Next, train a policy:

```bash
python -m train
```

Once trained, you can export the script using:

```bash
python -m convert /path/to/ckpt.bin /path/to/output/dir/
```

## Notes

If you're getting segfaults on Mujoco on a Linux-based headless machine, you should use:

```bash
export MUJOCO_GL=egl
export EGL_PLATFORM=surfaceless
```

This will require:

```bash
sudo apt-get install -y libosmesa6 libosmesa6-dev
```
