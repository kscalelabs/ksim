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
