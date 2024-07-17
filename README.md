<p align="center">
  <picture>
    <img alt="K-Scale Open Source Robotics" src="https://media.kscale.dev/kscale-open-source-header.png" style="max-width: 100%;">
  </picture>
</p>  

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kscalelabs/ksim/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1224056091017478166)](https://discord.gg/k5mSvCkYQh)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)
<br />
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![ruff](https://img.shields.io/badge/Linter-Ruff-red.svg?labelColor=gray)](https://github.com/charliermarsh/ruff)
<br />
[![Python Checks](https://github.com/kscalelabs/ksim/actions/workflows/test.yml/badge.svg)](https://github.com/kscalelabs/ksim/actions/workflows/test.yml)
[![Update Stompy S3 Model](https://github.com/kscalelabs/ksim/actions/workflows/update_s3_model.yml/badge.svg)](https://github.com/kscalelabs/ksim/actions/workflows/update_s3_model.yml)
[![AutoEval](https://github.com/kscalelabs/autoeval/actions/workflows/eval.ksim.yml/badge.svg)](https://github.com/kscalelabs/autoeval/actions/workflows/eval.ksim.yml)

</div>

# K-Scale Sim

A simple and efficient library for training humanoid locomotion in MJX and MuJoCo.

## Installation
1. Clone this repository:
```bash
git clone https://github.com/kscalelabs/ksim.git
cd ksim
```

2. It is recommended that you use a virtual environment to install the dependencies for this project. You can create a new conda environment using the following command:
```bash
conda create --name ksim python=3.11
conda activate ksim
```
To install the dependencies, run the following command:

```bash
make install-dev
```

Finally, add your Weights & Biases API key via conda:

```bash
conda env config vars set WANDB_API_KEY=<your api key>
```

## MJX Gym Usage
MJX Gym is a library for training and evaluating reinforcement learning agents in MJX environments. It is built on top of the Brax library and provides a simple interface for running experiments with Stompy and other humanoid formfactors. Currently, we support walking but plan on adding more tasks and simulator environments in the future.



### Training
For quick experimentation, you may specify all relevant training configurations via YAML files, and simply run train.py with the desired configuration. Our configurations allow for rapid reward function prototyping, environment specification, and hyperparameter tuning. Additionally, all training results are tracked and logged using Weights & Biases.

We recommend starting with the default humanoid environment to get a feel for the simulator. To train the default humanoid environment, first navigate to the /mjx_gym directory and define the paths of your model:
```bash
cd ksim/mjx_gym
export MODEL_DIR=path/to/your/robot/model
```

Then, run the following command:
```bash
python train.py --config experiments/default_humanoid_walk.yaml
```

Example training curves are shown below:
<p align="center">
  <img alt="K-Scale Open Source Robotics" src="https://private-user-images.githubusercontent.com/43460304/333030841-5a4bd24e-0478-4cbb-bb92-18cf5e9c72b7.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTY1MDc4ODcsIm5iZiI6MTcxNjUwNzU4NywicGF0aCI6Ii80MzQ2MDMwNC8zMzMwMzA4NDEtNWE0YmQyNGUtMDQ3OC00Y2JiLWJiOTItMThjZjVlOWM3MmI3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTIzVDIzMzk0N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWFiZTQyNDFlNjViN2M0MDRiMjgyZWRkYWI3ZmQxYWU3NDM4MDViZDVlYmJlY2NiNTIwZDU2NGVmNWE2YjU1NjcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.Yi2HzUGt3HYivMyGTuRFSRjsZviZRK2dFEx_KtZyEyA" width="400" />
  <img alt="K-Scale Open Source Robotics" src="https://private-user-images.githubusercontent.com/43460304/333030844-341f4a5c-95ea-49ed-b102-df4e00868c5a.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTY1MDc4ODcsIm5iZiI6MTcxNjUwNzU4NywicGF0aCI6Ii80MzQ2MDMwNC8zMzMwMzA4NDQtMzQxZjRhNWMtOTVlYS00OWVkLWIxMDItZGY0ZTAwODY4YzVhLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTIzVDIzMzk0N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTcxYzc5MzlkMGRiZGJkMmE2ZjYwY2FhMmZjZGQ3NTM4YWNlOTVkMTU4MjBjZmM2ZmRmZmE1OTkxYTlmYTJkZDImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.01D5U4QD8JooJwEDtcw9OKt1tmM_zGYj5U7dDstLC84" width="400" >
</p>


### Testing and Rendering
We provide an easy way to test and render the trained model using the play.py script. This script loads the trained model and runs it in the specified environment.

To test and render the trained model, run the following command:
```bash
python play.py --config experiments/default_humanoid_walk.yaml
```

Here is an example of the trained model walking in the default humanoid environment after roughly 15 minutes of training on the K-Scale cluster.

<video controls>
  <source alt="K-Scale Open Source Robotics" src="assets/333027934-8e12b0e6-48ea-4af0-8283-1dc4880767b4.mp4" type="video/mp4">
</video>

You may transfer the model (trained via MJX) to MuJoCo for CPU-based simulation via arguments to the play script. For example, to run the model in the default humanoid environment, use the following command:
```bash
python play.py --config experiments/default_humanoid_walk.yaml --use_mujoco
```

It is important that the configuration file used for testing is the **same as the one used for training.** This ensures that the model is loaded correctly and avoids any catastrophic failures during testing. For simplicity, we have included default training weights in the /mjx_gym/weights directory, which are automatically used when running the play script with the command above. However, you may specify a different weight file using the --params_path argument.

Here is an example of the trained model (in MJX) walking in the default humanoid environment in MuJoCo.

<video controls>
  <source alt="K-Scale Open Source Robotics" src="assets/333027930-7f158aeb-6bc9-4056-bd1d-12882adbd13c.mp4" type="video/mp4">
</video>

While the model performs well in the MJX environment, it is important to note that the model's performance may vary when transferred to MuJoCo. This is due to slight differences in the simulators and the underlying physics engines. Evaluating the model in both simulators is helpful for determining whether the model will generalize well across different environments.

Getting Stompy to walk in MJX is a challenging task, and we encourage the open source community to contribute toward training efforts. The most recent set of weights allows stompy (without upper body) to maintain balance in a stable environment. An example trajectory is included below:

<video controls>
  <source alt="Stompy Standing" src="assets/stompy_standing.mp4" type="video/mp4">
</video>

# Headless livestreaming

For setting up headless streaming, export the following setup variables and install `xvfb`:
```bash
export DISPLAY=:0
export MUJOCO_GL=egl
install xvfb
Xvfb :0 -screen 0 1024x768x24 &
```

For simplication, refer to `run.sh` in mjx_gym.

### Running from SLURM Cluster
To run from a SLURM cluster (e.g. the K-Scale Andromeda cluster), cd to /mjx_gym/ and run:

```bash
sbatch scripts/train.slurm
```

Training logs, errors, and gpu-utilization will be tracked in the /logs/ directory.

### Common Issues
You might see the following error when running train.py:
>An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.

One way to fix this issue is to uninstall jax and jaxlib and reinstall them with your specific CUDA version the following commands:
```bash
pip uninstall jax jaxlib
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Note that the CUDA version should match the one installed on your machine.

A similar issue might also occur:
> jaxlib.xla_extension.XlaRuntimeError: FAILED_PRECONDITION: DNN library initialization failed. Look at the errors above for more details.

This error is usually caused by an incorrect version of cudnn installed in your environment. To fix this issue, you can try installing cuDNN using the following command:
```bash
conda install cudnn
```

Another common issue occurs when rendering on a headless server. For now, we recommend rendering locally or using a remote desktop connection to view the rendering.

## MJCF Reference
To visualize any MJCF file, you can run the following command:
```bash
python3 -m mujoco.viewer --mjcf <path-to-mjcf-file>
```

The command above loads MuJoCo's GUI, which allows you to simulate the model, manually specify joints, and save keyframes.

## TODO
- Bootstrap 
- [ ] Get Stompy walking with PPO through bootstrapping
- [ ] Implement imitation learning techniques for end-to-end walking and recovering tasks
- [ ] Add CPU-based MuJoCo training platform for improved sim-to-sim support
- [ ] Add sim-to-real transfer learning techniques during training