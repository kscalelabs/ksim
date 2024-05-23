import wandb
import yaml
import argparse
from envs import get_env
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
import mediapy as media
import numpy as np
from utils.default import DEFAULT_REWARD_PARAMS
from utils.rollouts import render_mjx_rollout, render_mujoco_rollout

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run PPO training with specified config file.')
parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
parser.add_argument('--use_mujoco', type=bool, default=False, help='Use mujoco instead of mjx for rendering')
parser.add_argument('--params_path', type=str, default=None, help='Path to the params file')
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# Initialize wandb
wandb.init(project=config.get('project_name', 'robotic_locomotion_training') + "_test", name=config.get('experiment_name', 'ppo-training') + "_test")

# Load environment
env = get_env(
    name=config.get('env_name', 'stompy'),
    reward_params=config.get('reward_params', DEFAULT_REWARD_PARAMS),
    terminate_when_unhealthy=config.get('terminate_when_unhealthy', True),
    reset_noise_scale=config.get('reset_noise_scale', 1e-2),
    exclude_current_positions_from_observation=config.get('exclude_current_positions_from_observation', True),
    log_reward_breakdown=config.get('log_reward_breakdown', True)
)
print(f'Loaded environment {config.get("env_name", "")} with env.observation_size: {env.observation_size} and env.action_size: {env.action_size}')

# Loading params
if args.params_path is not None:
    model_path = args.params_path
else:
    model_path = "weights/" + config.get('project_name', 'model') + ".pkl"
params = model.load_params(model_path)
normalize = lambda x, y: x
if config.get('normalize_observations', False):
    normalize = running_statistics.normalize
policy_network = ppo_networks.make_ppo_networks(env.observation_size, env.action_size, preprocess_observations_fn=normalize)
inference_fn = ppo_networks.make_inference_fn(policy_network)(params)
print(f'Loaded params from {model_path}')

# rolling out a trajectory
render_every = 2
n_steps = 1000
if args.use_mujoco:
    images = render_mujoco_rollout(env, inference_fn, n_steps, render_every)
else:
    images = render_mjx_rollout(env, inference_fn, n_steps, render_every)
print(f'Rolled out {len(images)} steps')

# render the trajectory
images_thwc = np.array(images)
images_tchw = np.transpose(images_thwc, (0, 3, 1, 2))

fps = 1/env.dt
wandb.log({'training_rollouts': wandb.Video(images_tchw, fps=fps, format="mp4")})

media.write_video('video.mp4', images_thwc, fps=fps)