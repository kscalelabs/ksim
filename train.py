import yaml
import wandb
import argparse
from envs import get_env
from brax.training.agents.ppo import train as ppo
import functools
import matplotlib.pyplot as plt
from datetime import datetime
from brax import envs

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run PPO training with specified config file.')
parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
args = parser.parse_args()

# Load config from YAML file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# Initialize wandb
wandb.init(project="robotic-locomotion-training", name=config.get('experiment_name', 'ppo-training'))

DEFAULT_REWARD_PARAMS = {
    'rew_forward': {'weight': 1.25},
    'rew_healthy': {'weight': 5.0, 'healthy_z_lower': 1.0, 'healthy_z_upper': 2.0},
    'rew_ctrl_cost': {'weight': 0.1}
}

reward_params = config.get('reward_params', DEFAULT_REWARD_PARAMS)
terminate_when_unhealthy = config.get('terminate_when_unhealthy', True)
reset_noise_scale = config.get('reset_noise_scale', 1e-2)
exclude_current_positions_from_observation = config.get('exclude_current_positions_from_observation', True)
log_reward_breakdown = config.get('log_reward_breakdown', True)

env = get_env(
    "default_humanoid", 
    reward_params=reward_params,
    terminate_when_unhealthy=terminate_when_unhealthy,
    reset_noise_scale=reset_noise_scale,
    exclude_current_positions_from_observation=exclude_current_positions_from_observation,
    log_reward_breakdown=log_reward_breakdown
)

train_fn = functools.partial(
    ppo.train,
    num_timesteps=config['num_timesteps'],
    num_evals=config['num_evals'],
    reward_scaling=config['reward_scaling'],
    episode_length=config['episode_length'],
    normalize_observations=config['normalize_observations'],
    action_repeat=config['action_repeat'],
    unroll_length=config['unroll_length'],
    num_minibatches=config['num_minibatches'],
    num_updates_per_batch=config['num_updates_per_batch'],
    discounting=config['discounting'],
    learning_rate=config['learning_rate'],
    entropy_cost=config['entropy_cost'],
    num_envs=config['num_envs'],
    batch_size=config['batch_size'],
    seed=config['seed']
)

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

max_y, min_y = 13000, 0

def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    ydataerr.append(metrics['eval/episode_reward_std'])

    # Log metrics to wandb
    print(metrics)
    wandb.log({"steps": num_steps, **metrics})

    plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
    plt.ylim([min_y, max_y])

    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')

    plt.errorbar(
        x_data, y_data, yerr=ydataerr)
    plt.show()

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')
