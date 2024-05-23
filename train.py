import argparse
import functools
from datetime import datetime

import matplotlib.pyplot as plt
import wandb
import yaml
from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo
from envs import get_env

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run PPO training with specified config file.")
parser.add_argument("--config", type=str, required=True, help="Path to the config YAML file")
args = parser.parse_args()

# Load config from YAML file
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

# Initialize wandb
wandb.init(
    project=config.get("project_name", "robotic-locomotion-training"),
    name=config.get("experiment_name", "ppo-training"),
)

DEFAULT_REWARD_PARAMS = {
    "rew_forward": {"weight": 1.25},
    "rew_healthy": {"weight": 5.0, "healthy_z_lower": 1.0, "healthy_z_upper": 2.0},
    "rew_ctrl_cost": {"weight": 0.1},
}

reward_params = config.get("reward_params", DEFAULT_REWARD_PARAMS)
terminate_when_unhealthy = config.get("terminate_when_unhealthy", True)
reset_noise_scale = config.get("reset_noise_scale", 1e-2)
exclude_current_positions_from_observation = config.get("exclude_current_positions_from_observation", True)
log_reward_breakdown = config.get("log_reward_breakdown", True)

print(f"reward_params: {reward_params}")
print(f'training on {config["num_envs"]} environments')

env = get_env(
    name=config.get("env_name", "stompy"),
    reward_params=reward_params,
    terminate_when_unhealthy=terminate_when_unhealthy,
    reset_noise_scale=reset_noise_scale,
    exclude_current_positions_from_observation=exclude_current_positions_from_observation,
    log_reward_breakdown=log_reward_breakdown,
)
print(f'Env loaded: {config.get("env_name", "could not find environment")}')

train_fn = functools.partial(
    ppo.train,
    num_timesteps=config["num_timesteps"],
    num_evals=config["num_evals"],
    reward_scaling=config["reward_scaling"],
    episode_length=config["episode_length"],
    normalize_observations=config["normalize_observations"],
    action_repeat=config["action_repeat"],
    unroll_length=config["unroll_length"],
    num_minibatches=config["num_minibatches"],
    num_updates_per_batch=config["num_updates_per_batch"],
    discounting=config["discounting"],
    learning_rate=config["learning_rate"],
    entropy_cost=config["entropy_cost"],
    num_envs=config["num_envs"],
    batch_size=config["batch_size"],
    seed=config["seed"],
)

times = [datetime.now()]


def progress(num_steps, metrics):
    times.append(datetime.now())

    wandb.log({"steps": num_steps, "epoch_time": (times[-1] - times[-2]).total_seconds(), **metrics})


def save_model(current_step, make_policy, params):
    model_path = "weights/" + config.get("project_name", "model") + ".pkl"
    model.save_params(model_path, params)
    print(f"Saved model at step {current_step} to {model_path}")


make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress, policy_params_fn=save_model)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
