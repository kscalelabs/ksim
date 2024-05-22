import yaml
import wandb
import argparse
from envs import get_env
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
import mediapy as media
import jax

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run PPO training with specified config file.')
parser.add_argument('--config', type=str, required=True, help='Path to the config YAML file')
args = parser.parse_args()

# Load config from YAML file
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

# Initialize wandb
wandb.init(project=config.get('project_name', 'robotic_locomotion_training') + "_test", name=config.get('experiment_name', 'ppo-training') + "_test")

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

print(f'env_name: {config.get("env_name", "stompy")}')
print(f'reward_params: {reward_params}')
print(f'testing on {config["num_envs"]} environments')

env = get_env(
    name=config.get('env_name', 'stompy'),
    reward_params=reward_params,
    terminate_when_unhealthy=terminate_when_unhealthy,
    reset_noise_scale=reset_noise_scale,
    exclude_current_positions_from_observation=exclude_current_positions_from_observation,
    log_reward_breakdown=log_reward_breakdown
)

# Loading params
model_path = "weights/" + config.get('project_name', '') + ".pkl"
params = model.load_params(model_path)
policy_network = ppo_networks.make_ppo_networks(env.observation_size, env.action_size)
inference_fn = ppo_networks.make_inference_fn(policy_network)(params)
jit_inference_fn = jax.jit(inference_fn)
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# initialize the state
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state.pipeline_state]

# grab a trajectory
n_steps = 100
render_every = 2

for i in range(n_steps):
  act_rng, rng = jax.random.split(rng)
  ctrl, _ = jit_inference_fn(state.obs, act_rng)
  state = jit_step(state, ctrl)
  rollout.append(state.pipeline_state)

  if state.done:
    break

media.show_video(env.render(rollout[::render_every], camera='side'), fps=1.0 / env.dt / render_every)