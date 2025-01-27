import argparse
import os
import pickle
import shutil

import genesis as gs
from kbot_env import KbotEnv
from rsl_rl.runners import OnPolicyRunner


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 10,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "L_hip_y": 0.0,
            "L_hip_x": 0.0,
            "L_hip_z": 0.0,
            "L_knee": 0.0,
            "L_ankle_y": 0.0,
            "R_hip_y": 0.0,
            "R_hip_x": 0.0,
            "R_hip_z": 0.0,
            "R_knee": 0.0,
            "R_ankle_y": 0.0,
        },
        "dof_names": [
            "L_hip_y",
            "L_hip_x",
            "L_hip_z",
            "L_knee",
            "L_ankle_y",
            "R_hip_y",
            "R_hip_x",
            "R_hip_z",
            "R_knee",
            "R_ankle_y",
        ],
        # PD
        "kp": 120.0,
        "kd": 5.0,
        # termination
        "termination_if_roll_greater_than": 120,  # degree
        "termination_if_pitch_greater_than": 120,
        # base pose
        "base_init_pos": [0.0, 0.0, 1.02],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 39,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="gpr-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--show_viewer", type=bool, default=False)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = KbotEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=args.device,
        show_viewer=args.show_viewer,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
