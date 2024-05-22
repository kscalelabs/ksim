
from .default_humanoid.default_humanoid import DefaultHumanoidEnv

envs = {
    "default_humanoid": DefaultHumanoidEnv
}

def get_env(name: str):
    return envs[name]