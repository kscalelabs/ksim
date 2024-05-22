from brax import envs

from .default_humanoid.default_humanoid import DefaultHumanoidEnv

environments = {
    "default_humanoid": DefaultHumanoidEnv
}

def get_env(name: str, **kwargs) -> envs.Env:
    envs.register_environment(name, environments[name])
    return envs.get_environment(name, **kwargs)