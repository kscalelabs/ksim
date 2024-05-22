from brax import envs

from .default_humanoid.default_humanoid import DefaultHumanoidEnv
from .stompy.stompy import StompyEnv

environments = {
    "default_humanoid": DefaultHumanoidEnv,
    "stompy": StompyEnv
}

def get_env(name: str, **kwargs) -> envs.Env:
    envs.register_environment(name, environments[name])
    return envs.get_environment(name, **kwargs)