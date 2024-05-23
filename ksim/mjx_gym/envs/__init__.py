from brax import envs
from envs.default_humanoid_env.default_humanoid import DefaultHumanoidEnv
from envs.stompy_env.stompy import StompyEnv

environments = {"default_humanoid": DefaultHumanoidEnv, "stompy": StompyEnv}


def get_env(name: str, **kwargs) -> envs.Env:
    envs.register_environment(name, environments[name])
    return envs.get_environment(name, **kwargs)
