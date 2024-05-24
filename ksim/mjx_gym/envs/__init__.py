from typing import Any

from brax import envs

from ksim.mjx_gym.envs.default_humanoid_env.default_humanoid import DefaultHumanoidEnv
from ksim.mjx_gym.envs.stompy_env.stompy import StompyEnv

environments = {"default_humanoid": DefaultHumanoidEnv, "stompy": StompyEnv}


def get_env(name: str, **kwargs: Any) -> envs.Env:  # noqa: ANN401
    envs.register_environment(name, environments[name])
    return envs.get_environment(name, **kwargs)
