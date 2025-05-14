"""Defines debugging utility functions."""

__all__ = [
    "JitLevel",
]


class JitLevel:
    OUTER_LOOP = 10
    RL_CORE = 20
    UNROLL = 30
    INITIALIZATION = 40
    ENGINE = 50
    HELPER_FUNCTIONS = 60
    MJX = 70
