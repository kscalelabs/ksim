"""Defines debugging utility functions."""

__all__ = [
    "JitLevel",
]


class JitLevel:
    OUTER_LOOP = 1
    RL_CORE = 5
    ENGINE_STEP = 10
    AUX_FUNCTIONS = 11
    MJX_STEP = 12
