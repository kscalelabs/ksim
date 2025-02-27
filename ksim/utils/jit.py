"""JIT utilities."""

from typing import Callable

import equinox as eqx


def toggleable_jit(func=None, *, disabled=True, **kwargs) -> Callable:
    """Decorator that conditionally applies eqx.filter_jit based on a flag."""

    def decorator(fn):
        if disabled:
            return fn
        return eqx.filter_jit(fn, **kwargs)

    if func is None:
        return decorator
    return decorator(func)
