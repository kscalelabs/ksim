"""Profiling utilities."""

import os
import time
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")  # For function parameters
R = TypeVar("R")  # For function return type


def profile(fn: Callable[P, R]) -> Callable[P, R]:
    """Profiling decorator that tracks function call count and execution time.

    Activated when the PROFILE environment variable is set to "1".

    Returns:
        A decorated function with profiling capabilities.
    """

    class ProfileState:
        call_count = 0
        total_time = 0.0

    @wraps(fn)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        if os.environ.get("PROFILE", "0") != "1":
            return fn(*args, **kwargs)

        start_time = time.time()
        res = fn(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time

        ProfileState.call_count += 1
        ProfileState.total_time += runtime

        # Handle class methods by showing class name
        if fn.__name__ == "__call__" or (args and hasattr(args[0], "__class__")):
            try:
                class_name = args[0].__class__.__name__ + "."
            except (IndexError, AttributeError):
                class_name = ""
        else:
            class_name = ""

        print(
            f"{class_name}{fn.__name__} - call #{ProfileState.call_count}, took {runtime:.6f}s, total: {ProfileState.total_time:.6f}s"
        )

        return res

    return wrapped
