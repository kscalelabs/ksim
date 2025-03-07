"""JIT utilities."""

import inspect
import os
import time
from functools import wraps
from typing import Any, Callable, ParamSpec, TypeVar, cast

import jax

DEFAULT_COMPILE_TIMEOUT = 1.0


def get_hash(obj: object) -> int:
    """Get a hash of an object.

    If the object is hashable, use the hash. Otherwise, use the id.
    """
    try:
        return hash(obj)
    except (TypeError, RecursionError):
        return id(obj)


P = ParamSpec("P")  # For function parameters
R = TypeVar("R")  # For function return type


def legit_jit(
    static_argnames: list[str] | None = None,
    compile_timeout: float = DEFAULT_COMPILE_TIMEOUT,
    **jit_kwargs: Any,  # noqa: ANN401
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Jit that works on class methods, is toggleable, and detects recompilations.

    To avoid confusion...
    - `legit_jit` is a decorator factory and returns `decorator`.
    - `decorator` is the actual decorator, and the decorated function calls `wrapped`.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        class JitState:
            compilation_count = 0
            last_arg_dict: dict[str, int] | None = None

        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())

        jitted_fn = jax.jit(
            fn,
            static_argnames=tuple(list(static_argnames) if static_argnames is not None else []),
            **jit_kwargs,
        )

        @wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            if os.environ.get("DEBUG", "0") == "1":  # skipping during debug
                return fn(*args, **kwargs)

            start_time = time.time()
            res = jitted_fn(*args, **kwargs)
            end_time = time.time()
            runtime = end_time - start_time

            # if this is true, if runtime is higher than COMPILE_TIMEOUT, we recompile
            # TODO: we should probably reimplement the lower-level jitting logic to avoid this
            if os.environ.get("JIT_PROFILE", "0") == "1":
                arg_dict = {}
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        arg_dict[param_names[i]] = get_hash(arg)
                for k, v in kwargs.items():
                    arg_dict[k] = get_hash(v)

                if fn.__name__ == "__call__":
                    class_name = (args[0].__class__.__name__) + "."
                else:
                    class_name = ""
                print(f"Running {class_name}{fn.__name__} (count: {JitState.compilation_count})")
                print(f"- took {runtime} seconds")
                JitState.compilation_count += 1

                if JitState.last_arg_dict is not None:
                    all_keys = set(arg_dict.keys()) | set(JitState.last_arg_dict.keys())
                    for k in all_keys:
                        prev = JitState.last_arg_dict.get(k, "N/A")
                        curr = arg_dict.get(k, "N/A")

                        if prev != curr:
                            print(f"- Arg '{k}' hash changed: {prev} -> {curr}")

                JitState.last_arg_dict = arg_dict

            return cast(R, res)

        return wrapped

    return decorator
