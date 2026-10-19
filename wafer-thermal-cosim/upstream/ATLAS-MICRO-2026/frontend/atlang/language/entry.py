"""Decorator entry point for atlang kernels."""

from __future__ import annotations

from typing import Any, Callable

from .utils import reject_kwargs


# Kernel decorator

def main(func: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
    reject_kwargs("main", kwargs)
    if func is None:
        return lambda inner: main(inner)
    import inspect

    from ..capture.parser import parse_main_function

    current_frame = inspect.currentframe()
    caller_frame = None if current_frame is None else current_frame.f_back
    caller_locals = {} if caller_frame is None else dict(caller_frame.f_locals)
    return parse_main_function(func, caller_locals=caller_locals)


__all__ = ["main"]
