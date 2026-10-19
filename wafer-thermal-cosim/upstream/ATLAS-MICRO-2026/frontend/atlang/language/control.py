"""Loop and host-range helpers for atlang source kernels."""

from __future__ import annotations

import builtins
from typing import Any

from .objects import SerialRange
from .utils import reject_kwargs


# Loop helpers

def Serial(start: Any, stop: Any | None = None, step: Any | None = None, **kwargs: Any) -> SerialRange:
    reject_kwargs("Serial", kwargs)
    if stop is None:
        return SerialRange(0, start, 1 if step is None else step)
    return SerialRange(start, stop, 1 if step is None else step)


range = builtins.range


__all__ = ["Serial", "range"]
