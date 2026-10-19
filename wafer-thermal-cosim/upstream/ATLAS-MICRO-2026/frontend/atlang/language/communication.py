"""Inter-core communication calls in the atlang language surface."""

from __future__ import annotations

from typing import Any

from .objects import ProcessStyleCall
from .utils import reject_kwargs


# Communication operations

def send(src_core: Any, dst_core: Any, buffer_like: Any, **kwargs: Any) -> ProcessStyleCall:
    reject_kwargs("send", kwargs)
    return ProcessStyleCall(op="send", args=(src_core, dst_core, buffer_like), out=None)


def recv(src_core: Any, dst_core: Any, buffer_like: Any, **kwargs: Any) -> ProcessStyleCall:
    reject_kwargs("recv", kwargs)
    return ProcessStyleCall(op="recv", args=(src_core, dst_core), out=buffer_like)


__all__ = ["recv", "send"]
