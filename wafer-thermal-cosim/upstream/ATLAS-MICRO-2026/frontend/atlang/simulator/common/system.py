"""System-type helpers shared by atlang extraction modules."""

from __future__ import annotations

from typing import Any


def is_edge_system_config(system_config: Any) -> bool:
    return type(system_config).__name__ == "EdgeSystemConfig"


def get_primary_context(snapshot: Any) -> Any:
    if not snapshot.core_array_contexts:
        raise ValueError("Simulator metadata snapshot does not contain any CoreArrayContext.")
    return snapshot.core_array_contexts[0]


def get_num_workers(core_array_context: Any) -> int | None:
    value = core_array_context.global_kwargs.get("num_workers")
    return int(value) if value is not None else None


def get_num_layers(core_array_context: Any) -> int:
    value = core_array_context.global_kwargs.get("num_layers")
    return int(value) if value is not None else 1


__all__ = ["get_num_layers", "get_num_workers", "get_primary_context", "is_edge_system_config"]
