"""Shared DRAM task normalization helpers for atlang extraction."""

from __future__ import annotations

from typing import Any


def finalize_dram_tasks(dram_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated_tasks: list[dict[str, Any]] = []
    seen_keys: set[tuple[Any, ...]] = set()
    for dram_task in dram_tasks:
        dedup_key = (
            dram_task["name"],
            bool(dram_task["is_write"]),
            int(dram_task["init_iter"]),
            int(dram_task["stride_iter"]),
            int(dram_task["total_iter"]),
            tuple(int(value) for value in dram_task["access_base"]),
            tuple(int(value) for value in dram_task["access_extent"]),
        )
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        deduplicated_tasks.append(dram_task)
    return sorted(
        deduplicated_tasks,
        key=lambda task: (
            int(task["init_iter"]),
            task["name"],
            bool(task["is_write"]),
            tuple(int(value) for value in task["access_base"]),
            tuple(int(value) for value in task["access_extent"]),
        ),
    )


__all__ = ["finalize_dram_tasks"]
