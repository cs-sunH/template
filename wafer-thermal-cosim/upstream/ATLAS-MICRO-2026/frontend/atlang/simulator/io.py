"""Shared YAML and output helpers for atlang extraction paths."""

from __future__ import annotations

import contextlib
import multiprocessing
import numbers
import os
import signal
import sys
from typing import Any, Callable, Iterator

import yaml


# Plain YAML conversion

def to_yaml_plain(value: Any) -> Any:
    if value is None:
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, dict):
        return {str(key): to_yaml_plain(item_value) for key, item_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_yaml_plain(item_value) for item_value in value]
    return str(value)


def safe_dump_plain(data: Any, output_file, *, sort_keys: bool = False) -> None:
    yaml.safe_dump(to_yaml_plain(data), output_file, sort_keys=sort_keys)


# Process utilities

@contextlib.contextmanager
def redirect_process_output(log_file_path: str) -> Iterator[None]:
    os.makedirs(os.path.dirname(log_file_path) or ".", exist_ok=True)
    with open(log_file_path, "w") as log_file:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        try:
            os.dup2(log_file.fileno(), 1)
            os.dup2(log_file.fileno(), 2)
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


def _pool_worker_initializer(initializer: Callable[[], None] | None) -> None:
    if initializer is not None:
        initializer()
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def run_pool_starmap_interruptible(
    worker_fn: Callable[..., Any],
    work_items: list[tuple[Any, ...]],
    *,
    processes: int,
    initializer: Callable[[], None] | None = None,
    start_method: str = "fork",
    timeout_s: float = 0.2,
    chunksize: int | None = None,
) -> list[Any]:
    if not work_items:
        return []
    ctx = multiprocessing.get_context(start_method)
    pool = ctx.Pool(processes, initializer=_pool_worker_initializer, initargs=(initializer,))
    async_result = pool.starmap_async(worker_fn, work_items, chunksize=chunksize)
    try:
        while True:
            try:
                results = async_result.get(timeout=timeout_s)
                pool.close()
                pool.join()
                return results
            except multiprocessing.TimeoutError:
                continue
    except BaseException:
        pool.terminate()
        pool.join()
        raise


__all__ = ["redirect_process_output", "run_pool_starmap_interruptible", "safe_dump_plain", "to_yaml_plain"]
