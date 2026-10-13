#!/usr/bin/env python3
"""Run a child command and collect outer CPU/resource metadata without changing its timers."""
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[1] != "--metadata":
        print("usage: run_with_usage.py --metadata OUTPUT.json -- COMMAND [ARG ...]", file=sys.stderr)
        return 2
    output = Path(sys.argv[2])
    if sys.argv[3] != "--" or len(sys.argv) < 5:
        print("missing -- COMMAND", file=sys.stderr)
        return 2
    command = sys.argv[4:]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter_ns()
    completed = subprocess.run(command, check=False)
    elapsed = time.perf_counter_ns() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    # ru_maxrss is KiB on Linux. RUSAGE_CHILDREN also accumulates the waited child process CPU.
    record = {
        "command": command,
        "exit_code": completed.returncode,
        "outer_elapsed_ns": elapsed,
        "direct_child_user_cpu_seconds_delta": after.ru_utime - before.ru_utime,
        "direct_child_system_cpu_seconds_delta": after.ru_stime - before.ru_stime,
        "cumulative_children_max_rss_kib": after.ru_maxrss,
        "rss_units": "KiB on Linux",
        "scope_note": "outer process resource metadata; probe-internal caller timing is unchanged and is in raw JSON",
    }
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
