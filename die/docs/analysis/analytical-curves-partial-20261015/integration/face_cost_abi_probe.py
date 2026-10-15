#!/usr/bin/env python3
"""Exercise the FACE cost query against one real coarse DIE shared library.

This probe imports the selected FACE checkout's checked-in Python sources,
loads the supplied shared library with ctypes, and drives both the oracle and
the actual scheduler selection. It distinguishes the additive cost-only C
symbol from the established full-report fallback without mocking the library.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path


REQUIRED_OLD_SYMBOLS = (
    "die_coarse_create_model_v2",
    "die_coarse_estimate_batch_v2",
    "die_coarse_destroy_model_v2",
    "die_coarse_free_result_v2",
)
COST_SYMBOL = "die_coarse_estimate_cost_batch_v2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--expect", choices=("new", "old"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    library = args.library.resolve()
    if not library.is_file():
        raise SystemExit(f"coarse DIE library does not exist: {library}")
    workload = root / "sh_test_mesh/workload/llama2_7b_inference"
    online = workload / "online"
    if not workload.is_dir() or not online.is_dir():
        raise SystemExit(f"FACE workload sources are missing under {root}")

    # Ensure imports read the exact selected checkout while keeping this
    # validation from leaving bytecode in a source/vendor tree.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(workload))
    sys.path.insert(0, str(online))
    from face_scheduler import (  # noqa: E402
        FaceHardware,
        FaceModel,
        estimate_iteration_time_ns,
    )
    from online.die_model_oracle import DieModelOracle  # noqa: E402

    exported = ctypes.CDLL(str(library))
    missing = [name for name in REQUIRED_OLD_SYMBOLS if not hasattr(exported, name)]
    if missing:
        raise SystemExit(f"library is not a usable coarse v2 ABI: missing {missing}")
    has_cost = hasattr(exported, COST_SYMBOL)
    if args.expect == "new" and not has_cost:
        raise SystemExit(f"expected new ABI symbol {COST_SYMBOL} in {library}")
    if args.expect == "old" and has_cost:
        raise SystemExit(f"expected legacy coarse ABI without {COST_SYMBOL}: {library}")

    config = root / "extern/die_model/coarse/configs/example_hardware.json"
    tasks_path = root / "extern/die_model/coarse/configs/example_tasks.json"
    tasks_doc = json.loads(tasks_path.read_text(encoding="utf-8"))
    task = next(
        row for row in tasks_doc["specs"]
        if isinstance(row, dict) and row.get("schema") == "coarse_die_task_v2"
    )
    oracle = DieModelOracle(lib_path=str(library), config_json_path=str(config))
    try:
        if not oracle.coarse:
            raise AssertionError("example hardware did not select coarse_die_v2")
        if oracle.supports_estimate_cost != has_cost:
            raise AssertionError("ctypes symbol probe and FACE oracle disagree")

        full = oracle.estimate([task])
        if full[0] is None:
            raise AssertionError(f"full coarse estimate failed: {oracle.last_errors}")
        selected = oracle.estimate_cost([task])
        if selected[0] is None:
            raise AssertionError(f"FACE estimate_cost failed: {oracle.last_errors}")
        if args.expect == "new":
            row = selected[0]
            if set(row) != {
                "estimated_latency_ns", "plan_hash", "status", "total_seconds"
            }:
                raise AssertionError("new cost-only row has unexpected fields")
            if row["estimated_latency_ns"] != full[0]["estimated_latency_ns"]:
                raise AssertionError("cost-only latency differs from full report")
            if row["plan_hash"] != full[0]["plan_fingerprint"]:
                raise AssertionError("cost-only Plan hash differs from full report")
            if row["status"] != full[0]["status"]:
                raise AssertionError("cost-only status differs from full report")
        elif selected[0] != full[0]:
            raise AssertionError("legacy cost query did not preserve full report")

        # Observe the private dispatch argument while leaving the real method
        # and C ABI calls intact. The scheduler must select only the new
        # endpoint when present; otherwise it calls estimate(), which retains
        # the legacy full response.
        calls: list[bool] = []
        original = oracle._estimate

        def observed(specs, *, cost_only):
            calls.append(bool(cost_only))
            return original(specs, cost_only=cost_only)

        oracle._estimate = observed
        scheduler_ns = estimate_iteration_time_ns(
            FaceHardware(3, 2, 100, 1.0, 2.0, 1.0, 0, 0),
            FaceModel(1, 8, 16, 4, 32, 2, "gelu"),
            instance_size=2,
            p_chunk=8,
            d_batch=1,
            d_token=257,
            cost_oracle=oracle,
        )
        expected_calls = [args.expect == "new"]
        if calls != expected_calls:
            raise AssertionError(
                f"FACE scheduler selected wrong endpoint: {calls}, expected {expected_calls}"
            )
        if scheduler_ns <= 0 or oracle.last_errors != [None] * 7:
            raise AssertionError(
                f"FACE scheduler estimate failed: ns={scheduler_ns}, errors={oracle.last_errors}"
            )

        result = {
            "schema": "face_cost_abi_probe_v1",
            "variant": root.name,
            "expected_api": args.expect,
            "coarse_library": str(library),
            "coarse_library_sha256": sha256(library),
            "cost_only_symbol_exported": has_cost,
            "oracle_supports_estimate_cost": oracle.supports_estimate_cost,
            "selected_example_task": task,
            "full_estimate_latency_ns": full[0]["estimated_latency_ns"],
            "cost_query_latency_ns": selected[0]["estimated_latency_ns"],
            "cost_query_hash": selected[0].get("plan_hash"),
            "scheduler_dispatch_cost_only_flags": calls,
            "scheduler_estimate_ns": scheduler_ns,
            "source_sha256": {
                "face_scheduler.py": sha256(workload / "face_scheduler.py"),
                "online/die_model_oracle.py": sha256(online / "die_model_oracle.py"),
            },
        }
    finally:
        oracle.close()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "variant": result["variant"],
        "expected_api": result["expected_api"],
        "cost_only_symbol_exported": result["cost_only_symbol_exported"],
        "scheduler_dispatch_cost_only_flags": result["scheduler_dispatch_cost_only_flags"],
        "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
